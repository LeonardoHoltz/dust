"""DUST's estimator on a small CNN with one loss per image, trained on MNIST.

The transformer recipe rewards each token's jitter with that token's own loss. A CNN classifier has no
tokens: one scalar loss per image has to credit every channel at every pixel of a layer's output. What
carries over from dust.DUST:
  1. A clean forward caches each Conv2d/Linear input and output and the per-image losses.
  2. Direct draws jitter one layer's output, y + sigma * a with a ~ N(0, I), rerun the rest of the
     network, and reward the jitter with the image's loss reduction, centered across each chunk of
     draws. The reward-weighted noise, averaged over draws, is the estimated error at that output.
     Only the layers after the jittered one are rerun, so the head (whose suffix is just the loss) is
     cheap and gets many draws, like DUST's logit-slab jitter.
  3. The weight gradient is the exact local linear map of that error: error times input for a Linear,
     and the same correlation over every kernel position (unfold) for a Conv2d. Biases sum the error.
     Then one SGD step. No autograd or backward pass is used.
Weight sharing is what makes this workable: a conv kernel's gradient sums the error estimate over
every pixel of every image, which averages its noise away.
"""
import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vision.mnist import load_mnist, train


def make_model(device, seed):
    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Conv2d(1, 16, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),     # 16 x 14 x 14
        nn.Conv2d(16, 32, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),    # 32 x 7 x 7
        nn.Flatten(), nn.Linear(32 * 7 * 7, 128), nn.ReLU(), nn.Linear(128, 10),
    )
    return model.to(device).requires_grad_(False)


@dataclass
class Recipe:
    """Draws per update for each layer, in model order (conv1, conv2, fc1, head)."""
    draws: list[int]
    sigma: float
    nrep: int
    lr: float
    momentum: float


def cnn_recipe(population, lr, momentum, sigma=0.05, nrep=32):
    """population counts the draws for the layers that need a rerun; the head gets 4x on top, as in DUST."""
    draws = [round(population * f / nrep) * nrep for f in (0.375, 0.375, 0.25)]
    return Recipe(draws + [4 * population], sigma, nrep, lr, momentum)


class CNNDust:
    def __init__(self, model, settings, seed=42):
        self.model, self.settings = model, settings
        self.device = next(model.parameters()).device
        self.generator = torch.Generator(self.device).manual_seed(seed + 3)
        self.sites = [i for i, m in enumerate(model) if isinstance(m, (nn.Conv2d, nn.Linear))]
        if len(self.sites) != len(settings.draws) or any(k % settings.nrep for k in settings.draws):
            raise ValueError('Need one draw count per layer, each a multiple of nrep.')

    def run(self, h, start, y):
        """Per-image losses of the network from module `start` on, given its input h."""
        for module in self.model[start:]:
            h = module(h)
        return F.cross_entropy(h, y.repeat(len(h) // len(y)), reduction='none')

    def capture(self, x, y):
        self.inputs, self.outputs = {}, {}
        h = x
        for i, module in enumerate(self.model):
            if i in self.sites:
                self.inputs[i] = h
                self.outputs[i] = h = module(h)
            else:
                h = module(h)
        self.clean_loss = F.cross_entropy(h, y, reduction='none')

    def direct_error(self, site, y, draws):
        """One-sided loss drops at one layer's output, centered across each chunk of draws."""
        n, sigma = self.settings.nrep, self.settings.sigma
        out = self.outputs[site]
        error = torch.zeros_like(out)
        for _ in range(draws // n):
            a = torch.randn((n, *out.shape), device=self.device, generator=self.generator)
            perturbed = (out.unsqueeze(0) + sigma * a).flatten(0, 1)
            loss = self.run(perturbed, site + 1, y).view(n, -1)
            reward = self.clean_loss.unsqueeze(0) - loss
            reward = reward - reward.mean(0, keepdim=True)
            error -= torch.einsum('nb,nb...->b...', reward, a) / sigma
        return error / draws

    def reassemble(self, site, error):
        """The exact local linear map from output error to weight and bias gradients."""
        module, x = self.model[site], self.inputs[site]
        if isinstance(module, nn.Linear):
            weight = torch.einsum('bo,bi->oi', error, x)
            return weight, error.sum(0)
        patches = F.unfold(x, module.kernel_size, padding=module.padding)        # [b, in * k * k, pixels]
        weight = torch.einsum('bol,bil->oi', error.flatten(2), patches)
        return weight.view_as(module.weight), error.sum((0, 2, 3))

    @torch.no_grad()
    def step(self, x, y, optimizer):
        self.capture(x, y)
        optimizer.zero_grad(set_to_none=True)
        for site, draws in zip(self.sites, self.settings.draws):
            weight, bias = self.reassemble(site, self.direct_error(site, y, draws))
            # Dividing the summed estimates by the batch size gives the gradient of the mean loss.
            self.model[site].weight.grad = weight / len(y)
            self.model[site].bias.grad = bias / len(y)
        optimizer.step()
        return self.clean_loss.mean().item()


class WeightES:
    """Conventional evolution strategy, for comparison: one draw jitters every weight of the network at once,
    theta + sigma * e, and is scored by the whole batch's mean loss. Draws come in antithetic pairs (+e, -e);
    the loss difference times e, averaged over pairs, estimates the gradient. population counts forward
    passes, so it is twice the number of pairs.
    """

    def __init__(self, model, population, sigma, nrep=32, seed=42):
        if population % (2 * nrep):
            raise ValueError('population must be a multiple of 2 * nrep.')
        self.model, self.population, self.sigma, self.nrep = model, population, sigma, nrep
        self.device = next(model.parameters()).device
        self.generator = torch.Generator(self.device).manual_seed(seed + 3)
        self.batched_loss = torch.func.vmap(self.loss, in_dims=(0, None, None))

    def loss(self, parameters, x, y):
        return F.cross_entropy(torch.func.functional_call(self.model, parameters, (x,)), y)

    @torch.no_grad()
    def step(self, x, y, optimizer):
        loss = F.cross_entropy(self.model(x), y).item()  # before the update, as the other methods report it
        parameters = dict(self.model.named_parameters())
        estimate = {n: torch.zeros_like(p) for n, p in parameters.items()}
        for _ in range(self.population // (2 * self.nrep)):
            noise = {
                n: torch.randn((self.nrep, *p.shape), device=self.device, generator=self.generator)
                for n, p in parameters.items()
            }
            plus = self.batched_loss({n: p + self.sigma * noise[n] for n, p in parameters.items()}, x, y)
            minus = self.batched_loss({n: p - self.sigma * noise[n] for n, p in parameters.items()}, x, y)
            weight = (plus - minus) / (2 * self.sigma)
            for n, e in noise.items():
                estimate[n] += torch.einsum('k,k...->...', weight, e)
        optimizer.zero_grad(set_to_none=True)
        for n, p in parameters.items():
            p.grad = estimate[n] / (self.population // 2)
        optimizer.step()
        return loss


def backprop_step(model, optimizer, x, y):
    model.requires_grad_(True)
    optimizer.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    optimizer.step()
    model.requires_grad_(False)
    return loss.item()


@torch.no_grad()
def evaluate(model, split, batch_size=1000):
    images, labels = split
    loss, correct = 0.0, 0
    for x, y in zip(images.split(batch_size), labels.split(batch_size)):
        logits = model(x)
        loss += F.cross_entropy(logits, y, reduction='sum').item()
        correct += (logits.argmax(-1) == y).sum().item()
    return dict(loss=loss / len(labels), accuracy=correct / len(labels))


def gradient_check(model, estimator, x, y):
    """Cosine and norm ratio between the estimate and the backprop gradient, per parameter."""
    model.requires_grad_(True)
    F.cross_entropy(model(x), y).backward()
    exact = {n: p.grad.clone() for n, p in model.named_parameters()}
    model.requires_grad_(False)
    estimator.step(x, y, torch.optim.SGD(model.parameters(), lr=0.0))
    print(f"{'parameter':12s} {'cosine':>7s} {'|est|/|bp|':>11s}")
    for name, parameter in model.named_parameters():
        a, b = parameter.grad.flatten(), exact[name].flatten()
        cosine = F.cosine_similarity(a, b, dim=0).item()
        print(f'{name:12s} {cosine:7.3f} {a.norm().item() / b.norm().item():11.3f}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=('dust', 'es', 'backprop'), default='dust')
    parser.add_argument('--population', type=int, default=256)
    parser.add_argument(
        '--sigma', type=float,
        help='Jitter scale: on layer outputs for dust (default 0.05), on weights for es (default 0.005).',
    )
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=float, default=1.0)
    parser.add_argument('--eval-interval', type=int, default=50)
    parser.add_argument('--data-dir', type=Path, default=Path('data/mnist'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--check', action='store_true', help='Only compare the estimate with backprop on one batch.')
    parser.add_argument('--checkpoint', type=Path, help='With --check: measure at a trained model.')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        parser.error('Training requires a CUDA GPU.')
    if args.population % 256:
        parser.error('--population must be a multiple of 256.')
    device = torch.device('cuda', 0)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = False  # rewards are small loss differences; keep them in full fp32
    train_split, validation, test = [(x.unsqueeze(1), y) for x, y in load_mnist(args.data_dir, device)]
    model = make_model(device, args.seed)
    if args.sigma is None:
        args.sigma = 0.005 if args.method == 'es' else 0.05
    settings = cnn_recipe(args.population, args.lr, args.momentum, args.sigma)
    estimator = {
        'dust': lambda seed: CNNDust(model, settings, seed),
        'es': lambda seed: WeightES(model, args.population, args.sigma, seed=seed),
        'backprop': lambda seed: None,
    }[args.method]
    if args.check:
        if args.method == 'backprop':
            parser.error('--check compares dust or es with backprop.')
        if args.checkpoint:
            model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
        return gradient_check(
            model, estimator(1), train_split[0][:args.batch_size], train_split[1][:args.batch_size],
        )
    if args.output is None:
        parser.error('--output is required for training.')
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    estimator = estimator(args.seed)
    step = (
        (lambda x, y: estimator.step(x, y, optimizer)) if estimator
        else (lambda x, y: backprop_step(model, optimizer, x, y))
    )
    train(model, step, evaluate, (train_split, validation, test), args, dict(
        checkpoint=None, recipe=asdict(settings) if args.method == 'dust' else None,
    ))


if __name__ == '__main__':
    main()
