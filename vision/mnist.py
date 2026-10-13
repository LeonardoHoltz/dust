"""Train the patch transformer (vision/model.py) on MNIST with DUST or with backpropagation SGD.

The DUST estimator is the unchanged dust.DUST; only the per-layer draw allocation (a 4-block recipe) and
the data are new. Accuracy is read at the last patch, which has seen the whole image. The last 5,000
training images are held out for checkpoint selection; the 10,000 test images are scored once at the end.
"""
import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dust import DUST, Recipe
from vision.model import PatchGPTConfig, make_model, patchify

MEAN, STD = 0.1307, 0.3081
VALIDATION_IMAGES = 5000


def mnist_recipe(population, lr, momentum, nrep=8):
    """Draws per update for 4 blocks; population ~ direct-loss draws, in multiples of 256."""
    scale = population // 256
    writers = [v * scale for v in [40, 32, 24, 16]]
    return Recipe(
        writers=writers, hidden=writers.copy(), hubs=[v * scale for v in [8, 8, 8, 8]],
        embedding=16 * scale, head=4 * population,
        local={k: v * scale for k, v in dict(q=8, v=24, gate=8, k=16, ve=8).items()},
        nrep=nrep, lr=lr, momentum=momentum,
    )


def load_mnist(root, device):
    """Normalized [N, 28, 28] images and labels: train, validation (last 5,000 train images) and test."""
    from torchvision.datasets import MNIST

    splits = {}
    for train in (True, False):
        data = MNIST(root, train=train, download=True)
        splits[train] = ((data.data.float() / 255 - MEAN) / STD).to(device), data.targets.to(device)
    (x, y), test = splits[True], splits[False]
    return (x[:-VALIDATION_IMAGES], y[:-VALIDATION_IMAGES]), (x[-VALIDATION_IMAGES:], y[-VALIDATION_IMAGES:]), test


def per_token(labels, length):
    """Every patch position is trained to predict the image's label."""
    return labels[:, None].expand(-1, length).contiguous()


@torch.no_grad()
def evaluate(model, split, batch_size=1000):
    images, labels = split
    loss, correct = 0.0, 0
    context = torch.autocast('cuda', dtype=torch.bfloat16) if images.is_cuda else nullcontext()
    for x, y in zip(images.split(batch_size), labels.split(batch_size)):
        with context:
            logits = model(x)[:, -1]
        loss += torch.nn.functional.cross_entropy(logits, y, reduction='sum').item()
        correct += (logits.argmax(-1) == y).sum().item()
    return dict(loss=loss / len(labels), accuracy=correct / len(labels))


def make_optimizer(model, lr, momentum):
    groups = {}
    for name, parameter in model.named_parameters():
        groups.setdefault(0.03 if 'lambdas' in name else lr, []).append(parameter)
    return torch.optim.SGD([{'params': ps, 'lr': g} for g, ps in groups.items()], momentum=momentum)


def backprop_step(model, optimizer, x, y):
    model.requires_grad_(True)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss = model(x, y)
    loss.backward()
    optimizer.step()
    model.requires_grad_(False)
    return loss.item()


def train(model, step, evaluate, splits, args, extra):
    """Shared by both image scripts: random batches, periodic validation, best checkpoint, one test score.

    step(images, labels) runs one update and returns the training loss; extra goes into config.json.
    """
    train_split, validation, test = splits
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f'Choose an empty output directory: {args.output}')
    steps = round(args.epochs * (len(train_split[1]) // args.batch_size))
    args.output.mkdir(parents=True, exist_ok=True)
    parameters = sum(p.numel() for p in model.parameters())
    metadata = {
        **vars(args), 'data_dir': str(args.data_dir), 'output': str(args.output), 'steps': steps,
        'parameters': parameters, 'torch_version': torch.__version__, **extra,
    }
    (args.output / 'config.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f"{args.method}: {steps} steps, {parameters:,} parameters", flush=True)

    device = train_split[0].device
    generator = torch.Generator(device).manual_seed(args.seed + 1)
    best = evaluate(model, validation)
    torch.save(model.state_dict(), args.output / 'best.pt')
    start = time.monotonic()
    order = torch.empty(0, dtype=torch.long, device=device)
    for step_index in range(1, steps + 1):
        if len(order) < args.batch_size:
            order = torch.randperm(len(train_split[1]), device=device, generator=generator)
        index, order = order[:args.batch_size], order[args.batch_size:]
        loss = step(train_split[0][index], train_split[1][index])
        if not math.isfinite(loss):
            raise FloatingPointError(f'Non-finite training loss at step {step_index}')
        record = dict(step=step_index, train_loss=loss, elapsed_seconds=time.monotonic() - start)
        if step_index % args.eval_interval == 0 or step_index == steps:
            record['val'] = evaluate(model, validation)
            if record['val']['loss'] < best['loss']:
                best = record['val']
                torch.save(model.state_dict(), args.output / 'best.pt')
            print(json.dumps(record), flush=True)
        with (args.output / 'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(record) + '\n')

    model.load_state_dict(torch.load(args.output / 'best.pt', map_location=device, weights_only=True))
    result = dict(best_val=best, test_at_best_val=evaluate(model, test), seconds=time.monotonic() - start)
    (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(f"Best validation {best}; test {result['test_at_best_val']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=('dust', 'backprop'), default='dust')
    parser.add_argument('--population', type=int, default=256, help='DUST direct-loss draws; a multiple of 256.')
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=float, default=1.0)
    parser.add_argument('--eval-interval', type=int, default=50)
    parser.add_argument('--data-dir', type=Path, default=Path('data/mnist'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        parser.error('Training requires a CUDA GPU.')
    if args.population % 256:
        parser.error('--population must be a multiple of 256.')
    device = torch.device('cuda', 0)
    torch.set_float32_matmul_precision('high')

    config = PatchGPTConfig()
    splits = [(patchify(x, config.patch_size), y) for x, y in load_mnist(args.data_dir, device)]
    model = make_model(config, device, args.seed)
    optimizer = make_optimizer(model, args.lr, args.momentum)
    settings = mnist_recipe(args.population, args.lr, args.momentum)
    estimator = DUST(model, settings, args.seed) if args.method == 'dust' else None

    def step(x, labels):
        y = per_token(labels, config.sequence_len)
        return estimator.step(x, y, optimizer) if estimator else backprop_step(model, optimizer, x, y)

    try:
        train(model, step, evaluate, splits, args, dict(
            model=asdict(config), recipe=asdict(settings) if estimator else None,
        ))
    finally:
        if estimator:
            estimator.close()


if __name__ == '__main__':
    main()
