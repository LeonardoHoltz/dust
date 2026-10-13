"""Compare DUST's gradient estimate with the backprop gradient on one MNIST batch, per parameter.

Cosine similarity says whether the estimate points the right way; the norm ratio whether its scale is
right. The text recipe's estimate is a biased per-token approximation, so cosines well below 1 are
expected; what matters is that they are clearly positive for every group.
"""
import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dust import DUST
from vision.mnist import load_mnist, mnist_recipe, per_token
from vision.model import PatchGPTConfig, make_model, patchify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--population', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--checkpoint', type=Path, help='Measure at a trained model instead of at init.')
    parser.add_argument('--data-dir', type=Path, default=Path('data/mnist'))
    args = parser.parse_args()
    device = torch.device('cuda', 0)

    config = PatchGPTConfig()
    (images, labels), _, _ = load_mnist(args.data_dir, device)
    x = patchify(images[:args.batch_size], config.patch_size)
    y = per_token(labels[:args.batch_size], config.sequence_len)
    model = make_model(config, device, seed=0)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))

    model.requires_grad_(True)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        model(x, y).backward()
    exact = {n: p.grad.clone() for n, p in model.named_parameters()}
    model.requires_grad_(False)

    estimator = DUST(model, mnist_recipe(args.population, lr=0.0, momentum=0.0), seed=1)
    estimator.step(x, y, torch.optim.SGD(model.parameters(), lr=0.0))
    estimator.close()

    print(f"{'parameter':42s} {'cosine':>7s} {'|dust|/|bp|':>11s}")
    for name, parameter in model.named_parameters():
        a, b = parameter.grad.flatten().float(), exact[name].flatten().float()
        cosine = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
        print(f'{name:42s} {cosine:7.3f} {a.norm().item() / b.norm().item():11.3f}')


if __name__ == '__main__':
    main()
