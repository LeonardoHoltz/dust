"""Activation maximization: for each chosen neuron, find the 28x28 image that drives it highest.

The image's pixels are the only thing optimized (Adam on the input, through the trained, frozen model);
this is an analysis tool, so it uses autograd regardless of how the model was trained. Pixels stay in
[0, 1], the image is shifted by up to a pixel at random every step so that the result is not tuned to
exact pixel positions, and a small total-variation penalty discourages pixel noise. Every model starts
from the same image, so differences between rows come from the models.

Neurons, as layer:index[-index]:
  cnn    conv1:c, conv2:c   a conv channel's pre-ReLU unit at the center of its map
         fc1:u               a hidden unit of the 128-wide layer, pre-ReLU
         logit:k             the score of class k
  patch  mlp<block>:u        an MLP hidden unit (mlp.c_fc output, pre-activation) at the last patch
         logit:k             the score of class k at the last patch, before the soft cap

Example:
  python vision/visualize.py --arch cnn --neurons logit:0-9 \\
      --models backprop=runs/cnn-bp/best.pt dust256=runs/cnn-dust-p256/best.pt --output runs/viz/cnn-logits
"""
import argparse
import json
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vision.mnist import MEAN, STD

SCALE, PAD, LABEL_WIDTH, HEADER = 4, 4, 90, 18


def parse_neurons(spec):
    neurons = []
    for item in spec:
        layer, index = item.split(':')
        lo, _, hi = index.partition('-')
        neurons += [(layer, i) for i in range(int(lo), int(hi or lo) + 1)]
    return neurons


def load_model(arch, path, device):
    if arch == 'cnn':
        from vision.cnn import make_model
        model = make_model(device, seed=0)
    else:
        from vision.model import PatchGPTConfig, make_model
        model = make_model(PatchGPTConfig(), device, seed=0)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model.eval().requires_grad_(False)


def site(arch, model, layer):
    """The module whose output holds the layer's neurons, and how to read one neuron from it."""
    if arch == 'cnn':
        index = {'conv1': 0, 'conv2': 3, 'fc1': 7, 'logit': 9}[layer]
        if layer.startswith('conv'):
            # A channel's mean over the map is nearly linear in the pixels (conv1 exactly), so it is maximized by
            # a flat image; the center unit shows the pattern within its receptive field instead.
            read = lambda out, i: out[:, i, out.shape[2] // 2, out.shape[3] // 2]
        else:
            read = lambda out, i: out[:, i]
        return model[index], read
    read = lambda out, i: out[:, -1, i]
    if layer == 'logit':
        return model.lm_head, read
    return model.transformer.h[int(layer.removeprefix('mlp'))].mlp.c_fc, read


def forward(arch, model, pixels):
    """Logits for [n, 28, 28] pixels in [0, 1]."""
    x = (pixels - MEAN) / STD
    if arch == 'cnn':
        return model(x.unsqueeze(1))
    from vision.model import patchify
    return model(patchify(x, model.config.patch_size))[:, -1]


def total_variation(pixels):
    return (pixels[:, 1:] - pixels[:, :-1]).abs().mean() + (pixels[:, :, 1:] - pixels[:, :, :-1]).abs().mean()


def maximize(arch, model, neurons, steps, lr, tv, seed, device):
    """Optimize one image per neuron, all in one batch; returns the images and their final activations."""
    generator = torch.Generator(device).manual_seed(seed)
    # Near-black, like an MNIST background; a flat grey start leaves every first-layer ReLU of some models off.
    pixels = 0.05 * torch.rand(len(neurons), 28, 28, device=device, generator=generator)
    pixels.requires_grad_(True)
    optimizer = torch.optim.Adam([pixels], lr=lr)
    captured = {}
    hooks = [
        module.register_forward_hook(lambda m, a, out, layer=layer: captured.__setitem__(layer, out))
        for layer, (module, _) in {l: site(arch, model, l) for l, _ in neurons}.items()
    ]

    def activations(images):
        captured.clear()
        forward(arch, model, images)
        return torch.stack([
            site(arch, model, layer)[1](captured[layer][n:n + 1], i).squeeze(0)
            for n, (layer, i) in enumerate(neurons)
        ])

    try:
        for _ in range(steps):
            dy, dx = torch.randint(-1, 2, (2,), generator=generator, device=device).tolist()
            shifted = torch.roll(pixels, (dy, dx), (1, 2))
            loss = -activations(shifted).sum() + tv * len(neurons) * total_variation(pixels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                pixels.clamp_(0, 1)
        with torch.no_grad():
            final = activations(pixels)
            probabilities = F.softmax(forward(arch, model, pixels).float(), -1)
    finally:
        for hook in hooks:
            hook.remove()
    return pixels.detach(), final, probabilities


def save_grid(path, rows, neurons):
    """rows: [(label, [n, 28, 28] images)] -> one PNG, models down, neurons across."""
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.load_default(size=12)
    cell = 28 * SCALE
    width = LABEL_WIDTH + len(neurons) * (cell + PAD)
    height = HEADER + len(rows) * (cell + PAD)
    canvas = Image.new('L', (width, height), 255)
    draw = ImageDraw.Draw(canvas)
    for n, (layer, i) in enumerate(neurons):
        draw.text((LABEL_WIDTH + n * (cell + PAD) + 2, 2), f'{layer}:{i}', fill=0, font=font)
    for r, (label, images) in enumerate(rows):
        top = HEADER + r * (cell + PAD)
        draw.text((4, top + cell // 2 - 6), label, fill=0, font=font)
        for n, image in enumerate(images):
            tile = Image.fromarray((image.cpu().numpy() * 255).round().astype('uint8'), 'L')
            canvas.paste(tile.resize((cell, cell), Image.NEAREST), (LABEL_WIDTH + n * (cell + PAD), top))
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--arch', choices=('cnn', 'patch'), required=True)
    parser.add_argument('--models', nargs='+', required=True, help='label=checkpoint pairs, one row each.')
    parser.add_argument('--neurons', nargs='+', required=True, help='e.g. logit:0-9 or conv2:0-7')
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--tv', type=float, default=0.3, help='Weight of the total-variation penalty.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', type=Path, required=True, help='Writes <output>.png and <output>.json.')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.allow_tf32 = False
    neurons = parse_neurons(args.neurons)
    rows, report = [], {}
    for item in args.models:
        label, path = item.split('=', 1)
        model = load_model(args.arch, Path(path), device)
        images, final, probabilities = maximize(
            args.arch, model, neurons, args.steps, args.lr, args.tv, args.seed, device,
        )
        rows.append((label, images))
        report[label] = [
            dict(neuron=f'{layer}:{i}', activation=round(a, 3), predicted=p.argmax().item(),
                 confidence=round(p.max().item(), 3))
            for (layer, i), a, p in zip(neurons, final.tolist(), probabilities)
        ]
        print(label, ' '.join(f"{r['neuron']}={r['activation']:.2f}->{r['predicted']}" for r in report[label]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_grid(args.output.with_suffix('.png'), rows, neurons)
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
