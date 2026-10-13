"""Self-check: given the exact error at a layer's output, reassemble must return autograd's weight gradient.

DUST's noise hides a wrong local map (training just gets worse), so this feeds the exact backprop error
in instead of an estimate. Runs on CPU in a few seconds: python vision/test_reassemble.py
"""
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dust import DUST
from vision.cnn import CNNDust, cnn_recipe, make_model as make_cnn
from vision.mnist import mnist_recipe, per_token
from vision.model import PatchGPTConfig, make_model as make_patch_gpt


def test_cnn():
    model = make_cnn('cpu', seed=0).requires_grad_(True)
    x, y = torch.randn(4, 1, 28, 28), torch.randint(0, 10, (4,))
    estimator = CNNDust(model, cnn_recipe(256, lr=0.0, momentum=0.0), seed=0)
    estimator.capture(x, y)  # outside step(), so the cached outputs stay in the autograd graph
    loss = estimator.clean_loss.sum()
    errors = torch.autograd.grad(loss, [estimator.outputs[s] for s in estimator.sites], retain_graph=True)
    loss.backward()
    for site, error in zip(estimator.sites, errors):
        weight, bias = estimator.reassemble(site, error)
        torch.testing.assert_close(weight, model[site].weight.grad, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(bias, model[site].bias.grad, rtol=1e-4, atol=1e-5)


def test_dust():
    config = PatchGPTConfig()
    model = make_patch_gpt(config, 'cpu', seed=0).requires_grad_(True)
    x, y = torch.randn(2, config.sequence_len, config.patch_size ** 2), per_token(torch.tensor([3, 7]), 49)
    estimator = DUST(model, mnist_recipe(256, lr=0.0, momentum=0.0), seed=0)
    try:
        estimator.capture(x, y)
        sites = ['transformer.wte', 'transformer.h.0.mlp.c_fc', 'transformer.h.3.attn.c_proj', 'value_embeds.1', 'lm_head']
        loss = estimator.clean_loss.sum()
        errors = torch.autograd.grad(loss, [estimator.outputs[s] for s in sites], retain_graph=True)
        loss.backward()
        parameters = dict(model.named_parameters())
        for site, error in zip(sites, errors):
            torch.testing.assert_close(
                estimator.reassemble(site, error.float()), parameters[site + '.weight'].grad, rtol=1e-4, atol=1e-5,
            )
    finally:
        estimator.close()


if __name__ == '__main__':
    test_cnn()
    test_dust()
    print('reassemble matches autograd for the CNN and the patch transformer')
