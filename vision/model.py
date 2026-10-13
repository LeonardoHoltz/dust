"""An image classifier DUST can train: the GPT blocks of ../model.py read a raster of image patches.

DUST rewards each token's jitter with the loss change at that same token, so the model needs a loss at
every token. Patches are read in raster order under causal attention and every position predicts the
image's class from the patches seen so far; the last position, which has seen the whole image, is the
prediction. Everything else is the text model: the patch projection takes the token embedding's place
(transformer.wte, now a Linear), value embeddings are indexed by patch position instead of token ID,
and the head is a 10-class classifier.
"""
from dataclasses import dataclass
import math
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import GPT, GPTConfig, has_ve, norm


@dataclass
class PatchGPTConfig(GPTConfig):
    image_size: int = 28
    patch_size: int = 4
    sequence_len: int = 49     # (image_size / patch_size) ** 2 patches
    vocab_size: int = 10       # classes
    n_layer: int = 4
    n_head: int = 4
    n_kv_head: int = 4
    n_embd: int = 128
    window_pattern: str = 'L'

    def __post_init__(self):
        assert self.image_size % self.patch_size == 0
        assert self.sequence_len == (self.image_size // self.patch_size) ** 2


def patchify(images, patch_size):
    """[batch, H, W] -> [batch, patches, patch_size ** 2], patches in raster order."""
    b, h, w = images.shape
    p = patch_size
    patches = images.view(b, h // p, p, w // p, p).permute(0, 1, 3, 2, 4)
    return patches.reshape(b, (h // p) * (w // p), p * p)


class PatchGPT(GPT):
    def __init__(self, config):
        super().__init__(config, pad_vocab_size_to=1)
        head_dim = config.n_embd // config.n_head
        self.transformer.wte = nn.Linear(config.patch_size ** 2, config.n_embd, bias=False)
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(config.sequence_len, config.n_kv_head * head_dim)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        })

    @torch.no_grad()
    def init_weights(self):
        s = 3 ** 0.5 * self.config.n_embd ** (-0.5)
        nn.init.uniform_(self.transformer.wte.weight, -3 ** 0.5, 3 ** 0.5)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        for block in self.transformer.h:
            for linear in (block.attn.c_q, block.attn.c_k, block.attn.c_v, block.mlp.c_fc):
                nn.init.uniform_(linear.weight, -s, s)
            # Output projections at 1.5 / sqrt(fan_in), as dust.make_model does for the text model.
            for linear in (block.attn.c_proj, block.mlp.c_proj):
                scale = 1.5 * math.sqrt(3 / linear.weight.shape[1])
                nn.init.uniform_(linear.weight, -scale, scale)
            if block.attn.ve_gate is not None:
                nn.init.zeros_(block.attn.ve_gate.weight)
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)
        for ve in self.value_embeds.values():
            nn.init.uniform_(ve.weight, -s, s)
        head_dim = self.config.n_embd // self.config.n_head
        self.cos, self.sin = self._precompute_rotary(self.rotary_seq_len, head_dim)

    def forward(self, patches, targets=None, loss_reduction='mean'):
        B, T, _ = patches.size()
        cos_sin = (self.cos[:, :T], self.sin[:, :T])
        positions = torch.arange(T, device=patches.device).expand(B, T)
        x = norm(self.transformer.wte(patches))
        x0 = x
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](positions) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i])
        logits = self.lm_head(norm(x)).float()
        logits = 15 * torch.tanh(logits / 15)  # same soft cap DUST.head_error assumes
        if targets is not None:
            return F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.reshape(-1),
                ignore_index=-1, reduction=loss_reduction,
            )
        return logits


def make_model(config, device, seed):
    torch.manual_seed(seed)
    with torch.device('meta'):
        model = PatchGPT(config)
    model.to_empty(device=device)
    model.init_weights()
    return model.requires_grad_(False)
