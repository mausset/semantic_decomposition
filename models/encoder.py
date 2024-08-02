# Description: Encoder module for Vision Transformer
# Modified from x-transformers library

import torch
from einops import pack, rearrange, repeat, unpack
from models.positional_encoding import get_2d_sincos_pos_embed
from torch import nn
from utils.helpers import apply_mask
from x_transformers import Encoder


class ViTransformer(nn.Module):

    def __init__(
        self,
        *,
        image_size,
        patch_size,
        attn_layers,
        channels=3,
        num_register_tokens=0,
        sincos=False,
    ):
        super().__init__()
        assert (
            image_size % patch_size == 0
        ), "image dimensions must be divisible by the patch size"
        dim = attn_layers.dim
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size**2

        self.patch_size = patch_size

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        if sincos:
            pos_embedding = get_2d_sincos_pos_embed(dim, (image_size // patch_size))
            self.pos_embedding.data.copy_(torch.from_numpy(pos_embedding))
            self.pos_embedding.requires_grad_(False)

        has_register_tokens = num_register_tokens > 0
        self.has_register_tokens = has_register_tokens

        if has_register_tokens:
            self.register_tokens = nn.Parameter(torch.randn(num_register_tokens, dim))

        self.patch_to_embedding = nn.Sequential(
            nn.LayerNorm(patch_dim), nn.Linear(patch_dim, dim), nn.LayerNorm(dim)
        )

        self.attn_layers = attn_layers

    def forward(self, img, mask=None):
        b, p = img.shape[0], self.patch_size

        x = rearrange(img, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=p, p2=p)
        x = self.patch_to_embedding(x)

        x = x + self.pos_embedding

        if mask is not None:
            x = apply_mask(x, mask)

        if self.has_register_tokens:
            r = repeat(self.register_tokens, "n d -> b n d", b=b)
            x, ps = pack((x, r), "b * d")

        embed = self.attn_layers(x)

        if self.has_register_tokens:
            embed, _ = unpack(embed, ps, "b * d")  # type: ignore

        return embed


class Predictor(nn.Module):
    def __init__(self, dim, attn_layers, resolution, sincos=False):
        super().__init__()

        num_patches = resolution[0] * resolution[1]

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        if sincos:
            pos_embedding = get_2d_sincos_pos_embed(dim, resolution[0])
            self.pos_embedding.data.copy_(torch.from_numpy(pos_embedding))
            self.pos_embedding.requires_grad_(False)

        self.transformer = attn_layers

    def forward(self, x, target_mask):

        _, n_ctx, _ = x.shape
        _, m, *_ = target_mask.shape

        target = repeat(self.pos_embedding, "1 n d -> (b m 1) n d", b=x.shape[0], m=m)

        target_mask = rearrange(target_mask, "b m ... -> (b m) ...")
        target = apply_mask(target, target_mask)
        x = repeat(x, "b n d -> (b m) n d", m=m)
        target = torch.cat((x, target), dim=1)

        target = self.transformer(target)
        target = target[:, n_ctx:]

        return target
