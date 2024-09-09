import torch
from torch import nn
from x_transformers import Encoder
from models.positional_encoding import get_2d_sincos_pos_embed
from einops import rearrange, repeat, pack, unpack

from models.attention import Attention, DualAttention, AttentionDecode

from functools import reduce


class TransformerDecoderV2(nn.Module):

    def __init__(
        self,
        dim=384,
        depth=4,
        resolution=(32, 32),
        sincos=False,
    ):
        super().__init__()

        assert not (
            len(resolution) > 2 and sincos
        ), "Only 2D SinCos positional encoding supported"

        num_patches = reduce(lambda x, y: x * y, resolution, 1)
        self.resolution = resolution
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))

        if sincos:
            pos_embedding = get_2d_sincos_pos_embed(dim, resolution[0], resolution[1])
            self.pos_embedding.data.copy_(torch.from_numpy(pos_embedding))
            self.pos_embedding.requires_grad_(False)

        self.transformer = nn.ModuleList([])
        for _ in range(depth):
            self.transformer.append(AttentionDecode(dim, heads=8))

    def forward(self, x):
        """
        args:
            x: (B, N, D), extracted object representations
        returns:
            (B, HW, D), decoded features
        """

        # target = self.pe(x, resolution)
        target = repeat(self.pos_embedding, "1 n d -> (b 1) n d", b=x.shape[0])

        for block in self.transformer:
            target = block(target, x)

        return target


class TransformerDecoderV1(nn.Module):

    def __init__(
        self,
        dim=384,
        depth=4,
        resolution=(32, 32),
        sincos=False,
    ):
        super().__init__()

        assert not (
            len(resolution) > 2 and sincos
        ), "Only 2D SinCos positional encoding supported"

        num_patches = reduce(lambda x, y: x * y, resolution, 1)
        self.resolution = resolution
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))

        if sincos:
            pos_embedding = get_2d_sincos_pos_embed(dim, resolution[0], resolution[1])
            self.pos_embedding.data.copy_(torch.from_numpy(pos_embedding))
            self.pos_embedding.requires_grad_(False)

        self.transformer = Encoder(
            dim=dim,
            depth=depth,
            heads=8,
            cross_attend=True,
            ff_glu=True,
            attn_flash=True,
            # ff_swish=True,
        )

    def forward(self, x, mask=None):
        """
        args:
            x: (B, N, D), extracted object representations
        returns:
            (B, HW, D), decoded features
        """

        # target = self.pe(x, resolution)
        target = repeat(self.pos_embedding, "1 n d -> (b 1) n d", b=x.shape[0])

        target = self.transformer(target, context=x, context_mask=mask)

        return target
