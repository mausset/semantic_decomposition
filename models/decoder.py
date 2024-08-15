import torch
from torch import nn
from x_transformers import Encoder
from models.positional_encoding import get_2d_sincos_pos_embed
from einops import rearrange, repeat, pack, unpack

from models.attention import Attention, DualAttention, AttentionDecode


class TransformerDecoderV2(nn.Module):

    def __init__(
        self,
        dim=384,
        dim_context=384,
        depth=4,
        resolution=(32, 32),
        sincos=False,
    ):
        super().__init__()

        # self.pe = PE2D(dim)
        num_patches = resolution[0] * resolution[1]
        self.resolution = resolution
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))

        if sincos:
            pos_embedding = get_2d_sincos_pos_embed(dim, (resolution[0]))
            self.pos_embedding.data.copy_(torch.from_numpy(pos_embedding))
            self.pos_embedding.requires_grad_(False)

        self.transformer = nn.ModuleList([])
        for _ in range(depth):
            self.transformer.append(DualAttention(dim, 8))

    def forward(self, x):
        """
        args:
            x: (B, N, D), extracted object representations
        returns:
            (B, HW, D), decoded features
        """

        target = repeat(self.pos_embedding, "1 n d -> (b 1) n d", b=x.shape[0])

        # x, ps = pack((x, target), "b * d")

        for block in self.transformer:
            target, x = block(target, x)

        # x, target = unpack(x, ps, "b * d")

        return target


class TransformerDecoder(nn.Module):

    def __init__(
        self,
        dim=384,
        depth=4,
        resolution=(32, 32),
        sincos=False,
    ):
        super().__init__()

        # self.pe = PE2D(dim)
        num_patches = resolution[0] * resolution[1]
        self.resolution = resolution
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))

        if sincos:
            pos_embedding = get_2d_sincos_pos_embed(dim, (resolution[0]))
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
