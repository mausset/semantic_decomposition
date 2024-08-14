import torch
from torch import nn
from x_transformers import Encoder
from models.positional_encoding import get_2d_sincos_pos_embed
from einops import rearrange, repeat, pack, unpack

from models.attention import TransformerLayer, DualTransformerLayer


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
            self.transformer.append(DualTransformerLayer(dim, 8))

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
        dim_context=384,
        depth=4,
        resolution=(32, 32),
        patch_size=None,
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

        self.transformer = Encoder(
            dim=dim,
            dim_context=dim_context,
            depth=depth,
            ff_glu=True,
            ff_swish=True,
            attn_flash=True,
            cross_attend=True,
        )
        self.patch_size = patch_size

        if patch_size is not None:
            self.ff = nn.Linear(dim, patch_size * patch_size * 3)

    def forward(self, x, mask=None, context_mask=None):
        """
        args:
            x: (B, N, D), extracted object representations
        returns:
            (B, HW, D), decoded features
        """

        # target = self.pe(x, resolution)
        target = repeat(self.pos_embedding, "1 n d -> (b 1) n d", b=x.shape[0])

        result = self.transformer(
            target,
            mask=mask,
            context=x,
            context_mask=context_mask,
        )

        if self.patch_size is not None:
            result = self.ff(result)
            result = rearrange(
                result,
                "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
                h=self.resolution[0],
                w=self.resolution[1],
                p1=self.patch_size,
                p2=self.patch_size,
                c=3,
            )

        return result
