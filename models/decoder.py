import lightning as pl
import torch
from einops import rearrange, repeat
from torch import nn
from torch.nn import functional as F
from utils.soft_pos_enc import FourierPositionalEncoding, make_grid
from x_transformers import ContinuousTransformerWrapper, Encoder


class MLPDecoder(pl.LightningModule):

    def __init__(self, dim, depth, resolution, expansion_factor=2) -> None:
        super().__init__()

        self.dim = dim
        self.depth = depth
        self.resolution = resolution

        self.pos_enc = FourierPositionalEncoding(in_dim=2, out_dim=dim)

        mlp = []
        for i in range(depth - 1):
            mlp.append(
                nn.Linear(dim * (expansion_factor if i else 1), dim * expansion_factor)
            )
            mlp.append(nn.ReLU())
        mlp.append(nn.Linear(dim * expansion_factor, dim))
        self.mlp = nn.Sequential(*mlp)

        self.alpha_projection = nn.Linear(dim, 1)

    def forward(self, x):
        """
        args:
            x: (B, N, D), extracted object representations
        returns:
            (B, H, W), predicted masks
        """

        grid = make_grid(self.resolution, device=x.device)
        grid = repeat(grid, "b h w d -> (b r) n h w d", r=x.shape[0], n=x.shape[1])

        scaffold = self.pos_enc(grid)
        x = repeat(x, "b n d -> b n h w d", h=self.resolution[0], w=self.resolution[1])
        x = x + scaffold

        result = self.mlp(x)
        alpha = self.alpha_projection(result)
        alpha = F.softmax(alpha, dim=1)
        result = (result * alpha).sum(dim=1)

        return result, alpha


class TransformerDecoder(pl.LightningModule):

    def __init__(self, dim, depth, resolution) -> None:
        super().__init__()

        self.dim = dim
        self.depth = depth
        self.resolution = resolution

        self.pos_enc = FourierPositionalEncoding(in_dim=2, out_dim=dim)

        self.transformer = ContinuousTransformerWrapper(
            max_seq_len=None,
            attn_layers=Encoder(
                dim=dim,
                depth=depth,
                cross_attend=True,
                ff_glu=True,
                ff_swish=True,
            ),
            use_abs_pos_emb=False,
        )

        self.alpha_projection = nn.Linear(dim, 1)

    def forward(self, x, split=2):
        """
        args:
            x: (B, N, D), extracted object representations
        returns:
            (B, H, W), predicted masks
        """
        b, n, _ = x.shape

        x = repeat(x, "b n d -> (r b) n d", r=split)

        mask = torch.rand((b, n), device=x.device) > 0.5
        mask = torch.cat([mask, ~mask], dim=0)

        grid = make_grid(self.resolution, device=x.device)
        grid = repeat(grid, "b h w d -> (b r) (h w) d", r=b * split)

        scaffold = self.pos_enc(grid)

        result = self.transformer(scaffold, context=x, context_mask=mask)
        result = rearrange(
            result,
            "(s b) (h w) d -> b s h w d",
            s=split,
            h=self.resolution[0],
            w=self.resolution[1],
        )
        alpha = self.alpha_projection(result)
        alpha = F.softmax(alpha, dim=1)
        result = (result * alpha).sum(dim=1)

        return result, alpha
