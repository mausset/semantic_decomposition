import lightning as pl
from einops import repeat
from torch import nn
from torch.nn import functional as F
from utils.pos_enc import FourierPositionalEncoding, make_grid
from x_transformers import Encoder


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

        self.transformer = Encoder(
            dim=dim,
            depth=depth,
            cross_attend=True,
            ff_glu=True,
            ff_swish=True,
        )

    def forward(self, x, context_mask=None):
        """
        args:
            x: (B, N, D), extracted object representations
        returns:
            (B, HW, D), decoded features
            (B, HW, N), cross-attention map
        """
        b, _, _ = x.shape

        grid = make_grid(self.resolution, device=x.device)
        grid = repeat(grid, "h w d -> b (h w) d", b=b)

        scaffold = self.pos_enc(grid)

        result, hiddens = self.transformer(
            scaffold, context=x, context_mask=context_mask, return_hiddens=True
        )

        # Last cross-attention map, mean across heads
        attn_map = hiddens.attn_intermediates[-1].post_softmax_attn.mean(dim=1)

        return result, attn_map
