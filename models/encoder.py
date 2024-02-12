import lightning as pl
from einops import rearrange, repeat
from torch import nn
from torch.nn import functional as F
from utils.soft_pos_enc import FourierPositionalEncoding, make_grid


class ResNetEncoder(pl.LightningModule):

    def __init__(self, dim, depth) -> None:
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.stride_layers = (1, 2)
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=3 if not i else dim,
                    out_channels=dim,
                    kernel_size=5,
                    stride=2 if i in self.stride_layers else 1,
                    padding=2 if i in self.stride_layers else "same",
                )
                for i in range(depth)
            ]
        )

        self.pos_enc = FourierPositionalEncoding(
            in_dim=2, out_dim=dim, num_pos_feats=64
        )

    def forward(self, x):
        """
        args:
            x: (B, 3, H, W), images
        returns:
            (B, D, H, W), feature map
        """

        for conv in self.convs:
            x = F.relu(conv(x))

        grid = make_grid(x.shape[-2:], device=x.device)
        grid = repeat(grid, "b h w c -> (b r) h w c", r=x.shape[0])
        x = x + rearrange(self.pos_enc(grid), "b h w c -> b c h w")

        return x
