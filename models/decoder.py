import lightning as pl
from einops import rearrange, repeat
from utils.soft_pos_enc import FourierPositionalEncoding, make_grid
from torch import nn
from torch.nn import ConvTranspose2d, functional as F


class SpatialBroadcastDecoder(pl.LightningModule):

    def __init__(self, dim, depth, init_resolution) -> None:
        super().__init__()

        self.dim = dim
        self.depth = depth
        self.init_resolution = init_resolution

        convs = [
            ConvTranspose2d(
                in_channels=dim,
                out_channels=dim,
                kernel_size=5,
                stride=2,
                padding=2,
                output_padding=1,
            )
            for _ in range(depth - 2)
        ]
        convs.append(
            ConvTranspose2d(
                in_channels=dim,
                out_channels=dim,
                kernel_size=5,
                stride=1,
                padding=2,
            )
        )
        convs.append(
            ConvTranspose2d(
                in_channels=dim,
                out_channels=4,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        )
        self.convs = nn.ModuleList(convs)

        self.pos_enc = FourierPositionalEncoding(
            in_dim=2, out_dim=dim, num_pos_feats=64
        )

    def base_forward(self, x):

        _, n, _ = x.shape
        x = repeat(
            x,
            "b n d -> (b n) d h w",
            h=self.init_resolution[0],
            w=self.init_resolution[1],
        )
        grid = make_grid(self.init_resolution, device=x.device)
        grid = repeat(grid, "b h w c -> (b r) h w c", r=x.shape[0])
        x = x + rearrange(self.pos_enc(grid), "b h w c -> b c h w")

        for conv in self.convs[:-1]:
            x = F.relu(conv(x))
        x = self.convs[-1](x)

        x = rearrange(x, "(b n) d h w -> b n d h w", n=n)

        return x

    def forward(self, x):
        """
        args:
            x: (B, N, D), extracted object representations
        returns:
            (B, 3, H, W), predicted masks
        """

        x = self.base_forward(x)
        alphas = F.softmax(x[:, :, 3:4], dim=1)
        x = x[:, :, :-1]
        x = (x * alphas).sum(dim=1)

        return x

    def forward_components(self, x):
        """
        args:
            x: (B, N, D), extracted object representations
        returns:
            (B, N, 4, H, W), components
        """

        x = self.base_forward(x)
        alphas = F.softmax(x[:, :, 3:4], dim=1)
        x[:, :, 3:4] = alphas

        return x
