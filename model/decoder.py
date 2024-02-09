import lightning as pl
from einops import rearrange, repeat
from positional_encodings.torch_encodings import PositionalEncodingPermute2D
from torch import nn
from torch.nn import functional as F


class SpatialBroadcastDecoder(pl.LightningModule):

    def __init__(self, dim, depth, resolution) -> None:
        super().__init__()

        self.dim = dim
        self.depth = depth
        self.resolution = resolution

        self.convs = nn.ModuleList(
            [
                nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1)
                for _ in range(depth - 1)
            ]
        )
        self.last_conv = nn.Conv2d(
            in_channels=dim, out_channels=4, kernel_size=3, padding=1
        )

        self.pos_enc = PositionalEncodingPermute2D(dim)

    def forward(self, x):
        """
        args:
            x: (B, N, D), extracted object representations
        returns:
            (B, 3, H, W), predicted masks
        """

        _, n, _ = x.shape
        x = repeat(
            x, "b n d -> (b n) d h w", h=self.resolution[0], w=self.resolution[1]
        )
        x = x + self.pos_enc(x)

        for conv in self.convs:
            x = F.relu(conv(x))
        x = self.last_conv(x)

        x = rearrange(x, "(b n) d h w -> b n d h w", n=n)

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

        _, n, _ = x.shape
        x = repeat(
            x, "b n d -> (b n) d h w", h=self.resolution[0], w=self.resolution[1]
        )
        x = x + self.pos_enc(x)

        for conv in self.convs:
            x = F.relu(conv(x))
        x = self.last_conv(x)

        x = rearrange(x, "(b n) d h w -> b n d h w", n=n)

        alphas = F.softmax(x[:, :, 3:4], dim=1)
        x[:, :, 3:4] = alphas

        return x
