import torch
import torch.nn as nn
import numpy as np
from einops import repeat


class FourierPositionalEncoding(nn.Module):
    "Positional encoding learned in the Fourier basis."

    def __init__(self, in_dim: int = 2, out_dim: int = 256):
        super().__init__()
        scale = 1.0

        self.num_pos_feats = out_dim // 2
        self.out_dim = out_dim

        self.positional_encoding_gaussian = nn.Parameter(
            scale * torch.randn((in_dim, self.num_pos_feats))
        )

        self.ffn = nn.Sequential(
            nn.Linear(2 * self.num_pos_feats, self.num_pos_feats),
            nn.GELU(),
            nn.Linear(self.num_pos_feats, out_dim),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian
        coords = 2 * np.pi * coords
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): (d1,...,dn, 2).
        returns:
            (d1,...,dn, out_dim), positional encoding
        """
        x = self._pe_encoding(x)
        x = self.ffn(x)

        return x


class FourierScaffold(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.pos_enc = FourierPositionalEncoding(in_dim=in_dim, out_dim=out_dim)

    def make_grid(self, resolution, device=None):
        """Generate a 2D grid of size resolution."""
        h, w = resolution
        grid = torch.stack(
            torch.meshgrid(
                torch.linspace(0, 1, h, device=device),
                torch.linspace(0, 1, w, device=device),
                indexing="ij",
            ),
            dim=-1,
        )
        return grid

    def forward(self, x, resolution):
        b, _, _ = x.shape
        grid = self.make_grid(resolution, x.device)
        grid = repeat(grid, "h w d -> b (h w) d", b=b)
        scaffold = self.pos_enc(grid)
        return scaffold
