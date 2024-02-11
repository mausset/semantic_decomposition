import torch
import torch.nn as nn
import numpy as np


def make_grid(resolution, device=None):
    """Generate a 2D grid of size resolution."""
    h, w = resolution
    grid = torch.stack(
        torch.meshgrid(
            torch.linspace(0, 1, h, device=device),
            torch.linspace(0, 1, w, device=device),
        ),
        dim=-1,
    )
    grid = torch.unsqueeze(grid, 0)
    return grid


class FourierPositionalEncoding(nn.Module):
    "Positional encoding learned in the Fourier basis."

    def __init__(self, in_dim: int = 2, out_dim: int = 256, num_pos_feats: int = 64):
        super().__init__()
        scale = 1.0

        self.num_pos_feats = num_pos_feats
        self.out_dim = out_dim

        self.positional_encoding_gaussian = nn.Parameter(
            scale * torch.randn((in_dim, num_pos_feats))
        )

        self.ffn = nn.Sequential(
            nn.Linear(2 * num_pos_feats, num_pos_feats),
            nn.GELU(),
            nn.Linear(num_pos_feats, out_dim),
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
