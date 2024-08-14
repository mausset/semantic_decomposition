import torch.nn.functional as F
from einops import pack, unpack
from torch import nn


class TransformerLayer(nn.Module):

    HEAD_DIM = 64

    def __init__(self, dim, heads, expansion=4):
        super(TransformerLayer, self).__init__()
        self.dim = dim
        self.heads = heads
        self.attn_dim = heads * self.HEAD_DIM
        self.expansion = expansion

        self.to_q = nn.Linear(self.dim, self.attn_dim)
        self.to_k = nn.Linear(self.dim, self.attn_dim)
        self.to_v = nn.Linear(self.dim, self.attn_dim)

        self.attn_out = nn.Linear(self.attn_dim, self.dim)

        self.norm1 = nn.LayerNorm(self.dim)
        self.norm2 = nn.LayerNorm(self.dim)

        self.mlp = nn.Sequential(
            nn.Linear(self.dim, self.dim * expansion),
            nn.GELU(),
            nn.Linear(self.dim * expansion, self.dim),
        )

    def forward(self, x):
        b, n, d = x.shape

        # Linear projections
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q = q.view(b, n, self.heads, self.HEAD_DIM).transpose(1, 2)
        k = k.view(b, n, self.heads, self.HEAD_DIM).transpose(1, 2)
        v = v.view(b, n, self.heads, self.HEAD_DIM).transpose(1, 2)

        with nn.attention.sdpa_kernel(nn.attention.SDPBackend.FLASH_ATTENTION):
            attn_output = F.scaled_dot_product_attention(q, k, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(b, n, self.attn_dim)
        attn_output = self.attn_out(attn_output)

        x = self.norm1(x + attn_output)

        mlp_output = self.mlp(x)
        x = self.norm2(x + mlp_output)

        return x


class DualTransformerLayer(nn.Module):

    HEAD_DIM = 64

    def __init__(self, dim, heads, expansion=4):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.attn_dim = heads * self.HEAD_DIM
        self.expansion = expansion

        self.to_q = nn.Linear(self.dim, self.attn_dim)
        self.to_k = nn.Linear(self.dim, self.attn_dim)
        self.to_v = nn.Linear(self.dim, self.attn_dim)

        self.attn_out = nn.Linear(self.attn_dim, self.dim)

        self.norm1_1 = nn.LayerNorm(self.dim)
        self.norm2_1 = nn.LayerNorm(self.dim)
        self.mlp1 = nn.Sequential(
            nn.Linear(self.dim, self.dim * expansion),
            nn.GELU(),
            nn.Linear(self.dim * expansion, self.dim),
        )

        self.norm1_2 = nn.LayerNorm(self.dim)
        self.norm2_2 = nn.LayerNorm(self.dim)
        self.mlp2 = nn.Sequential(
            nn.Linear(self.dim, self.dim * expansion),
            nn.GELU(),
            nn.Linear(self.dim * expansion, self.dim),
        )

    def forward(self, x, y):

        x, ps = pack((x, y), "b * d")

        b, n, d = x.shape

        # Linear projections
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q = q.view(b, n, self.heads, self.HEAD_DIM).transpose(1, 2)
        k = k.view(b, n, self.heads, self.HEAD_DIM).transpose(1, 2)
        v = v.view(b, n, self.heads, self.HEAD_DIM).transpose(1, 2)

        with nn.attention.sdpa_kernel(nn.attention.SDPBackend.FLASH_ATTENTION):
            attn_output = F.scaled_dot_product_attention(q, k, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(b, n, self.attn_dim)
        attn_output = self.attn_out(attn_output)

        x = x + attn_output
        x, y = unpack(x, ps, "b * d")

        x = self.norm1_1(x)
        mlp_output = self.mlp1(x)
        x = self.norm1_2(x + mlp_output)

        y = self.norm1_2(y)
        mlp_output = self.mlp2(y)
        y = self.norm2_2(y + mlp_output)

        return x, y
