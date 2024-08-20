import torch.nn.functional as F
from einops import pack, rearrange, unpack
from torch import nn


class Attention(nn.Module):

    HEAD_DIM = 64

    def __init__(self, dim, heads, expansion=4):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.attn_dim = heads * self.HEAD_DIM
        self.expansion = expansion

        self.to_q = nn.Linear(self.dim, self.attn_dim, bias=False)
        self.to_k = nn.Linear(self.dim, self.attn_dim, bias=False)
        self.to_v = nn.Linear(self.dim, self.attn_dim, bias=False)

        self.attn_out = nn.Linear(self.attn_dim, self.dim, bias=False)

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


class DualAttention(nn.Module):

    HEAD_DIM = 64

    def __init__(self, dim, heads, expansion=4):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.attn_dim = heads * self.HEAD_DIM
        self.expansion = expansion

        self.to_q = nn.Linear(self.dim, self.attn_dim, bias=False)
        self.to_k = nn.Linear(self.dim, self.attn_dim, bias=False)
        self.to_v = nn.Linear(self.dim, self.attn_dim, bias=False)

        self.attn_out = nn.Linear(self.attn_dim, self.dim, bias=False)

        self.norm1_1 = nn.LayerNorm(self.dim)
        self.norm1_2 = nn.LayerNorm(self.dim)
        self.mlp1 = nn.Sequential(
            nn.Linear(self.dim, self.dim * expansion),
            nn.GELU(),
            nn.Linear(self.dim * expansion, self.dim),
        )

        self.norm2_1 = nn.LayerNorm(self.dim)
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

        y = self.norm2_1(y)
        mlp_output = self.mlp2(y)
        y = self.norm2_2(y + mlp_output)

        return x, y


class SwiGLU(nn.Module):

    def __init__(self, dim, expansion=4):
        super().__init__()

        self.dim = dim
        self.project_in = nn.Linear(dim, dim * expansion * 2)
        self.project_out = nn.Linear(dim * expansion, dim)
        self.act = nn.SiLU()

    def forward(self, x):
        x, y = self.project_in(x).chunk(2, dim=-1)

        return self.project_out(x * self.act(y))


class AttentionDecode(nn.Module):

    HEAD_DIM = 64

    def __init__(self, dim, heads, expansion=4):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.attn_dim = heads * self.HEAD_DIM
        self.expansion = expansion

        self.norm_cross_target = nn.LayerNorm(self.dim)
        self.to_q_cross = nn.Linear(self.dim, self.attn_dim, bias=False)
        self.to_k_cross = nn.Linear(self.dim, self.attn_dim, bias=False)
        self.to_v_cross = nn.Linear(self.dim, self.attn_dim, bias=False)
        self.attn_out_cross = nn.Linear(self.attn_dim, self.dim, bias=False)

        self.norm_self = nn.LayerNorm(self.dim)
        self.to_q_self = nn.Linear(self.dim, self.attn_dim, bias=False)
        self.to_k_self = nn.Linear(self.dim, self.attn_dim, bias=False)
        self.to_v_self = nn.Linear(self.dim, self.attn_dim, bias=False)
        self.attn_out_self = nn.Linear(self.attn_dim, self.dim, bias=False)

        self.norm_mlp = nn.LayerNorm(self.dim)
        self.mlp = SwiGLU(dim, expansion)
        self.norm_final = nn.LayerNorm(self.dim)

    def forward(self, target, context):

        target = self.norm_cross_target(target)

        # Linear projections
        q = self.to_q_cross(target)
        k = self.to_k_cross(context)
        v = self.to_v_cross(context)

        q = rearrange(q, "b n (h d) -> b h n d", h=self.heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.heads)

        with nn.attention.sdpa_kernel(nn.attention.SDPBackend.FLASH_ATTENTION):
            attn_out = F.scaled_dot_product_attention(q, k, v)

        attn_out = rearrange(attn_out, "b h n d -> b n (h d)")
        attn_out = self.attn_out_cross(attn_out)

        target = target + attn_out

        target = self.norm_self(target)

        q = self.to_q_self(target)
        k = self.to_k_self(target)
        v = self.to_v_self(target)

        q = rearrange(q, "b n (h d) -> b h n d", h=self.heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.heads)

        with nn.attention.sdpa_kernel(nn.attention.SDPBackend.FLASH_ATTENTION):
            attn_out = F.scaled_dot_product_attention(q, k, v)

        attn_out = rearrange(attn_out, "b h n d -> b n (h d)")
        attn_out = self.attn_out_self(attn_out)

        target = target + attn_out
        mlp_in = self.norm_mlp(target)
        mlp_out = self.mlp(mlp_in)
        target = target + mlp_out

        target = self.norm_final(target)

        return target
