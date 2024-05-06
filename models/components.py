import torch
from torch import nn
from torch.nn import functional as F
from einops import repeat


class GaussianPrior(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.init_mu = nn.Parameter(torch.randn(dim))
        init_log_sigma = torch.empty((1, 1, dim))
        nn.init.xavier_uniform_(init_log_sigma)
        self.init_log_sigma = nn.Parameter(init_log_sigma.squeeze())

    def forward(self, x, n, sample=None):
        b, _, _ = x.shape

        mu = repeat(self.init_mu, "d -> b n d", b=b, n=n)
        log_sigma = repeat(self.init_log_sigma, "d -> b n d", b=b, n=n)
        if sample is None:
            sample = mu + log_sigma.exp() * torch.randn_like(mu)
        else:
            sample = mu + sample * log_sigma.exp()

        return sample


class SwiGLUFFN(nn.Module):

    def __init__(self, dim, expansion_factor=4):
        super().__init__()

        self.hidden_dim = dim * expansion_factor
        self.hidden_dim = int(2 * self.hidden_dim / 3)

        self.W1 = nn.Linear(dim, self.hidden_dim, bias=False)
        self.W2 = nn.Linear(self.hidden_dim, dim, bias=False)
        self.V = nn.Linear(dim, self.hidden_dim, bias=False)

    def forward(self, x):
        return self.W2(F.silu(self.W1(x)) * self.V(x))
