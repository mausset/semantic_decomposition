import torch
from torch import nn
from torch.linalg import eigh
from torch.nn import functional as F
from einops import repeat


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


class CodeBook(nn.Module):

    def __init__(self, dim, n_codes=64, sign_disambiguation=True):
        super().__init__()

        self.sign_disambiguation = sign_disambiguation

        self.mu = nn.Parameter(torch.randn(n_codes, dim))

        log_sigma = torch.empty(n_codes, 1, dim)
        nn.init.xavier_uniform_(log_sigma)
        self.log_sigma = nn.Parameter(log_sigma.squeeze())

    def forward(self, x, n_slots=8):
        _, n, _ = x.shape

        x = F.normalize(x, p=2, dim=-1)

        cov = torch.einsum("bnd,bmd->bnm", x, x)

        eig_values, eig_vectors = eigh(cov)

        eig_values = torch.flip(eig_values, [-1])
        eig_vectors = torch.flip(eig_vectors, [1]).mT
        eig_values = eig_values[:, :n_slots]
        eig_vectors = eig_vectors[:, :n_slots]

        if self.sign_disambiguation:
            mask = (eig_vectors > 0).float().mean(dim=-1) < 0.5
            mask = repeat(mask, "b k -> b k n", n=n)

            flip_mask = torch.ones_like(eig_vectors)
            flip_mask[mask] = -1

            eig_vectors = eig_vectors * flip_mask

        projection = torch.matmul(eig_vectors, x)

        cosine_sim = F.cosine_similarity(
            projection.unsqueeze(2), self.mu.unsqueeze(0).unsqueeze(0), dim=-1
        )

        idx = torch.argmax(cosine_sim, dim=-1)

        mu_sample = self.mu[idx]
        sigma_sample = torch.exp(self.log_sigma[idx])

        sample = mu_sample + sigma_sample * torch.randn_like(mu_sample, device=x.device)

        return sample
