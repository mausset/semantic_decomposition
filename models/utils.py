import torch
from torch import nn
from torch.nn import functional as F


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


class ConceptBank(nn.Module):

    def __init__(self, dim, n_concepts=64):
        super().__init__()

        self.mu = nn.Parameter(torch.randn(n_concepts, dim))

        log_sigma = torch.empty(n_concepts, 1, dim)
        nn.init.xavier_uniform_(log_sigma)
        self.log_sigma = nn.Parameter(log_sigma.squeeze())

    def forward(self, x, n_slots=8):

        cosine_sim = F.cosine_similarity(
            x.unsqueeze(2), self.mu_concepts.unsqueeze(0).unsqueeze(0), dim=-1
        )

        cosine_sim_soft = F.softmax(cosine_sim, dim=-1)

        r = F.softmax(cosine_sim_soft.sum(dim=1), dim=-1)

        _, idx = torch.topk(r, n_slots, dim=-1)

        mu_sample = self.mu_concepts[idx]
        sigma_sample = torch.exp(self.log_sigma_concepts[idx])

        sample = mu_sample + sigma_sample * torch.randn_like(mu_sample)

        return sample
