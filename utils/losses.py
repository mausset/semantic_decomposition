import torch


class EntropyLoss(torch.nn.Module):
    def __init__(self):
        super(EntropyLoss, self).__init__()

    def forward(self, x):
        n = torch.tensor(x.size(1))
        return -torch.sum(x * torch.log(x + 1e-8), dim=-1) / torch.log(n)
