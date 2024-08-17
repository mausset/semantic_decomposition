import torch


def block_causal_mask(t, n, device):
    mask = torch.triu(torch.ones(t * n, t * n, device=device)).bool()
    for i in range(t):
        mask[i * n : (i + 1) * n, i * n : (i + 1) * n] = True

    return mask


def apply_mask(x, mask):
    """
    Gather the elements of x that are not masked. 1 = not masked, 0 = masked.
    mask: (B, N)
    """

    indices = torch.arange(x.size(1), device=x.device).repeat(x.size(0), 1)
    indices = indices[mask].view(x.size(0), -1).unsqueeze(-1).repeat(1, 1, x.size(-1))

    gathered = torch.gather(x, 1, indices)

    return gathered


@torch.no_grad()
def attn_to_register_mask(attn, n_registers):
    """
    attn: (B, N, N)
    """
    attn = attn[:, :n_registers]

    mask = (
        torch.nn.functional.one_hot(attn.argmax(dim=1), num_classes=n_registers)
        .permute(0, 2, 1)
        .float()
    )
    mask[:, :n_registers, :n_registers] = 0
    for i in range(n_registers):
        mask[:, i, i] = 1

    masks = []
    for i in range(n_registers):
        masks.append(torch.bmm(mask[:, i, None].transpose(1, 2), mask[:, i, None]))

    return torch.stack(masks, dim=1).sum(dim=1).bool()
