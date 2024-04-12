import numpy as np
import seaborn as sns
import torch
from torch.nn import functional as F
from torchvision.transforms import Normalize
from einops import rearrange, repeat


def denormalize_imagenet(img):
    mean = torch.tensor([0.485, 0.456, 0.406], device=img.device)
    std = torch.tensor([0.229, 0.224, 0.225], device=img.device)
    return Normalize((-mean / std).tolist(), (1.0 / std).tolist())(img)


def plot_attention(
    img,
    attn_maps,
    res=224,
    patch_size=14,
    palette="muted",
    alpha=0.4,
    include_original=False,
):
    _, n = attn_maps[0][0].shape

    img = denormalize_imagenet(img)

    palette = np.array(sns.color_palette(palette, n))
    colors = torch.tensor(palette[:, :3], dtype=torch.float32).to(img.device)
    colors = repeat(colors, "n c -> n c 1 1")

    cat_imgs = [img]
    for attn_map in attn_maps:
        attn_map = attn_map[0]
        attn_map = rearrange(attn_map, "(h w) n -> h w n", h=res // patch_size)
        max_idx = attn_map.argmax(dim=-1, keepdim=True)

        attn_mask = (
            F.one_hot(max_idx.squeeze(-1), num_classes=n)
            .float()
            .permute(2, 0, 1)
            .unsqueeze(0)
        )
        attn_mask = F.interpolate(
            attn_mask, scale_factor=patch_size, mode="nearest"
        ).squeeze(0)

        attn_mask = repeat(attn_mask, "n h w -> n 1 h w")
        segment = (colors * attn_mask).sum(dim=0)
        segmented_img = img * alpha + segment * (1 - alpha)
        cat_imgs.append(segmented_img)

    if include_original:
        cat_img = torch.cat(cat_imgs, dim=2)
    else:
        cat_img = torch.cat(cat_imgs[1:], dim=2)

    return cat_img


def plot_attention_hierarchical(
    img,
    attn_maps,
    res=224,
    patch_size=14,
    palette="muted",
    alpha=0.4,
):

    collect = []
    for attn_t in attn_maps:
        collect.append(plot_attention(img, attn_t, res, patch_size, palette, alpha))

    w = int(np.ceil(np.sqrt(len(collect))))
    h = int(np.ceil(len(collect) / w))

    extra = int(w * h - len(collect))

    for _ in range(extra):
        collect.append(torch.zeros_like(collect[0]))

    rows = []
    for i in range(h):
        rows.append(torch.cat(collect[i * w : (i + 1) * w], dim=2))

    return torch.cat(rows, dim=1)
