import numpy as np
import seaborn as sns
import torch
from einops import rearrange, repeat
from matplotlib.colors import to_hex
import matplotlib.pyplot as plt
from torch.nn import functional as F
from torchvision.transforms import Normalize

from functools import reduce
from operator import mul

from PIL import Image, ImageDraw, ImageFont


def denormalize_imagenet(img):
    mean = torch.tensor([0.485, 0.456, 0.406], device=img.device)
    std = torch.tensor([0.229, 0.224, 0.225], device=img.device)
    return Normalize((-mean / std).tolist(), (1.0 / std).tolist())(img)


def plot_attention_interpreter(
    imgs,
    attn_maps,
    res=224,
    patch_size=14,
    palette="tab20",
    alpha=0.4,
    include_original=False,
):
    b, _, n = attn_maps.shape

    img = denormalize_imagenet(imgs)

    palette = np.array(sns.color_palette(palette, n))
    colors = torch.tensor(palette[:, :3], dtype=torch.float16).to(img.device)
    colors = repeat(colors, "n c -> 1 n c 1 1")

    cat_imgs = []
    attn_map = rearrange(attn_maps, "b (h w) n -> b n h w", h=res // patch_size)
    attn_map = F.interpolate(attn_map, scale_factor=patch_size, mode="bilinear")
    max_idx = attn_map.argmax(dim=1)

    attn_mask = F.one_hot(max_idx, num_classes=n).float().permute(0, 3, 1, 2)

    attn_mask = repeat(attn_mask, "b n h w -> b n 1 h w")
    segment = (colors * attn_mask).sum(dim=1)
    # segmented_img = img * alpha
    # for color_index in range(n):
    #     color_mask = colors[:, color_index, :, :, :]
    #     attn_mask_single = attn_mask[:, color_index, :, :, :]
    #     segment = (color_mask * attn_mask_single).sum(dim=1, keepdim=True)
    #     segmented_img += segment * (1 - alpha)
    segmented_img = img * alpha + segment * (1 - alpha)

    return segmented_img


def plot_attention_interpreter_hierarchical(
    x,
    attn_maps,
    shrink_factors,
    t_max,
    res,
    patch_size,
):

    total_shrink = int(reduce(mul, shrink_factors, 1))
    multiples = t_max // total_shrink

    comb_attn = []

    for j in range(multiples):
        t = t_max
        attn_hierarchy = []
        total_shrink = int(reduce(mul, shrink_factors, 1))
        for i in range(len(attn_maps)):
            total_shrink //= shrink_factors[i]
            t = min(total_shrink, t_max)
            attn_hierarchy.append(attn_maps[i][t * j : t * (j + 1)])

        attn_hierarchy = attn_hierarchy[::-1]
        backward_sf = shrink_factors[: len(attn_hierarchy)][::-1]

        propagated_attn = []
        for i in range(len(attn_hierarchy)):

            a = attn_hierarchy[i]
            for a_, sf in zip(attn_hierarchy[i + 1 :], backward_sf[i:]):
                a = rearrange(a, "b (s n) m -> (b s) n m", s=sf)
                a = torch.bmm(a_, a)

            propagated_attn.append(a)

        comb_attn.append(propagated_attn)

    comb_attn = list(zip(*comb_attn))
    comb_attn = [torch.cat(attn, dim=0) for attn in comb_attn]
    attn_plots = []
    # t = min(t_max, int(reduce(mul, shrink_factors, 1)))
    for p_attn in reversed(comb_attn):
        attn_plots.append(
            plot_attention_interpreter(
                x[0][:t_max],
                p_attn,
                res=res[0],
                patch_size=patch_size,
            )
        )

    return attn_plots


def plot_attention(
    img,
    attn_map,
    res=224,
    patch_size=14,
    palette="muted",
    alpha=0.4,
    include_original=False,
):
    _, n = attn_map.shape

    img = denormalize_imagenet(img)

    palette = np.array(sns.color_palette(palette, n))
    colors = torch.tensor(palette[:, :3], dtype=torch.float32).to(img.device)
    colors = repeat(colors, "n c -> n c 1 1")

    cat_imgs = [img]
    attn_map = (
        rearrange(attn_map, "(h w) n -> h w n", h=res // patch_size)
        .permute(2, 0, 1)
        .unsqueeze(0)
    )

    attn_map = F.interpolate(
        attn_map, scale_factor=patch_size, mode="bilinear"
    ).squeeze(0)

    max_idx = attn_map.argmax(dim=0)

    attn_mask = F.one_hot(max_idx, num_classes=n).float().permute(2, 0, 1)

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
        collect.append(plot_attention(img, attn_t[0], res, patch_size, palette, alpha))

    w = int(np.ceil(np.sqrt(len(collect))))
    h = int(np.ceil(len(collect) / w))

    extra = int(w * h - len(collect))

    for _ in range(extra):
        collect.append(torch.zeros_like(collect[0]))

    rows = []
    for i in range(h):
        rows.append(torch.cat(collect[i * w : (i + 1) * w], dim=2))

    return torch.cat(rows, dim=1)
