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
    attn_map_sa,
    attn_map_decoder,
    res=224,
    patch_size=14,
    palette="muted",
    alpha=0.4,
):
    _, n = attn_map_sa.shape
    attn_map_sa = rearrange(attn_map_sa, "(h w) n -> h w n", h=res // patch_size)
    max_idx_sa = attn_map_sa.argmax(dim=-1, keepdim=True)

    attn_mask_sa = (
        F.one_hot(max_idx_sa.squeeze(-1), num_classes=n)
        .float()
        .permute(2, 0, 1)
        .unsqueeze(0)
    )

    attn_mask_sa = F.interpolate(
        attn_mask_sa, scale_factor=patch_size, mode="nearest"
    ).squeeze(0)

    attn_map_decoder = rearrange(
        attn_map_decoder, "(h w) n -> h w n", h=res // patch_size
    )
    max_idx_decoder = attn_map_decoder.argmax(dim=-1, keepdim=True)

    attn_mask_decoder = (
        F.one_hot(max_idx_decoder.squeeze(-1), num_classes=n)
        .float()
        .permute(2, 0, 1)
        .unsqueeze(0)
    )

    attn_mask_decoder = F.interpolate(
        attn_mask_decoder, scale_factor=patch_size, mode="nearest"
    ).squeeze(0)

    palette = np.array(sns.color_palette(palette, n))
    colors = torch.tensor(palette[:, :3], dtype=torch.float32).to(img.device)
    colors = repeat(colors, "n c -> n c 1 1")

    attn_mask_sa = repeat(attn_mask_sa, "n h w -> n 1 h w")
    segmented_img = (colors * attn_mask_sa).sum(dim=0)

    attn_mask_decoder = repeat(attn_mask_decoder, "n h w -> n 1 h w")
    segmented_img_decoder = (colors * attn_mask_decoder).sum(dim=0)

    img = denormalize_imagenet(img)
    overlayed_img_sa = img * alpha + segmented_img * (1 - alpha)
    overlayed_img_decoder = img * alpha + segmented_img_decoder * (1 - alpha)
    combined_img = torch.cat([img, overlayed_img_sa, overlayed_img_decoder], dim=2)

    return combined_img
