import numpy as np
import seaborn as sns
import torch
from einops import rearrange, repeat
from matplotlib.colors import to_hex
import matplotlib.pyplot as plt
from torch.nn import functional as F
from torchvision.transforms import Normalize

from PIL import Image, ImageDraw, ImageFont


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


def plot_alignment(
    img, img_attn_map, txt, txt_attn_map, patch_size=14, palette="muted", alpha=0.4
):
    """
    args:
        img: (3, H, W), image tensor
        img_attn_map: (B, HW, N), image attention map

        txt: (L), caption tensor
        txt_attn_map: (L, N), caption attention map
    returns:
        (3, H, W), image with attention map overlay
        (L), caption colored by attention map, same colors as image
    """
    _, n = img_attn_map.shape

    img = denormalize_imagenet(img)

    side = img.shape[-1] // patch_size
    img_attn_map = rearrange(img_attn_map, "(h w) n -> h w n", h=side)

    max_idx = img_attn_map.argmax(dim=-1, keepdim=True)

    attn_mask = (
        F.one_hot(max_idx.squeeze(-1), num_classes=n)
        .float()
        .permute(2, 0, 1)
        .unsqueeze(0)
    )
    attn_mask = F.interpolate(
        attn_mask, scale_factor=patch_size, mode="nearest"
    ).squeeze(0)

    palette = sns.color_palette(palette, n)
    palette_array = np.array(palette)
    colors = torch.tensor(palette_array[:, :3], dtype=torch.float32).to(img.device)

    colors = repeat(colors, "n c -> n c 1 1")

    attn_mask = repeat(attn_mask, "n h w -> n 1 h w")
    segment = (colors * attn_mask).sum(dim=0)
    segmented_img = img * alpha + segment * (1 - alpha)

    rendered_txt = render_text_image(txt, palette_array, txt_attn_map)

    rendered_txt = (
        torch.tensor(rendered_txt, device=segmented_img.device).permute(2, 0, 1).float()
        / 255.0
    )

    figure = torch.cat([segmented_img, rendered_txt], dim=2)

    return figure


def render_text_image(
    text,
    palette_array,
    text_attn_map,
    font_path="AmericanTypewriter.ttc",
    resolution=(224, 224),
    show_image=False,
):
    """
    Render the text with colored background as an image using Pillow, dynamically adjusting the font size and
    wrapping the text to more conservatively fill the image while keeping within specified resolution and avoiding cut-offs.
    args:
        text: list of words
        palette_array: the array of colors
        text_attn_map: the attention map for text
        font_path: path to the font file
        resolution: tuple (width, height) of the final image
        show_image: if True, display the image using Pillow's display method
    returns:
        A numpy array of the rendered text image.
    """
    width, height = resolution
    font_size = 10  # Start with a minimal font size to increase later
    font = ImageFont.truetype(font_path, font_size)

    # Increase font size conservatively to fit text
    while True:
        font = ImageFont.truetype(font_path, font_size)
        dummy_image = Image.new("RGB", (1, 1))
        draw = ImageDraw.Draw(dummy_image)
        line_width = 0
        line_height = (
            font_size * 1.2
        )  # Using a 1.2 scaling factor for line height to ensure padding
        lines = []
        current_line = []

        for word in text:
            word_width = draw.textlength(word, font=font) + 10  # Width with padding
            if line_width + word_width > width - 20:  # Consider padding on both sides
                lines.append(current_line)
                current_line = [word]
                line_width = word_width
            else:
                current_line.append(word)
                line_width += word_width
        if current_line:
            lines.append(current_line)

        total_height = line_height * len(lines)

        if (
            total_height > height - 20 or font_size > height / 10
        ):  # Stop if too tall or font size unreasonable
            font_size -= 2  # Step back by two sizes to be more conservative
            break
        font_size += 1

    # Draw the text onto the final image with the last fitting font size
    font = ImageFont.truetype(font_path, font_size)
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    y_offset = 10  # Start drawing slightly down from the top
    for line in lines:
        x_offset = 10  # Start slightly in from the left
        for word in line:
            color = tuple(
                int(c * 255)
                for c in palette_array[text_attn_map.argmax(dim=-1)[text.index(word)]]
            )
            draw.text((x_offset, y_offset), word, font=font, fill=color)
            x_offset += draw.textlength(word, font=font) + 10
        y_offset += int(line_height)  # Move down to the next line

    if show_image:
        image.show()

    return np.array(image)
