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

from sklearn.decomposition import PCA


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

    # cat_imgs = []
    attn_map = rearrange(attn_maps, "b (h w) n -> b n h w", h=res // patch_size)
    attn_map = F.interpolate(attn_map, scale_factor=patch_size, mode="bilinear")
    max_idx = attn_map.argmax(dim=1)

    segment_list = []
    for i in range(b):
        attn_mask = (
            F.one_hot(max_idx[i, None], num_classes=n).float().permute(0, 3, 1, 2)
        )
        attn_mask = repeat(attn_mask, "b n h w -> b n 1 h w")
        segment = (colors * attn_mask).sum(dim=1)
        segment_list.append(segment)

    segment = torch.cat(segment_list, dim=0)
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
    # This function is a mess, needs refactoring

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
    for p_attn in reversed(comb_attn):
        attn_plots.append(
            plot_attention_interpreter(
                x[0][: p_attn.shape[0]],
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
    palette="tab20",
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


def visualize_features(feature_map, patch_size, original_image):
    """
    Visualize the first principal component magnitude of patches from a self-supervised model.

    Args:
    feature_map (torch.Tensor): The feature map from the model of shape [N, D] where
                                N is the number of patches and D is the dimensionality of features.
    patch_size (int): The size of each patch (assuming square patches).
    original_image (torch.Tensor): The original image tensor.
    """
    # Calculate the number of patches along width and height assuming square patches
    img_height, img_width = original_image.shape[1], original_image.shape[2]
    num_patches_side = img_height // patch_size

    # Perform PCA on the feature map to reduce to the top component
    pca = PCA(n_components=25)
    feature_map_np = feature_map.detach().cpu().numpy()  # Convert to numpy for PCA
    principal_components = pca.fit_transform(feature_map_np)[:, 15, None]

    # Map the principal component scores back to an image grid
    component_image = torch.tensor(principal_components).float()
    component_image = rearrange(
        component_image, "(h w) 1 -> 1 h w", h=num_patches_side, w=num_patches_side
    )

    # Scale to [0, 1] for visualization
    component_image -= component_image.min()
    component_image /= component_image.max()

    # Resize to the original image dimensions
    component_image = torch.nn.functional.interpolate(
        component_image.unsqueeze(0), size=(img_height, img_width), mode="nearest"
    ).squeeze(0)

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original_image.permute(1, 2, 0).cpu().numpy().astype("uint8"))
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Patch-wise First Principal Component")
    plt.imshow(component_image.squeeze(), cmap="hot")
    plt.axis("off")
    plt.show()


def visualize_top_components(feature_map, patch_size, original_image, n_components=5):
    """
    Visualize the top n principal components of patches from a self-supervised model.

    Args:
    feature_map (torch.Tensor): The feature map from the model of shape [N, D] where
                                N is the number of patches and D is the dimensionality of features.
    patch_size (int): The size of each patch (assuming square patches).
    original_image (torch.Tensor): The original image tensor.
    n_components (int): Number of top principal components to visualize.
    """
    # Calculate the number of patches along width and height assuming square patches
    img_height, img_width = original_image.shape[1], original_image.shape[2]
    num_patches_side = img_height // patch_size

    # Perform PCA on the feature map to reduce to the top n components
    pca = PCA(n_components=n_components)
    feature_map_np = feature_map.detach().cpu().numpy()  # Convert to numpy for PCA
    principal_components = pca.fit_transform(
        feature_map_np
    )  # This is the correct transformed data

    # Plotting
    fig, axes = plt.subplots(1, n_components + 1, figsize=(n_components * 3, 3))

    # Display original image
    axes[0].imshow(original_image.permute(1, 2, 0).cpu().numpy().astype("uint8"))
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    for i in range(n_components):
        component = torch.tensor(principal_components[:, i]).float()
        component_image = rearrange(
            component, "(h w) -> 1 h w", h=num_patches_side, w=num_patches_side
        )

        # Scale to [0, 1] for visualization
        component_image -= component_image.min()
        component_image /= component_image.max()

        # Resize to the original image dimensions
        component_image = torch.nn.functional.interpolate(
            component_image.unsqueeze(0), size=(img_height, img_width), mode="nearest"
        ).squeeze(0)

        axes[i + 1].imshow(component_image.squeeze(), cmap="hot")
        axes[i + 1].set_title(f"Component {i + 1}")
        axes[i + 1].axis("off")

    plt.tight_layout()
    plt.show()
