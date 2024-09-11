import argparse
import warnings

import torch
import torch.nn.functional as F
from dataset.ytvis import YTVIS
from einops import rearrange
from models.interpreter import Interpreter
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.metrics import UnsupervisedMaskIoUMetric, ARIMetric
from utils.helpers import propagate_attention

# from utils.plot import plot_attention_hierarchical

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="configs/config.yaml")
parser.add_argument("--checkpoint", type=str, default=None)

config = OmegaConf.load(parser.parse_args().config)

mbo_i = UnsupervisedMaskIoUMetric(
    matching="best_overlap",
    ignore_background=True,
    ignore_overlaps=True,
).to("cuda")

miou = UnsupervisedMaskIoUMetric(
    matching="hungarian",
    ignore_background=True,
    ignore_overlaps=True,
).to("cuda")

fg_ari = ARIMetric(foreground=True, ignore_overlaps=True).to("cuda")


model = (
    Interpreter(
        config.model.init_args.base_config,
        config.model.init_args.block_configs,
    )
    .to("cuda")
    .eval()
)

state_dict = torch.load(parser.parse_args().checkpoint)["state_dict"]
state_dict = {k[6:]: v for k, v in state_dict.items()}
decoder_keys = [k for k in state_dict.keys() if "decoder" in k]
for k in decoder_keys:
    del state_dict[k]

model.load_state_dict(state_dict, strict=False)

assert len(model.blocks) == 2, "Model must have 2 blocks for video segmentation"

ytvis = DataLoader(
    YTVIS(**config.data.init_args.dataset_config, split="val"),
    batch_size=1,
    shuffle=False,
)

resolution = config.data.init_args.dataset_config["resolution"]
patch_size = model.base.patch_size

for batch in tqdm(ytvis):
    frames = batch["frames"].to("cuda")
    masks = batch["masks"].to("cuda").long().squeeze(2)

    with torch.no_grad():
        _, attn = model.forward_features(frames)

    # attn_plot = plot_attention_hierarchical(frames, attn, (336, 504), patch_size)
    # print(attn_plot[1].shape)
    # exit()

    attn = propagate_attention(attn)[-1]

    attn = rearrange(
        attn,
        "b (h w) n -> b n h w",
        h=resolution[0] // patch_size,
    )
    pred_mask = (
        F.interpolate(attn, scale_factor=patch_size, mode="bilinear")
        .argmax(1)
        .unsqueeze(0)
    )

    p = F.one_hot(pred_mask).to(torch.float32).permute(0, 1, 4, 2, 3)
    t = F.one_hot(masks).to(torch.float32).permute(0, 1, 4, 2, 3)

    mbo_i.update(p, t)
    miou.update(p, t)
    for i in range(p.shape[1]):
        fg_ari.update(p[:, i], t[:, i])

    print("MBO-IoU:", mbo_i.compute())
    print("MIoU:", miou.compute())
    print("FG-ARI:", fg_ari.compute())
