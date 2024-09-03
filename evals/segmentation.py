import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from einops import rearrange
from models.interpreter import Interpreter
from omegaconf import OmegaConf
from utils.metrics import UnsupervisedMaskIoUMetric
from dataset.ytvis import YTVIS
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="configs/config.yaml")
parser.add_argument("--checkpoint", type=str, default=None)

config = OmegaConf.load(parser.parse_args().config)


metrics = {
    "mbo_i": UnsupervisedMaskIoUMetric(
        matching="best_overlap",
        ignore_background=True,
        ignore_overlaps=True,
    ).to("cuda"),
    "miou": UnsupervisedMaskIoUMetric(
        matching="hungarian",
        ignore_background=True,
        ignore_overlaps=True,
    ).to("cuda"),
}


model = (
    Interpreter(
        config.model.init_args.base_config,
        config.model.init_args.block_configs,
    )
    .to("cuda")
    .eval()
)

model.load_state_dict(torch.load(parser.parse_args().checkpoint), strict=False)

assert len(model.blocks) == 2, "Model must have 2 blocks for video segmentation"

ytvis = DataLoader(
    YTVIS(**config.data.init_args.dataset_config, split="val"),
    batch_size=1,
)

resolution = config.data.init_args.dataset_config["resolution"]
patch_size = model.base.patch_size

for batch in tqdm(ytvis):
    frames = batch["frames"].to("cuda")
    masks = batch["masks"].to("cuda")
    sequence_mask = batch["sequence_mask"].to("cuda")

    with torch.no_grad():
        _, _, attn_maps, _ = model(frames, seq_mask=sequence_mask)

    _, t, _, _ = attn_maps[0].shape
    a0 = rearrange(attn_maps[0], "b t ... -> (b t) ...")
    a1 = rearrange(attn_maps[1], "b t (s n) ... -> (b t s) n ...", s=t)

    attn = torch.bmm(a0, a1)

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
    pred_mask = pred_mask[:, sequence_mask.squeeze(0)]
    masks = masks[:, sequence_mask.squeeze(0)]

    p = F.one_hot(pred_mask).to(torch.float32).permute(0, 1, 4, 2, 3)
    t = F.one_hot(masks).to(torch.float32).permute(0, 1, 4, 2, 3)

    metrics["mbo_i"].update(p, t)
    metrics["miou"].update(p, t)


print("MBO-IoU:", metrics["mbo_i"].compute())
print("MIoU:", metrics["miou"].compute())
