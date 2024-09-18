import argparse
import warnings

import torch
import torch.nn.functional as F
from dataset.ytvis import YTVIS
from dataset.movi_e import MOVIe
from einops import rearrange
from models.interpreter import Interpreter
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.metrics import UnsupervisedMaskIoUMetric, ARIMetric
from utils.helpers import propagate_attention

from torchmetrics import MeanMetric

# from utils.plot import plot_attention_hierarchical

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="configs/config.yaml")
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--dataset", type=str, default="ytvis", choices=["ytvis", "movi_e"])

args = parser.parse_args()

config = OmegaConf.load(args.config)

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

fg_ari_img_per = ARIMetric(foreground=True, ignore_overlaps=True).to("cuda")
fg_ari_img = MeanMetric().to("cuda")
fg_ari_video = ARIMetric(foreground=True, ignore_overlaps=True).to("cuda")


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

dataset_config = config.data.init_args.dataset_config
if "repeat" in dataset_config:
    dataset_config["repeat"] = "repeat_sequence"


if args.dataset == "ytvis":
    dataset = YTVIS(**dataset_config, split="val")
elif args.dataset == "movi_e":
    dataset = MOVIe(**dataset_config, split="test")
else:
    raise ValueError("Invalid dataset")

dataset = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
)

resolution = config.data.init_args.dataset_config["resolution"]
patch_size = model.base.patch_size

with tqdm(dataset, desc="Evaluating") as pbar:
    for batch in pbar:
        frames = batch["frames"].to("cuda")
        n_frames = batch.get("n_frames", frames.shape[1])
        sequence_mask = batch["sequence_mask"].to("cuda")
        masks = batch["masks"].to("cuda").long().squeeze(2)

        with torch.no_grad():
            _, attn = model.forward_features(frames, sequence_mask)

        attn = propagate_attention(attn)[-1][:n_frames]

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
            fg_ari_img_per.update(p[:, i], t[:, i])
        fg_ari_img.update(fg_ari_img_per.compute())
        fg_ari_img_per.reset()
        fg_ari_video.update(p, t)

        pbar.set_postfix(
            {
                "MBO-IoU": f"{mbo_i.compute():.4f}",
                "MIoU": f"{miou.compute():.4f}",
                "FG-ARI Image": f"{fg_ari_img.compute():.4f}",
                "FG-ARI Vid": f"{fg_ari_video.compute():.4f}",
            }
        )
