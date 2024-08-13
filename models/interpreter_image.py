import lightning as pl
import timm
import torch
import torch.nn.functional as F
import wandb
from einops import rearrange
from geomloss import SamplesLoss
from models.decoder import TransformerDecoder, TransformerDecoderV2
from models.slot_attention import SA
from torch import nn
from torch.optim import AdamW
from torchmetrics.aggregation import MeanMetric
from utils.metrics import ARIMetric, UnsupervisedMaskIoUMetric
from utils.plot import plot_attention
from utils.schedulers import WarmupCosineSchedule


class TimmWrapper(nn.Module):

    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        num_classes: int = 0,
        in_chans: int = 3,
        resolution: int = 224,
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=in_chans,
            img_size=resolution,
        )

        self.discard = 1 + 4 if "reg" in model_name else 1

        self.patch_size = self.model.patch_embed.patch_size[0]

    def forward(self, x):
        return self.model.forward_features(x)[:, self.discard :]


class InterpreterBlock(nn.Module):

    def __init__(
        self,
        config: dict,
        n_decode: int,
    ):
        super().__init__()
        self.dim = config["dim"]
        self.slot_dim = config["slot_dim"]
        self.n_slots = config["n_slots"]

        self.decode_res = next(
            (i, n_decode // i)
            for i in range(int(n_decode**0.5), 0, -1)
            if n_decode % i == 0
        )

        print(f"Decode res: {self.decode_res}")

        self.slot_attention = SA(
            input_dim=self.dim,
            slot_dim=self.slot_dim,
            n_slots=self.n_slots,
            n_iters=8,
        )
        self.decoder = TransformerDecoderV2(
            dim=self.dim,
            dim_context=self.slot_dim,
            depth=config["dec_depth"],
            resolution=self.decode_res,
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x):

        slots, attn_map = self.slot_attention(x, n_slots=self.n_slots)

        return slots, attn_map


class Interpreter(nn.Module):

    def __init__(
        self,
        base_config: dict,
        block_configs: list[dict],
    ):
        super().__init__()

        self.base = TimmWrapper(**base_config).eval().requires_grad_(False)

        base_resolution = base_config["resolution"][0] // self.base.patch_size
        n_decode = base_resolution**2
        self.blocks = nn.ModuleList([])
        for i in range(len(block_configs)):
            self.blocks.append(InterpreterBlock(block_configs[i], n_decode=n_decode))
            n_decode = block_configs[i]["n_slots"]

        for block in self.blocks[:-1]:  # type: ignore
            block.eval().requires_grad_(False)

    def forward(self, x):

        x = self.base(x)

        attn_maps = []
        features = [x]
        for block in self.blocks:  # type: ignore
            x, attn_map = block(x)
            features.append(x)
            attn_maps.append(attn_map)

        return x, attn_maps

    # Assumes that only the last block is to be trained
    def forward_train(self, x):
        attn_maps = []
        with torch.no_grad():
            x = self.base(x)

            for block in self.blocks[:-1]:  # type: ignore
                x, attn_map = block(x)
                attn_maps.append(attn_map)

        target = x.clone()

        x, attn_map = self.blocks[-1](x)
        attn_maps.append(attn_map)
        decoded = self.blocks[-1].decoder(x)

        return decoded, target, attn_maps


class InterpreterTrainer(pl.LightningModule):

    def __init__(
        self,
        base_config: dict,
        block_configs: list[dict],
        optimizer_config: dict = {},
        checkpoint_path: str | None = None,
    ):
        super().__init__()

        self.interpreter = torch.compile(Interpreter(base_config, block_configs))

        if checkpoint_path is not None:
            self.load_state_dict(
                torch.load(checkpoint_path)["state_dict"], strict=False
            )

        self.patch_size = self.interpreter.base.patch_size  # type: ignore
        self.resolution = base_config["resolution"]
        self.n_blocks = len(self.interpreter.blocks)  # type: ignore
        self.active_block = self.n_blocks - 1
        self.optimizer_config = optimizer_config

        self.loss_fn = SamplesLoss(loss="sinkhorn", p=2, blur=0.05, scaling=0.5)

        self.metrics = nn.ModuleDict(
            {
                "norm": MeanMetric(),
                "var": MeanMetric(),
                "loss": MeanMetric(),
            }
        )

        self.val_metrics = nn.ModuleDict(
            {
                "mbo_i": UnsupervisedMaskIoUMetric(
                    matching="best_overlap",
                    ignore_background=True,
                    ignore_overlaps=True,
                ).to(self.device),
                "mbo_c": UnsupervisedMaskIoUMetric(
                    matching="best_overlap",
                    ignore_background=True,
                    ignore_overlaps=True,
                ).to(self.device),
                "miou": UnsupervisedMaskIoUMetric(
                    matching="hungarian",
                    ignore_background=True,
                    ignore_overlaps=True,
                ).to(self.device),
            }
        )

        self.image = None

    def forward(self, x, stage="train"):

        decoded, target, attn_maps = self.interpreter.forward_train(x)  # type: ignore

        loss = self.loss_fn(decoded, target).mean()

        flat = rearrange(decoded, "b n d -> (b n) d")
        self.metrics["norm"](torch.norm(flat.detach(), dim=1).mean())  # type: ignore
        self.metrics["var"](torch.var(flat.detach(), dim=0).mean())  # type: ignore
        self.metrics["loss"](loss.detach())  # type: ignore

        return loss, attn_maps

    def log_metrics(self, metrics):

        for k, v in metrics.items():
            value = v.compute()
            if torch.isnan(value):
                continue
            self.log(
                f"train/{k}_{self.active_block}",
                value,
                prog_bar=True,
                sync_dist=True,
            )
            v.reset()

    def training_step(self, x, batch_idx: int):

        x = x

        loss, _ = self.forward(x)

        if batch_idx % self.trainer.accumulate_grad_batches == 0:
            self.log_metrics(self.metrics)

        return loss

    def on_validation_epoch_end(self):
        self.log_metrics(self.val_metrics)
        self.image = None

    @torch.no_grad()
    def validation_step(self, x):
        x, *masks = x

        loss, attn_maps = self.forward(x)

        attn_map = attn_maps[0]
        for i in range(1, len(attn_maps)):
            attn_map = torch.bmm(attn_map, attn_maps[i])

        self.update_segmentation_metrics(masks, attn_map)

        if (
            isinstance(self.logger, pl.pytorch.loggers.WandbLogger)  # type: ignore
            and self.trainer.is_global_zero
            and self.image is None
        ):

            attn_plot = plot_attention(
                img=x[0],
                attn_map=attn_map[0],
                res=self.resolution[0],
                patch_size=self.patch_size,
            )
            log_dict = {
                f"attention_{self.active_block}": wandb.Image(
                    (attn_plot.cpu().permute(1, 2, 0).numpy() * 255).astype("uint8")
                )
            }
            self.logger.experiment.log(log_dict)  # type: ignore

            self.image = log_dict

        return loss

    @torch.no_grad()
    def update_segmentation_metrics(self, masks, attn_map):

        i_mask, c_mask, ignore_mask = masks

        attn_map = rearrange(
            attn_map,
            "b (h w) n -> b n h w",
            h=self.resolution[0] // self.patch_size,
        )
        pred_mask = (
            F.interpolate(attn_map, size=self.resolution, mode="bilinear")
            .unsqueeze(2)
            .argmax(1)
            .squeeze(1)
        )

        max_batch = 4

        for i in range(0, len(i_mask), max_batch):
            true_mask_i = (
                F.one_hot(i_mask[i : i + max_batch])
                .to(torch.float32)
                .permute(0, 3, 1, 2)
            )
            true_mask_c = (
                F.one_hot(c_mask[i : i + max_batch])
                .to(torch.float32)
                .permute(0, 3, 1, 2)
            )
            pred_mask_l = (
                F.one_hot(pred_mask[i : i + max_batch])
                .to(torch.float32)
                .permute(0, 3, 1, 2)
            )

            ignore_mask_l = ignore_mask[i : i + max_batch]

            self.val_metrics["mbo_i"].update(
                pred_mask_l,
                true_mask_i,
                ignore_mask_l,
            )
            self.val_metrics["mbo_c"].update(
                pred_mask_l,
                true_mask_c,
                ignore_mask_l,
            )
            self.val_metrics["miou"].update(
                pred_mask_l,
                true_mask_i,
                ignore_mask_l,
            )

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)

    def configure_optimizers(self):  # type: ignore
        optimizer = AdamW(
            self.parameters(), weight_decay=self.optimizer_config["weight_decay"]
        )

        steps_per_epoch = self.trainer.estimated_stepping_batches / self.trainer.max_epochs  # type: ignore

        scheduler = WarmupCosineSchedule(
            optimizer,
            warmup_steps=self.optimizer_config["warmup_epochs"] * steps_per_epoch,
            start_lr=self.optimizer_config["start_lr"],
            ref_lr=self.optimizer_config["ref_lr"],
            final_lr=self.optimizer_config["final_lr"],
            T_max=self.trainer.max_epochs * steps_per_epoch,  # type: ignore
        )

        config = {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

        return config
