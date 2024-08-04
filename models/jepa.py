from copy import deepcopy

import lightning as pl
import torch
import wandb
from einops import rearrange, repeat
from geomloss import SamplesLoss
from models.encoder import Predictor, ViTransformer
from torch import nn
from torch.optim import AdamW
from torchmetrics.aggregation import MeanMetric
from utils.helpers import apply_mask
from utils.plot import denormalize_imagenet, visualize_top_components
from utils.schedulers import WarmupCosineSchedule, LinearSchedule
from x_transformers import Encoder


class JEPA(nn.Module):

    def __init__(self, config: dict):
        super().__init__()

        self.dim = config["dim"]
        self.patch_size = config["patch_size"]
        self.resolution = config["resolution"]
        self.feature_map_resolution = (
            self.resolution[0] // self.patch_size,
            self.resolution[1] // self.patch_size,
        )

        self.encoder = ViTransformer(
            image_size=self.resolution[0],
            patch_size=self.patch_size,
            attn_layers=Encoder(
                dim=self.dim,
                depth=config["enc_depth"],
                heads=config["enc_heads"],
                ff_glu=True,
                attn_flash=True,
            ),
            num_register_tokens=config.get("n_registers", 0),
            sincos=True,
        )
        self.teacher = deepcopy(self.encoder).eval().requires_grad_(False)

        self.predictor = Predictor(
            dim=self.dim,
            attn_layers=Encoder(
                dim=self.dim,
                depth=config["pred_depth"],
                heads=config["pred_heads"],
                ff_glu=True,
                attn_flash=True,
            ),
            resolution=self.feature_map_resolution,
            sincos=True,
        )


class JEPATrainer(pl.LightningModule):

    def __init__(
        self,
        config: dict,
        optimizer_config: dict = {},
    ):
        super().__init__()

        self.config = config
        self.n_target_blocks = config["n_target_blocks"]
        self.scale = config["scale"]
        self.patch_size = config["patch_size"]
        self.resolution = config["resolution"]
        self.feature_map_resolution = (
            self.resolution[0] // self.patch_size,
            self.resolution[1] // self.patch_size,
        )

        self.optimizer_config = optimizer_config

        self.jepa: JEPA = torch.compile(JEPA(config))  # type: ignore

        self.metric_loggers = nn.ModuleDict(
            {
                "loss": MeanMetric(),
                "target_var": MeanMetric(),
                "target_norm": MeanMetric(),
                "target_patch_var": MeanMetric(),
            }
        )

        if config["loss"] == "mse":
            self.loss_fn = nn.MSELoss(reduce="none")
        elif config["loss"] == "sinkhorn":
            self.loss_fn = SamplesLoss(loss="sinkhorn", p=2, blur=0.05, scaling=0.5)

        self.image = None

    @property
    def device(self):
        return next(self.parameters()).device

    def generate_mask(self, x):
        min_size = max(round(self.feature_map_resolution[0] * self.scale[0]), 1)
        max_size = round(self.feature_map_resolution[0] * self.scale[1])

        h_size = torch.randint(min_size, max_size, (1,))
        w_size = torch.randint(min_size, max_size, (1,))

        masks = []
        for _ in range(self.n_target_blocks):
            mask = torch.zeros(
                x.size(0),
                self.feature_map_resolution[0],
                self.feature_map_resolution[1],
                device=self.device,
            )

            h_start = torch.randint(
                0, self.feature_map_resolution[0] - h_size + 1, (1,)
            )
            w_start = torch.randint(
                0, self.feature_map_resolution[1] - w_size + 1, (1,)
            )
            h_end = h_start + h_size
            w_end = w_start + w_size

            mask[:, h_start:h_end, w_start:w_end] = 1

            masks.append(mask)

        target_mask = torch.stack(masks, dim=1)
        target_mask = rearrange(target_mask, "... h w -> ... (h w)").bool()

        context_mask = ~target_mask.sum(dim=1).bool()

        return context_mask, target_mask

    def forward(self, x, stage="train"):

        context_mask, target_mask = self.generate_mask(x)
        with torch.no_grad():
            target = self.jepa.teacher(x)

            self.metric_loggers["target_patch_var"](target.var(dim=0).mean())

            self.metric_loggers["target_var"](
                rearrange(target, "b n d -> (b n) d").var(dim=0).mean()
            )
            self.metric_loggers["target_norm"](
                rearrange(target, "b n d -> (b n) d").norm(dim=1).mean()
            )

            if self.image is None and self.trainer.is_global_zero:
                self.image = visualize_top_components(
                    target[0].detach(),
                    self.patch_size,
                    denormalize_imagenet(x[0].detach()) * 255,
                    n_components=63,
                )

            target = repeat(target, "b n d -> (b m) n d", m=self.n_target_blocks)
            masks = rearrange(target_mask, "b m n -> (b m) n")
            target = apply_mask(target, masks)

        x = self.jepa.encoder(x, mask=context_mask)
        predicted = self.jepa.predictor(x, target_mask)

        loss = self.loss_fn(predicted, target.detach()).mean()

        return loss

    def log_metrics(self, batch_idx, stage="train"):
        if batch_idx % self.trainer.accumulate_grad_batches != 0:
            return

        for k, v in self.metric_loggers.items():
            value = v.compute()
            if torch.isnan(value):
                continue
            self.log(f"{stage}/{k}", value, prog_bar=True, sync_dist=True)
            v.reset()

    def training_step(self, x, batch_idx: int):

        loss = self.forward(x)
        self.metric_loggers["loss"](loss)

        self.log_metrics(batch_idx)

        return loss

    def on_train_epoch_end(self):
        if self.image is not None and self.trainer.is_global_zero:
            self.logger.experiment.log({"target_components": wandb.Image(self.image)})  # type: ignore
            self.image = None

    def validation_step(self, x):

        loss = self.forward(x, stage="val")

        self.log("val/loss", loss, prog_bar=True, sync_dist=True)

        return loss

    def update_target(self):
        alpha = self.alpha(self.global_step)
        for p, target_p in zip(
            self.jepa.encoder.parameters(), self.jepa.teacher.parameters()  # type: ignore
        ):
            target_p.data.mul_(alpha).add_(p.data.detach(), alpha=1 - alpha)

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.update_target()

    def configure_optimizers(self):  # type: ignore
        self.alpha = LinearSchedule(
            self.config["alpha"][0],
            self.config["alpha"][1],
            self.trainer.estimated_stepping_batches,
        )

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
