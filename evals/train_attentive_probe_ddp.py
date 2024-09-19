import argparse
import os
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim.adamw import AdamW
from torchmetrics import Accuracy
from omegaconf import OmegaConf
from tqdm import tqdm

# Import NVIDIA DALI components
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.base_iterator import LastBatchPolicy
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from nvidia.dali import pipeline_def  # type: ignore

# Import your model components
from models.attentive_pooler import AttentiveClassifier
from models.interpreter import Interpreter
from utils.schedulers import WarmupCosineSchedule

# Suppress unnecessary warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

initial_prefetch_size = 8


@pipeline_def
def video_pipe(
    filenames=[],
    labels=[],
    resolution=(224, 224),
    sequence_length=16,
    stride=1,
    step=1,
    shuffle=True,
    shard_id=0,
    num_shards=1,
):
    videos, labels, timestamps = fn.readers.video_resize(  # type: ignore
        device="gpu",
        filenames=filenames,
        labels=labels,
        resize_shorter=resolution[0],
        sequence_length=sequence_length,
        enable_timestamps=True,
        pad_sequences=True,
        shard_id=shard_id,
        stride=stride,
        step=step,
        num_shards=num_shards,
        random_shuffle=shuffle,
        name="Reader",
        initial_fill=initial_prefetch_size,
        prefetch_queue_depth=1,
        file_list_include_preceding_frame=True,
    )

    sequence_mask = timestamps >= 0

    coin = fn.random.coin_flip(probability=0.5)
    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
    videos = fn.crop_mirror_normalize(
        videos,
        crop_h=resolution[0],
        crop_w=resolution[1],
        dtype=types.FLOAT,  # type: ignore
        mean=mean,
        std=std,
        output_layout="FCHW",
        mirror=coin,
    )

    return videos, labels, sequence_mask


class LightningWrapper(DALIGenericIterator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __next__(self):  # type: ignore
        out = super().__next__()
        out = out[0]

        frames = out["frames"]
        sequence_mask = out["sequence_mask"]

        n_frames = sequence_mask.sum(dim=-1).clamp(min=1)

        S = frames.size(1)
        batch_size = frames.size(0)
        device = frames.device
        range_tensor = torch.arange(S, device=device).unsqueeze(0).repeat(batch_size, 1)

        j = range_tensor % n_frames.unsqueeze(1)

        j_expanded = (
            j.unsqueeze(2)
            .unsqueeze(3)
            .unsqueeze(4)
            .expand(-1, -1, frames.size(2), frames.size(3), frames.size(4))
        )

        cyclic_frames = torch.gather(frames, 1, j_expanded)

        out["frames"] = cyclic_frames
        out["n_frames"] = n_frames
        out["sequence_mask"][:] = 1

        return out


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Attentive Probe with AdamW Optimizer, Cosine Decay Scheduler, Gradient Accumulation, and DALI Data Loader using DDP"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to the config file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to the pretrained model checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="trained_probe.pth",
        help="Path to save the trained probe",
    )
    parser.add_argument(
        "--epochs", type=int, default=20, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Total batch size for training across all GPUs",
    )
    parser.add_argument(
        "--accumulation_steps",
        type=int,
        default=10,
        help="Number of gradient accumulation steps",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Initial learning rate for AdamW optimizer",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-2, help="Weight decay for optimizer"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of DataLoader workers"
    )
    parser.add_argument(
        "--save_every", type=int, default=5, help="Save checkpoint every N epochs"
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=0,
        help="Number of warmup steps for the scheduler",
    )
    args = parser.parse_args()
    return args


def setup_distributed():
    """
    Initializes the distributed training environment.
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # Distributed mode
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)

        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
        )
        distributed = True
    else:
        # Single GPU mode
        print("Not using distributed mode")
        rank = 0
        world_size = 1
        local_rank = 0
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        distributed = False

    return rank, world_size, device, local_rank, distributed


def cleanup_distributed():
    """
    Cleans up the distributed training environment.
    """
    dist.destroy_process_group()


def prepare_dataloader(
    config, batch_size, num_workers, split, device_id, shard_id, num_shards
):
    """
    Prepares the DALI DataLoader for the specified dataset split.
    """
    # Read the filenames and labels for the specified split
    root = Path(config.data.init_args.pipeline_config["root"])
    if split == "train":
        data_dir = root / "train"
    elif split == "val":
        data_dir = root / "validation"
    elif split == "test":
        data_dir = root / "test"
    else:
        raise ValueError(f"Unknown split: {split}")

    filenames = sorted(list(data_dir.glob("**/*.mp4")))
    labels = [int(f.parent.name) for f in filenames]

    # Get other pipeline parameters from config
    pipeline_config = config.data.init_args.pipeline_config.copy()
    del pipeline_config["root"]
    del pipeline_config["batch_size"]
    del pipeline_config["num_threads"]

    # Create the DALI pipeline
    pipeline = video_pipe(
        **pipeline_config,
        filenames=filenames,
        labels=labels,
        shard_id=shard_id,
        num_shards=num_shards,
        seed=12 + shard_id,
        batch_size=batch_size,
        num_threads=num_workers,
        device_id=device_id,
    )

    # Create the DALI DataLoader
    data_loader = LightningWrapper(
        pipeline,
        reader_name="Reader",
        output_map=["frames", "labels", "sequence_mask"],
        last_batch_policy=LastBatchPolicy.DROP,
        auto_reset=True,
    )

    return data_loader


def main():
    args = parse_args()

    # Setup distributed training
    rank, world_size, device, local_rank, distributed = setup_distributed()
    is_main = rank == 0

    if is_main:
        print(f"Training on rank {rank} out of {world_size} processes.")

    # Load configuration
    config = OmegaConf.load(args.config)

    # Initialize and load the pretrained Interpreter model
    if args.checkpoint is None:
        raise ValueError("Please provide a checkpoint path using --checkpoint")
    interpreter_model = load_pretrained_model(config, args.checkpoint, device)

    # Determine the feature dimension from the config
    embed_dim = config.model.init_args.base_config.get(
        "embed_dim", 768
    )  # Default to 768 if not specified

    # Initialize the Attentive Classifier (Probe)
    num_classes = 174  # Adjust based on your dataset
    probe = AttentiveClassifier(
        embed_dim=embed_dim,
        num_heads=12,  # Adjust as needed
        mlp_ratio=4.0,
        depth=1,  # You can experiment with depth >1
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        qkv_bias=True,
        num_classes=num_classes,
        complete_block=True,
    )
    probe = probe.to(device)

    # Wrap the probe with DistributedDataParallel
    if distributed:
        probe = nn.parallel.DistributedDataParallel(
            probe, device_ids=[local_rank], output_device=local_rank
        )

    # Prepare the training DataLoader
    if distributed:
        per_process_batch_size = args.batch_size // world_size
    else:  # Single GPU mode
        per_process_batch_size = args.batch_size
    if per_process_batch_size < 1:
        if is_main:
            print(
                f"Warning: per-process batch size {per_process_batch_size} < 1. Setting to 1."
            )
        per_process_batch_size = 1

    train_loader = prepare_dataloader(
        config,
        per_process_batch_size,
        args.num_workers,
        split="train",
        device_id=local_rank,
        shard_id=rank if distributed else 0,
        num_shards=world_size if distributed else 1,
    )

    # Define the loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # Initialize the AdamW optimizer
    optimizer = AdamW(
        probe.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Calculate total training steps
    total_steps = args.epochs * (len(train_loader) // args.accumulation_steps)

    # Initialize the WarmupCosineSchedule scheduler
    scheduler = WarmupCosineSchedule(
        optimizer=optimizer,
        warmup_steps=args.warmup_steps,
        start_lr=0.0,
        ref_lr=args.lr,
        T_max=total_steps,
        final_lr=0.0,
    )

    # Initialize TorchMetrics' Accuracy
    accuracy_metric = Accuracy(task="multiclass", num_classes=num_classes).to(device)

    # Gradient Accumulation Steps
    accumulation_steps = args.accumulation_steps
    if accumulation_steps < 1:
        if is_main:
            print("Error: accumulation_steps must be at least 1")
        cleanup_distributed()
        raise ValueError("accumulation_steps must be at least 1")

    # Training loop
    for epoch in range(1, args.epochs + 1):
        probe.train()
        epoch_loss = 0.0
        accuracy_metric.reset()

        if is_main:
            progress_bar = tqdm(
                enumerate(train_loader),
                total=len(train_loader),
                desc=f"Epoch {epoch}/{args.epochs}",
            )
        else:
            progress_bar = enumerate(train_loader)

        optimizer.zero_grad()  # Reset gradients at the start of the epoch

        for batch_idx, batch in progress_bar:
            frames = batch["frames"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True).squeeze(-1).long()
            sequence_mask = batch.get("sequence_mask", None)
            if sequence_mask is not None:
                sequence_mask = sequence_mask.to(device, non_blocking=True)

            # Forward pass through the frozen Interpreter to get features
            with torch.no_grad():
                features, _ = interpreter_model.forward_features(frames, sequence_mask)

            features = features[-1].squeeze(
                1
            )  # Adjust based on your model's output structure

            # Forward pass through the probe
            outputs = probe(features)

            # Compute loss and scale by accumulation_steps
            loss = criterion(outputs, labels) / accumulation_steps
            loss.backward()

            # Compute accuracy
            preds = outputs.argmax(dim=1)
            accuracy_metric.update(preds, labels)

            epoch_loss += loss.item() * accumulation_steps  # Accumulate the actual loss

            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if is_main and (batch_idx + 1) % accumulation_steps == 0:
                current_acc = accuracy_metric.compute().item()
                progress_bar.set_postfix(  # type: ignore
                    {"Loss": loss.item() * accumulation_steps, "Acc": current_acc}
                )

        # Handle remaining gradients if the number of batches is not divisible by accumulation_steps
        if len(train_loader) % accumulation_steps != 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # Compute average loss and accuracy for the epoch
        avg_loss = epoch_loss / len(train_loader)
        avg_acc = accuracy_metric.compute().item()

        if is_main:
            print(
                f"Epoch [{epoch}/{args.epochs}] - Average Loss: {avg_loss:.4f} - Average Accuracy: {avg_acc:.4f}"
            )

        # Save the probe checkpoint
        if is_main and (epoch % args.save_every == 0 or epoch == args.epochs):
            checkpoint_dir = os.path.dirname(args.output)
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f"probe_epoch_{epoch}.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": probe.module.state_dict(),  # unwrap DDP
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": avg_loss,
                    "accuracy": avg_acc,
                },
                checkpoint_path,
            )
            print(f"Saved checkpoint: {checkpoint_path}")

    if is_main:
        # Save the final trained probe
        torch.save(probe.module.state_dict(), args.output)  # unwrap DDP
        print(f"Training completed. Final probe saved to {args.output}")

    # Clean up distributed training
    if distributed:
        cleanup_distributed()


def load_pretrained_model(config, checkpoint_path, device):
    """
    Loads the pretrained Interpreter model, removes decoder keys, and freezes its parameters.
    """
    # Initialize the Interpreter model
    model = Interpreter(
        config.model.init_args.base_config,
        config.model.init_args.block_configs,
    )
    model = model.to(device)
    model.eval()

    # Load the state dictionary
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

    state_dict = {k[6:]: v for k, v in state_dict.items()}

    # Remove decoder keys if present
    decoder_keys = [k for k in state_dict.keys() if "decoder" in k]
    for k in decoder_keys:
        del state_dict[k]

    del state_dict["base.model.pos_embed"]
    # Load the state dict into the model
    model.load_state_dict(state_dict, strict=False)

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    return model


if __name__ == "__main__":
    main()
