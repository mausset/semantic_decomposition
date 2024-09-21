import argparse
import os
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
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
    shuffle=False,  # Disable shuffling for evaluation
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
        mirror=0,  # No mirroring during evaluation
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
        description="Evaluate Attentive Probe with DALI Data Loader using DDP"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to the config file",
    )
    parser.add_argument(
        "--interpreter_checkpoint",
        type=str,
        required=True,
        help="Path to the pretrained Interpreter model checkpoint",
    )
    parser.add_argument(
        "--probe_checkpoint",
        type=str,
        required=True,
        help="Path to the trained Attentive Probe checkpoint",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Total batch size for evaluation across all GPUs",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of DataLoader workers",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate on",
    )
    args = parser.parse_args()
    return args


def setup_distributed():
    """
    Initializes the distributed evaluation environment.
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
    Cleans up the distributed evaluation environment.
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
    pipeline_config.pop("batch_size", None)
    pipeline_config.pop("num_threads", None)

    # Create the DALI pipeline
    pipeline = video_pipe(
        **pipeline_config,
        filenames=filenames,
        labels=labels,
        shard_id=shard_id,
        num_shards=num_shards,
        seed=42,  # Different seed for reproducibility
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


def load_pretrained_interpreter(config, checkpoint_path, device):
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

    # Remove the 'module.' prefix if present (common with DataParallel or DDP)
    state_dict = {k[6:]: v for k, v in state_dict.items()}

    # Remove decoder keys if present
    decoder_keys = [k for k in state_dict.keys() if "decoder" in k]
    for k in decoder_keys:
        del state_dict[k]

    # Remove the positional embedding if necessary
    if "base.model.pos_embed" in state_dict:
        del state_dict["base.model.pos_embed"]

    # Load the state dict into the model
    missing = model.load_state_dict(state_dict, strict=False)

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    return model


def load_probe_model(checkpoint_path, device):
    """
    Loads the trained Attentive Probe model from the checkpoint.
    """
    # Initialize the Attentive Classifier (Probe)
    # These parameters should match those used during training
    embed_dim = 768  # Example value; adjust based on your configuration
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

    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Load state dict
    probe.load_state_dict(state_dict, strict=False)

    return probe


def main():
    args = parse_args()

    # Setup distributed evaluation
    rank, world_size, device, local_rank, distributed = setup_distributed()
    is_main = rank == 0

    if is_main:
        print(f"Evaluating on rank {rank} out of {world_size} processes.")

    # Load configuration
    config = OmegaConf.load(args.config)

    # Initialize and load the pretrained Interpreter model
    interpreter_model = load_pretrained_interpreter(
        config, args.interpreter_checkpoint, device
    )

    # Initialize and load the trained Attentive Probe
    probe = load_probe_model(args.probe_checkpoint, device)

    # Prepare the DataLoader for evaluation
    per_process_batch_size = args.batch_size

    eval_loader = prepare_dataloader(
        config,
        per_process_batch_size,
        args.num_workers,
        split=args.split,
        device_id=local_rank,
        shard_id=rank if distributed else 0,
        num_shards=world_size if distributed else 1,
    )

    # Evaluation mode
    probe.eval()
    interpreter_model.eval()

    # Disable gradient computation
    with torch.no_grad():
        # Initialize accuracy tracking variables
        total_correct = 0
        total_samples = 0

        if is_main:
            progress_bar = tqdm(
                enumerate(eval_loader),
                total=len(eval_loader),
                desc=f"Evaluating on {args.split} split",
            )
        else:
            progress_bar = enumerate(eval_loader)

        for batch_idx, batch in progress_bar:
            frames = batch["frames"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True).squeeze(-1).long()
            sequence_mask = batch.get("sequence_mask", None)
            if sequence_mask is not None:
                sequence_mask = sequence_mask.to(device, non_blocking=True)

            # Forward pass through the frozen Interpreter to get features
            features, _ = interpreter_model.forward_features(frames, sequence_mask)

            features = features[-1].squeeze(
                1
            )  # Adjust based on your model's output structure

            # Forward pass through the probe
            outputs = probe(features)

            # Compute predictions
            preds = outputs.argmax(dim=1)
            correct = (preds == labels).sum().item()
            total = labels.size(0)

            total_correct += correct
            total_samples += total

            if is_main:
                progress_bar.set_postfix(
                    {
                        "Correct": total_correct,
                        "Samples": total_samples,
                        "Accuracy": f"{(total_correct / total_samples) * 100:.2f}%",
                    }
                )

    # Synchronize and gather metrics
    tensor_correct = torch.tensor(total_correct, dtype=torch.float64, device=device)
    tensor_total = torch.tensor(total_samples, dtype=torch.float64, device=device)

    if distributed:
        dist.all_reduce(tensor_correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(tensor_total, op=dist.ReduceOp.SUM)

    if is_main:
        if tensor_total.item() == 0:
            avg_acc = 0.0
        else:
            avg_acc = tensor_correct.item() / tensor_total.item()
        print(f"\nEvaluation Results on {args.split} split:")
        print(f"Total Samples: {int(tensor_total.item())}")
        print(f"Total Correct: {int(tensor_correct.item())}")
        print(f"Accuracy: {avg_acc * 100:.2f}%")

    # Clean up distributed evaluation
    if distributed:
        cleanup_distributed()


if __name__ == "__main__":
    main()
