import argparse
import math
import os
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset.ssv2 import VideoDataset
from models.attentive_pooler import AttentiveClassifier
from models.interpreter import Interpreter
from omegaconf import OmegaConf
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from tqdm import tqdm
from utils.schedulers import WarmupCosineSchedule

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Attentive Probe with AdamW Optimizer, Cosine Decay Scheduler, and Gradient Accumulation"
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
        "--batch_size", type=int, default=12, help="Batch size for training"
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
    state_dict = torch.load(checkpoint_path, map_location=device)["state_dict"]
    # Remove the 'module.' prefix if present (common with DataParallel)
    state_dict = {k[6:]: v for k, v in state_dict.items()}

    # Remove decoder keys if present
    decoder_keys = [k for k in state_dict.keys() if "decoder" in k]
    for k in decoder_keys:
        del state_dict[k]

    # Load the state dict into the model
    model.load_state_dict(state_dict, strict=False)

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    return model


def prepare_dataloader(config, batch_size, num_workers, split="train"):
    """
    Prepares the DataLoader for the specified dataset split.
    """
    # Initialize the dataset (assuming training split)
    pipeline_config = config.data.init_args.pipeline_config
    pipeline_config["batch_size"] = batch_size
    ssv2_lightning = VideoDataset(pipeline_config)
    ssv2_lightning.setup()
    dataloader = ssv2_lightning.train_dataloader()

    return dataloader


def main():
    args = parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load configuration
    config = OmegaConf.load(args.config)

    # Initialize and load the pretrained Interpreter model
    if args.checkpoint is None:
        raise ValueError("Please provide a checkpoint path using --checkpoint")
    interpreter_model = load_pretrained_model(config, args.checkpoint, device)

    # Determine the feature dimension from the config
    # Adjust based on your config structure
    embed_dim = config.model.init_args.base_config.get(
        "embed_dim", 768
    )  # Default to 768 if not specified

    # Initialize the Attentive Classifier (Probe)
    num_classes = 174
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

    # Prepare the training DataLoader
    train_loader = prepare_dataloader(
        config, args.batch_size, args.num_workers, split="train"
    )

    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    # Initialize the AdamW optimizer
    optimizer = AdamW(
        probe.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Calculate total training steps
    total_steps = args.epochs * len(train_loader)

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
        raise ValueError("accumulation_steps must be at least 1")

    for epoch in range(1, args.epochs + 1):
        probe.train()
        epoch_loss = 0.0
        accuracy_metric.reset()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")

        optimizer.zero_grad()  # Reset gradients at the start of the epoch

        for batch_idx, batch in enumerate(progress_bar):
            frames = batch["frames"].to(device)
            labels = batch["labels"].to(device).squeeze(-1).long()
            sequence_mask = batch["sequence_mask"].to(device)

            # Forward pass through the frozen Interpreter to get features
            with torch.no_grad():
                features, _ = interpreter_model.forward_features(frames, sequence_mask)

            features = features[-1].squeeze(1)
            # Forward pass through the probe
            outputs = probe(features)

            print(labels)
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

            # Update progress bar with loss and accuracy
            current_acc = accuracy_metric.compute().item()
            progress_bar.set_postfix(
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
        print(
            f"Epoch [{epoch}/{args.epochs}] - Average Loss: {avg_loss:.4f} - Average Accuracy: {avg_acc:.4f}"
        )

        # Save the probe checkpoint
        if epoch % args.save_every == 0 or epoch == args.epochs:
            checkpoint_dir = os.path.dirname(args.output)
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f"probe_epoch_{epoch}.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": probe.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": avg_loss,
                    "accuracy": avg_acc,
                },
                checkpoint_path,
            )
            print(f"Saved checkpoint: {checkpoint_path}")

    # Save the final trained probe
    torch.save(probe.state_dict(), args.output)
    print(f"Training completed. Final probe saved to {args.output}")


if __name__ == "__main__":
    main()
