import os

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision.io import read_file, decode_png
from torchvision.transforms.v2 import (
    Compose,
    Normalize,
    Resize,
    ToDtype,
    ToImage,
    RandomHorizontalFlip,
)


class MOVIe(Dataset):
    def __init__(self, file_root, sequence_length, resolution, split="train"):
        """
        Each video in MOVIe has 24 frames. Validation and test splits have masks.
        Args:
            file_root (string): Directory with all the video folders.
            sequence_length (int): Number of frames in each sequence. Must be 1 or 24.
        """
        assert split in ["train", "validation", "test"]
        assert (
            sequence_length == 1 or sequence_length == 24
        ), "sequence_length must be 1 (image) or 24 (full sequence)"

        self.file_root = file_root
        self.sequence_length = sequence_length
        self.split = split
        self.transform = Compose(
            [
                Resize(resolution[0]),
                ToImage(),
                ToDtype(torch.float, scale=True),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                RandomHorizontalFlip(0.5 if split == "train" else 0),
            ]
        )
        self.mask_transform = Compose(
            [
                Resize(
                    resolution[0],
                    interpolation=torchvision.transforms.InterpolationMode.NEAREST,
                ),
                ToImage(),
                ToDtype(torch.long),
            ],
        )

        # Collect all video folders
        self.video_sequences = []
        path = os.path.join(file_root, split, "images")
        for video in sorted(os.listdir(path)):
            video_path = os.path.join(path, video)
            frames = sorted(
                [os.path.join(video_path, frame) for frame in os.listdir(video_path)]
            )
            self.video_sequences.append(frames)

    def __len__(self):
        return sum(
            [
                max(len(video) - self.sequence_length + 1, 1)
                for video in self.video_sequences
            ]
        )

    def idx_to_video(self, idx):
        """Returns the video folder and starting index for a given sequence index"""

        video_idx = 0
        while idx >= max(
            len(self.video_sequences[video_idx]) - self.sequence_length + 1, 1
        ):
            idx -= max(
                len(self.video_sequences[video_idx]) - self.sequence_length + 1, 1
            )
            video_idx += 1
        return video_idx, idx

    def __getitem__(self, idx):
        video_idx, start_idx = self.idx_to_video(idx)

        frame_paths = self.video_sequences[video_idx][
            start_idx : start_idx + self.sequence_length
        ]

        frames = [decode_png(read_file(frame_path)) for frame_path in frame_paths]

        out = {}
        out["frames"] = self.transform(torch.stack(frames))
        out["sequence_mask"] = torch.ones(self.sequence_length).bool()

        if self.split == "validation" or self.split == "test":
            mask_paths = [path.replace("images", "masks") for path in frame_paths]
            masks = torch.stack(
                [decode_png(read_file(mask_path)) for mask_path in mask_paths]
            )
            out["masks"] = (
                self.mask_transform(masks).to(dtype=torch.uint8).squeeze(0)[:, 0]
            )

        return out


class MOVIeDataset(pl.LightningDataModule):

    def __init__(self, dataset_config: dict, loader_config: dict):
        super().__init__()

        self.dataset_config = dataset_config
        self.loader_config = loader_config

    def setup(self, stage=None):  # type: ignore

        self.train_dataset = MOVIe(**self.dataset_config, split="train")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self.loader_config, shuffle=True)

    def val_dataloader(self):
        raise NotImplementedError

    def test_dataloader(self):
        raise NotImplementedError


if __name__ == "__main__":
    dataset = MOVIe(
        file_root="./data/ytvis",
        sequence_length=24,
        resolution=(336, 504),
        split="validation",
    )
    item_0 = dataset[0]
