from functools import cache
import json
import os
from random import Random

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision.io import read_file, decode_jpeg, decode_png
from torchvision.transforms.v2 import (
    CenterCrop,
    Compose,
    Normalize,
    Resize,
    ToDtype,
    ToImage,
    RandomHorizontalFlip,
)


class YTVIS(Dataset):
    def __init__(self, file_root, sequence_length, resolution, split="train", repeat=1):
        """
        Args:
            file_root (string): Directory with all the video folders.
            sequence_length (int): Number of frames in each sequence.
        """
        self.file_root = file_root
        self.sequence_length = sequence_length
        self.split = split
        self.repeat = repeat

        self.transform = Compose(
            [
                Resize(resolution[0]),
                CenterCrop(resolution),
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
                CenterCrop(resolution),
                ToImage(),
                ToDtype(torch.long),
            ],
        )

        # Collect all video folders
        self.video_paths = []
        if split == "train":
            for split in ["train", "valid", "test"]:
                split_file_ann = os.path.join(file_root, split, split + ".json")
                ann_file = json.load(open(split_file_ann, "r"))
                for i, video in enumerate(ann_file["videos"]):
                    # Skip the validation subset
                    if split == "train" and 600 <= i < 900:
                        continue

                    video_path = os.path.join(
                        file_root,
                        split,
                        "JPEGImages",
                        video["file_names"][0].split("/")[0],
                    )
                    self.video_paths.append(video_path)
        elif split == "val":
            split_file_ann = os.path.join(file_root, "train", "train" + ".json")
            ann_file = json.load(open(split_file_ann, "r"))
            for video in ann_file["videos"][600:900]:  # same subset as SOLV
                video_path = os.path.join(
                    file_root,
                    "train",
                    "JPEGImages",
                    video["file_names"][0].split("/")[0],
                )
                self.video_paths.append(video_path)
        self.video_paths.sort()

        # Precompute the number of sequences available per video folder
        self.video_sequences = []
        for video_path in self.video_paths:
            frame_paths = sorted(
                [
                    os.path.join(video_path, f)
                    for f in os.listdir(video_path)
                    if f.endswith(".jpg")
                ]
            )

            self.video_sequences.append(frame_paths)

        max_length = max([len(video) for video in self.video_sequences])
        assert (
            max_length >= sequence_length
        ), "Sequence length is longer than the longest video."

    @cache
    def __len__(self):
        return (
            sum(
                [
                    max(len(video) - self.sequence_length + 1, 1)
                    for video in self.video_sequences
                ]
            )
            * self.repeat
        )

    def idx_to_video(self, idx):
        """Returns the video folder and starting index for a given sequence index"""

        idx = idx % (len(self) // self.repeat)

        video_idx = 0
        while idx >= max(
            len(self.video_sequences[video_idx]) - self.sequence_length + 1, 1
        ):
            idx -= max(
                len(self.video_sequences[video_idx]) - self.sequence_length + 1, 1
            )
            video_idx += 1
        return video_idx, idx

    def repeat_sequence(self, frames):

        new_frames = []
        while len(new_frames) < self.sequence_length:
            new_frames += frames

        return new_frames[: self.sequence_length]

    def __getitem__(self, idx):
        video_idx, start_idx = self.idx_to_video(idx)

        frame_paths = self.video_sequences[video_idx][
            start_idx : start_idx + self.sequence_length
        ]

        out = {}
        frames = [decode_jpeg(read_file(frame_path)) for frame_path in frame_paths]
        out["n_frames"] = len(frames)

        frames = self.repeat_sequence(frames)

        padding = self.sequence_length - len(frames)
        frames = self.transform(torch.stack(frames))
        out["frames"] = torch.cat(
            [frames, torch.zeros(padding, *frames.shape[1:], dtype=frames.dtype)], dim=0
        )
        out["sequence_mask"] = torch.zeros(self.sequence_length).bool()
        out["sequence_mask"][: len(frames)] = True

        if self.split == "val":
            mask_paths = [
                path.replace("JPEGImages", "Annotations").replace("jpg", "png")
                for path in frame_paths
            ]
            masks = torch.stack(
                [decode_png(read_file(mask_path)) for mask_path in mask_paths]
            )

            masks = self.mask_transform(masks).to(dtype=torch.uint8).squeeze(0)
            masks = torch.cat(
                [masks, torch.zeros(padding, *masks.shape[1:], dtype=masks.dtype)],
                dim=0,
            )

        return out


class YTVISDataset(pl.LightningDataModule):

    def __init__(self, dataset_config: dict, loader_config: dict):
        super().__init__()

        self.dataset_config = dataset_config
        self.loader_config = loader_config

    def setup(self, stage=None):  # type: ignore

        self.train_dataset = YTVIS(**self.dataset_config, split="train")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self.loader_config, shuffle=True)

    def val_dataloader(self):
        raise NotImplementedError

    def test_dataloader(self):
        raise NotImplementedError


if __name__ == "__main__":
    dataset = YTVIS(
        file_root="./data/ytvis",
        sequence_length=36,
        resolution=(336, 504),
        split="val",
    )
    item_0 = dataset[0]
