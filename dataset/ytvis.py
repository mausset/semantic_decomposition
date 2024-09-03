import json
import os

import lightning.pytorch as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision.transforms.v2 import (
    CenterCrop,
    Compose,
    Normalize,
    Resize,
    ToDtype,
    ToImage,
)


class YTVIS(Dataset):
    def __init__(self, file_root, sequence_length, resolution, split="train"):
        """
        Args:
            file_root (string): Directory with all the video folders.
            sequence_length (int): Number of frames in each sequence.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.file_root = file_root
        self.sequence_length = sequence_length
        self.split = split
        self.transform = Compose(
            [
                Resize(resolution[0]),
                CenterCrop(resolution),
                ToImage(),
                ToDtype(torch.float, scale=True),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.seg_transform = Compose(
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

    def __len__(self):
        return sum(
            [
                max(len(video) - self.sequence_length, 1)
                for video in self.video_sequences
            ]
        )

    def remove_border(
        self,
        img,
        bbox=None,
    ):
        if bbox is None:
            bbox = img.getbbox()
        return img.crop(bbox), bbox

    def idx_to_video(self, idx):
        """Returns the video folder and starting index for a given sequence index"""

        video_idx = 0
        while idx >= max(
            (len(self.video_sequences[video_idx]) - self.sequence_length), 1
        ):
            idx -= max(len(self.video_sequences[video_idx]) - self.sequence_length, 1)
            video_idx += 1
        return video_idx, idx

    def __getitem__(self, idx):
        video_idx, start_idx = self.idx_to_video(idx)

        frame_paths = self.video_sequences[video_idx][
            start_idx : start_idx + self.sequence_length
        ]

        padding = self.sequence_length - len(frame_paths)

        frames = [Image.open(frame_path).convert("RGB") for frame_path in frame_paths]

        out_dict = {}
        frames, bbox = zip(*[self.remove_border(frame) for frame in frames])
        frames = torch.stack([self.transform(frame) for frame in frames])
        frames = torch.cat([frames, torch.zeros(padding, *frames.shape[1:])])
        out_dict["frames"] = frames

        frame_mask = torch.ones(self.sequence_length)
        if padding > 0:
            frame_mask[-padding:] = 0
        out_dict["sequence_mask"] = frame_mask.bool()

        if self.split == "val":
            seg_paths = [
                path.replace("JPEGImages", "Annotations").replace("jpg", "png")
                for path in frame_paths
            ]
            masks = [Image.open(seg_path) for seg_path in seg_paths]
            masks, _ = zip(
                *[self.remove_border(frame, seg) for frame, seg in zip(masks, bbox)]
            )
            masks = torch.stack([self.seg_transform(seg) for seg in masks]).to(
                torch.uint8
            )
            out_dict["masks"] = torch.cat(
                [masks, torch.zeros(padding, *masks.shape[1:], dtype=torch.long)]
            ).squeeze(1)

        return out_dict


class YTVISDataset(pl.LightningDataModule):

    def __init__(self, dataset_config: dict, loader_config: dict):
        super().__init__()

        self.dataset_config = dataset_config
        self.loader_config = loader_config

    def setup(self, stage=None):  # type: ignore

        self.train_dataset = YTVIS(**self.dataset_config, split="train")
        self.val_dataset = YTVIS(**self.dataset_config, split="val")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self.loader_config, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, **self.loader_config, shuffle=False)

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
