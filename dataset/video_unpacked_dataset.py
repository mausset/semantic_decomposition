import os

import lightning.pytorch as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.v2 import Compose, Normalize, Resize, ToDtype, ToImage


class VideoUnpackedDataset(Dataset):
    def __init__(self, file_root, sequence_length, resolution):
        """
        Args:
            file_root (string): Directory with all the video folders.
            sequence_length (int): Number of frames in each sequence.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.file_root = file_root
        self.sequence_length = sequence_length
        self.transform = Compose(
            [
                Resize(resolution),
                ToImage(),
                ToDtype(torch.float, scale=True),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # Collect all video folders
        self.video_folders = [
            os.path.join(self.file_root, d)
            for d in os.listdir(self.file_root)
            if os.path.isdir(os.path.join(self.file_root, d))
        ]
        self.video_folders.sort()  # Optional: sort for consistent indexing

        # Precompute the number of sequences available per video folder
        self.video_sequences = []
        for video_folder in self.video_folders:
            frame_paths = sorted(
                [
                    os.path.join(video_folder, f)
                    for f in os.listdir(video_folder)
                    if f.endswith(".jpg")
                ]
            )
            if len(frame_paths) >= self.sequence_length:
                num_sequences = len(frame_paths) - self.sequence_length + 1
                self.video_sequences.extend(
                    [(video_folder, i) for i in range(num_sequences)]
                )

    def __len__(self):
        return len(self.video_sequences)

    def __getitem__(self, idx):
        video_folder, start_idx = self.video_sequences[idx]
        frame_paths = sorted(
            [
                os.path.join(video_folder, f)
                for f in os.listdir(video_folder)
                if f.endswith(".jpg")
            ]
        )

        selected_frames = frame_paths[start_idx : start_idx + self.sequence_length]

        frames = [
            Image.open(frame_path).convert("RGB") for frame_path in selected_frames
        ]
        frames = [self.transform(frame) for frame in frames]
        frames = torch.stack(frames)

        return frames


class VideoDataset(pl.LightningDataModule):

    def __init__(self, dataset_config: dict, loader_config: dict):
        super().__init__()

        self.dataset_config = dataset_config
        self.loader_config = loader_config

    def setup(self, stage=None):  # type: ignore

        self.train_dataset = VideoUnpackedDataset(**self.dataset_config)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self.loader_config, shuffle=True)

    def val_dataloader(self):
        return None

    def test_dataloader(self):
        raise NotImplementedError
