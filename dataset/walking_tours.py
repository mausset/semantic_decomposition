# from torchvision.datasets.utils import list_dir
# from torchvision.datasets.folder import make_dataset
# from torchvision.datasets.video_utils import VideoClips
# from torchvision.datasets import VisionDataset
# import torchvision
# import os
# from PIL import Image
import torch
import decord

import numpy as np

from torch.utils.data import DataLoader
from torchvision.transforms import (
    Compose,
    Resize,
    CenterCrop,
    RandomHorizontalFlip,
    Normalize,
    Lambda,
)
import lightning as pl

# import random


decord.bridge.set_bridge("torch")


class DecordInit(object):

    def __init__(self, num_threads=4, **kwargs):
        self.num_threads = num_threads
        self.ctx = decord.cpu(0)
        self.kwargs = kwargs

    def __call__(self, filename):

        reader = decord.VideoReader(
            filename, ctx=self.ctx, num_threads=self.num_threads
        )
        return reader

    def __repr__(self):
        repr_str = (
            f"{self.__class__.__name__}("
            f"sr={self.sr},"  # type: ignore
            f"num_threads={self.num_threads})"
        )
        return repr_str


class WTDataset1Vid(torch.utils.data.Dataset):  # type: ignore

    def __init__(self, root, num_frames, stride, transform=None):

        self.path = root

        self.transform = transform
        self.num_frames = num_frames
        self.stride = stride
        self.v_decoder = DecordInit()
        v_reader = self.v_decoder(self.path)
        total_frames = len(v_reader)

        self.total_frames = total_frames

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index):
        while True:
            try:

                v_reader = self.v_decoder(self.path)

                # Sampling video frames
                start_frame_ind = index
                end_frame_ind = start_frame_ind + (self.num_frames * self.stride)

                frame_indice = np.arange(
                    start_frame_ind, end_frame_ind, self.stride, dtype=int
                )

                video = v_reader.get_batch(frame_indice)
                del v_reader
                break
            except Exception as e:
                print(e)

        if self.transform is not None:
            video = self.transform(video)

        return video

    def __len__(self):
        return self.total_frames - (self.num_frames * self.stride)


def permute_and_norm(x):
    return x.permute(0, 3, 1, 2) / 255.0


class WalkingTours(pl.LightningDataModule):

    def __init__(self, dataset_config: dict, loader_config: dict, resolution=224):
        super().__init__()

        self.transform = Compose(
            [
                Lambda(permute_and_norm),
                Resize(resolution),
                CenterCrop(resolution),
                RandomHorizontalFlip(),
                Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        self.dataset_config = dataset_config
        self.loader_config = loader_config

    def setup(self, stage=None):
        print("Setting up WalkingTours dataset...")
        self.train = WTDataset1Vid(**self.dataset_config, transform=self.transform)
        self.val = WTDataset1Vid(**self.dataset_config, transform=self.transform)
        print("WalkingTours dataset setup complete.")

    def train_dataloader(self):
        return DataLoader(self.train, **self.loader_config)

    def val_dataloader(self):

        config = self.loader_config.copy()
        config["num_workers"] = 1

        return DataLoader(self.val, **config)

    def test_dataloader(self):
        raise NotImplementedError
