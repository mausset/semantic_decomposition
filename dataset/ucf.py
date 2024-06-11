import torch
from torch.utils.data import DataLoader
from torchvision.datasets import UCF101
from torchvision.transforms import (
    Compose,
    Resize,
    CenterCrop,
    RandomHorizontalFlip,
    Normalize,
    Lambda,
)
import lightning as pl


def collate_fn(batch):
    images, *_ = zip(*batch)
    images = torch.stack(images)
    return images


def to_float(x):
    return x / 255.0


class UCF(pl.LightningDataModule):

    def __init__(self, ucf_config: dict, loader_config: dict, resolution=224):
        super().__init__()

        self.transform = Compose(
            [
                Resize(resolution),
                CenterCrop(resolution),
                RandomHorizontalFlip(),
                Lambda(to_float),
                Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        self.ucf_config = ucf_config
        self.loader_config = loader_config

    def setup(self, stage=None):
        print("Setting up UCF101 dataset...")
        self.train = UCF101(**self.ucf_config, transform=self.transform, train=True)
        self.val = UCF101(**self.ucf_config, transform=self.transform, train=False)
        print("UCF101 dataset setup complete.")

    def train_dataloader(self):
        return DataLoader(self.train, **self.loader_config, collate_fn=collate_fn)

    def val_dataloader(self):

        config = self.loader_config.copy()
        config["num_workers"] = 1

        return DataLoader(self.val, **config, collate_fn=collate_fn)

    def test_dataloader(self):
        raise NotImplementedError
