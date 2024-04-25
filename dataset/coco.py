import os
import random

import lightning as pl
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CocoCaptions
from torchvision.io import read_image
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Grayscale,
    Normalize,
    RandomHorizontalFlip,
    Resize,
    ToTensor,
)


class COCODataset(Dataset):
    def __init__(
        self,
        data_dir,
        split="train",
        resolution=[64, 96],
    ) -> None:
        super().__init__()
        self.data_dir = os.path.join(data_dir, split)

        self.db = self._load_db()

        self.transform = Compose(
            [
                Resize(resolution),
                CenterCrop(resolution),
                RandomHorizontalFlip(),
                Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    def _load_db(self):
        db = []
        for filename in os.listdir(self.data_dir):
            db.append(filename)

        return sorted(db)

    def _index_to_entry(self, index):
        return self.db[index]

    def __len__(self):
        return len(self.db)

    def __getitem__(self, index):
        filename = self._index_to_entry(index)
        file_path = os.path.join(self.data_dir, filename)

        image = read_image(file_path).float() / 255.0
        if image.shape[0] == 1:
            return self.__getitem__(index + 1)

        image = self.transform(image)

        return image


class COCO(pl.LightningDataModule):

    def __init__(
        self,
        data_dir,
        batch_size=32,
        resolution=64,
        num_workers=4,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.resolution = resolution
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train = COCODataset(
            self.data_dir,
            split="train",
            resolution=self.resolution,
        )
        self.val = COCODataset(
            self.data_dir,
            split="val",
            resolution=self.resolution,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            shuffle=True,
        )

    def test_dataloader(self):
        raise NotImplementedError


class COCOCap(pl.LightningDataModule):

    def __init__(
        self,
        root_dir,
        batch_size=32,
        resolution=64,
        num_workers=4,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.resolution = resolution
        self.num_workers = num_workers

        self.transform = Compose(
            [
                ToTensor(),
                Grayscale(num_output_channels=3),
                Resize(resolution),
                CenterCrop(resolution),
                RandomHorizontalFlip(),
                Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    def setup(self, stage=None):
        train_root = os.path.join(self.root_dir, "coco2017/train")
        train_ann = os.path.join(self.root_dir, "annotations/captions_train2017.json")
        print(train_root)
        print(train_ann)
        self.train = CocoCaptions(
            root=train_root,
            annFile=train_ann,
            transform=self.transform,
            target_transform=random.choice,
        )

        val_root = os.path.join(self.root_dir, "coco2017/val")
        val_ann = os.path.join(self.root_dir, "annotations/captions_val2017.json")
        print(val_root)
        print(val_ann)
        self.val = CocoCaptions(
            root=val_root,
            annFile=val_ann,
            transform=self.transform,
            target_transform=random.choice,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            shuffle=True,
        )
