import os

import lightning as pl
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Grayscale,
    InterpolationMode,
    Normalize,
    PILToTensor,
    RandomHorizontalFlip,
    Resize,
    ToTensor,
)


class COCODataset(Dataset):
    NUM_CLASSES = 81

    # fmt: off
    CAT_LIST = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19,
    20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
    64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 
    89, 90]
    # fmt: on

    assert (NUM_CLASSES) == len(set(CAT_LIST))

    def __init__(
        self,
        data_dir,
        ann_file,
        split="train",
        resolution=[64, 96],
    ) -> None:
        super().__init__()

        self.data_dir = data_dir
        self.coco = COCO(ann_file)

        self.split = split

        self.ids = self.coco.getImgIds()

        self.transform_train = Compose(
            [
                Resize(resolution),
                Grayscale(num_output_channels=3),
                CenterCrop(resolution),
                RandomHorizontalFlip(p=0.5),
                ToTensor(),
                Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        self.transform_val = Compose(
            [
                Resize(resolution),
                Grayscale(num_output_channels=3),
                CenterCrop(resolution),
                ToTensor(),
                Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        self.transform_mask = Compose(
            [
                Resize(resolution, interpolation=InterpolationMode.NEAREST),
                CenterCrop(resolution),
                PILToTensor(),
            ]
        )

    def _make_img_gt_point_pair(self, index):
        coco = self.coco
        img_id = self.ids[index]
        img_metadata = coco.loadImgs(img_id)[0]
        path = img_metadata["file_name"]
        _img = Image.open(os.path.join(self.data_dir, path)).convert("RGB")
        cocotarget = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        _targets = self._gen_seg_n_insta_masks(
            cocotarget, img_metadata["height"], img_metadata["width"]
        )
        mask_class = Image.fromarray(_targets[0])
        mask_instance = Image.fromarray(_targets[1])
        mask_ignore = Image.fromarray(_targets[2])
        return _img, mask_instance, mask_class, mask_ignore

    def _gen_seg_n_insta_masks(self, target, h, w):
        seg_mask = np.zeros((h, w), dtype=np.uint8)
        insta_mask = np.zeros((h, w), dtype=np.uint8)
        ignore_mask = np.zeros((h, w), dtype=np.uint8)
        for i, instance in enumerate(target, 1):
            rle = mask_utils.frPyObjects(instance["segmentation"], h, w)
            m = mask_utils.decode(rle)
            cat = instance["category_id"]
            if cat in self.CAT_LIST:
                c = self.CAT_LIST.index(cat)
            else:
                continue
            if len(m.shape) < 3:
                seg_mask[:, :] += (seg_mask == 0) * (m * c)
                insta_mask[:, :] += (insta_mask == 0) * (m * i)
                ignore_mask[:, :] += m
            else:
                seg_mask[:, :] += (seg_mask == 0) * (
                    ((np.sum(m, axis=2)) > 0) * c
                ).astype(np.uint8)
                insta_mask[:, :] += (insta_mask == 0) * (
                    ((np.sum(m, axis=2)) > 0) * i
                ).astype(np.uint8)
                ignore_mask[:, :] += (((np.sum(m, axis=2)) > 0) * 1).astype(np.uint8)

        # Ignore overlaps
        ignore_mask = (ignore_mask > 1).astype(np.uint8)

        all_masks = np.stack([seg_mask, insta_mask, ignore_mask])
        return all_masks

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):

        img, mask_instance, mask_class, mask_ignore = self._make_img_gt_point_pair(
            index
        )

        # Mask not used
        if self.split == "train":
            img = self.transform_train(img)
            return img

        if self.split == "val":
            img = self.transform_val(img)
            mask_instance = self.transform_mask(mask_instance).squeeze().long()
            mask_class = self.transform_mask(mask_class).squeeze().long()
            mask_ignore = (
                self.transform_mask(mask_ignore).squeeze().long().unsqueeze(0)
            )  # Weird
            return img, mask_instance, mask_class, mask_ignore


class COCOSeg(pl.LightningDataModule):

    def __init__(
        self,
        data_dir,
        ann_dir,
        batch_size=32,
        resolution=64,
        num_workers=4,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.ann_dir = ann_dir
        self.batch_size = batch_size
        self.resolution = resolution
        self.num_workers = num_workers

    def setup(self, stage=None):
        train_dir = os.path.join(self.data_dir, "train2017")
        train_ann = os.path.join(self.ann_dir, "instances_train2017.json")

        self.train = COCODataset(
            train_dir,
            train_ann,
            "train",
            resolution=self.resolution,
        )

        val_dir = os.path.join(self.data_dir, "val2017")
        val_ann = os.path.join(self.ann_dir, "instances_val2017.json")

        self.val = COCODataset(
            val_dir,
            val_ann,
            "val",
            resolution=self.resolution,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            shuffle=True,
            drop_last=True,
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
