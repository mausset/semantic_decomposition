import os
import torch
from torch.utils.data import Dataset, DataLoader
import lightning as pl
from torchvision.io import read_image
from torchvision.transforms import (
    Compose,
    Resize,
    CenterCrop,
    RandomHorizontalFlip,
    Normalize,
)

FPS = 25
SECONDS = 5.12


class CLEVRERDataset(Dataset):
    def __init__(
        self, data_dir, split="train", n_frames=2 * FPS, resolution=[64, 96], stride=1
    ) -> None:
        super().__init__()
        self.data_dir = os.path.join(
            data_dir, split, "video_frames"
        )  # Updated path to match new structure
        self.n_frames = n_frames
        self.stride = stride
        self.strided_length = n_frames * stride
        self.video_length = int(FPS * SECONDS)

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
        for video_name in os.listdir(self.data_dir):
            frame_dir = os.path.join(self.data_dir, video_name)
            frame_count = len(os.listdir(frame_dir))
            db.append((video_name, frame_count))
        return db

    def _index_to_entry(self, index):
        total_frames = sum([frame_count for _, frame_count in self.db])
        if index >= total_frames - self.strided_length:
            raise ValueError("Index out of range")

        for video_name, frame_count in self.db:
            if index + self.strided_length <= frame_count:
                return video_name, index, index + self.strided_length
            index -= frame_count - self.strided_length + 1

    def __len__(self):
        return sum(frame_count - self.strided_length + 1 for _, frame_count in self.db)

    def __getitem__(self, index):
        video_name, start, end = self._index_to_entry(index)
        frames_dir = os.path.join(self.data_dir, video_name)

        frames = [
            read_image(os.path.join(frames_dir, f"frame_{i}.jpg")).float() / 255.0
            for i in range(start, end, self.stride)
        ]
        video = self.transform(torch.stack(frames))

        return video


class CLEVRER(pl.LightningDataModule):

    def __init__(
        self,
        data_dir,
        batch_size=32,
        n_frames=2 * FPS,
        resolution=64,
        stride=1,
        num_workers=4,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.n_frames = n_frames
        self.resolution = resolution
        self.stride = stride
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train = CLEVRERDataset(
            self.data_dir,
            split="train",
            n_frames=self.n_frames,
            resolution=self.resolution,
            stride=self.stride,
        )
        self.val = CLEVRERDataset(
            self.data_dir,
            split="val",
            n_frames=self.n_frames,
            resolution=self.resolution,
            stride=self.stride,
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
            shuffle=False,
        )

    def test_dataloader(self):
        raise NotImplementedError


if __name__ == "__main__":
    CLVR = CLEVRERDataset("data/clevrer", split="train")

    print(CLVR[0].shape)
