import os

import lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.transforms import Compose, Resize, CenterCrop

FPS = 25
SECONDS = 5.12


class CLEVRERDataset(Dataset):
    def __init__(
        self, data_dir, split="train", n_frames=2 * FPS, resolution=[64, 96]
    ) -> None:
        super().__init__()
        self.data_dir = os.path.join(
            data_dir, split, "video_frames"
        )  # Updated path to match new structure
        self.n_frames = n_frames
        self.video_length = int(FPS * SECONDS)

        self.db = self._load_db()

        self.transform = Compose([Resize(resolution), CenterCrop(resolution)])

    def _load_db(self):
        db = []
        for video_name in os.listdir(self.data_dir):
            frame_dir = os.path.join(self.data_dir, video_name)
            frame_count = len(os.listdir(frame_dir))
            db.append((video_name, frame_count))
        return db

    def _index_to_entry(self, index):
        total_frames = sum([frame_count for _, frame_count in self.db])
        if index >= total_frames - self.n_frames:
            raise ValueError("Index out of range")

        for video_name, frame_count in self.db:
            if index + self.n_frames <= frame_count:
                return video_name, index, index + self.n_frames
            index -= frame_count - self.n_frames + 1

    def __len__(self):
        return sum(frame_count - self.n_frames + 1 for _, frame_count in self.db)

    def __getitem__(self, index):
        video_name, start, end = self._index_to_entry(index)
        frames_dir = os.path.join(self.data_dir, video_name)

        frames = [
            read_image(os.path.join(frames_dir, f"frame_{i}.jpg")).float() / 255.0
            for i in range(start, end)
        ]
        video = self.transform(torch.stack(frames))

        return video


# class CLEVRERDataset(Dataset):
#
#     def __init__(
#         self, data_dir, split="train", n_frames=2 * FPS, resolution=64
#     ) -> None:
#         super().__init__()
#         self.data_dir = os.path.join(data_dir, split)
#         self.n_frames = n_frames
#         self.video_length = int(FPS * SECONDS)
#
#         self.db = self._load_db()
#
#         self.transform = tv.transforms.Compose(
#             [
#                 tv.transforms.Normalize((0.5,), (0.5,)),
#                 tv.transforms.Resize(resolution),
#             ]
#         )
#
#     def _load_db(self):
#         db = []
#         for dir in os.listdir(os.path.join(self.data_dir)):
#             if not dir.startswith("annotation"):
#                 continue
#
#             for file in os.listdir(os.path.join(self.data_dir, dir)):
#                 annotation_path = os.path.join(self.data_dir, dir, file)
#                 video_path = annotation_path.replace("annotation", "video").replace(
#                     ".json", ".mp4"
#                 )  # noqa: E501
#
#                 db.append((annotation_path, video_path))
#
#         return db
#
#     def _index_to_entry(self, index):
#         for annotation_path, video_path in self.db:
#             if index < self.video_length - self.n_frames:
#                 return annotation_path, video_path, index, index + self.n_frames
#             index -= self.video_length - self.n_frames
#
#         raise ValueError("Index out of range")
#
#     def __len__(self):
#         return len(self.db) * (self.video_length - self.n_frames)
#
#     def __getitem__(self, index):
#         annotation_path, video_path, start, end = self._index_to_entry(index)
#
#         # TODO: load annotation
#
#         start_time = time.time()
#
#         video = (
#             tv.io.read_video(video_path, output_format="TCHW", pts_unit="sec")[0][
#                 start:end
#             ].float()
#             / 255.0
#         )
#
#         print("Time to load video:", time.time() - start_time)
#
#         video = self.transform(video)
#
#         return video


class CLEVRER(pl.LightningDataModule):

    def __init__(
        self,
        data_dir,
        batch_size=32,
        n_frames=2 * FPS,
        resolution=64,
        num_workers=4,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.n_frames = n_frames
        self.resolution = resolution
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train = CLEVRERDataset(
            self.data_dir,
            split="train",
            n_frames=self.n_frames,
            resolution=self.resolution,
        )
        self.val = CLEVRERDataset(
            self.data_dir,
            split="val",
            n_frames=self.n_frames,
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
            shuffle=False,
        )

    def test_dataloader(self):
        raise NotImplementedError


if __name__ == "__main__":
    CLVR = CLEVRERDataset("data/clevrer", split="train")

    print(CLVR[0].shape)
