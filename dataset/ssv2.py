from pathlib import Path

import lightning as pl
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import torch
from nvidia.dali import pipeline_def  # type: ignore
from nvidia.dali.plugin.base_iterator import LastBatchPolicy
from nvidia.dali.plugin.pytorch import DALIGenericIterator

initial_prefetch_size = 8


@pipeline_def  # type: ignore
def video_pipe(
    filenames=[],
    labels=[],
    resolution=(224, 224),
    sequence_length=16,
    stride=1,
    step=1,
    shuffle=True,
    shard_id=0,
    num_shards=1,
):
    videos, labels, timestamps = fn.readers.video_resize(  # type: ignore
        device="gpu",
        filenames=filenames,
        labels=labels,
        resize_shorter=resolution[0],
        sequence_length=sequence_length,
        enable_timestamps=True,
        pad_sequences=True,
        shard_id=shard_id,
        stride=stride,
        step=step,
        num_shards=num_shards,
        random_shuffle=shuffle,
        name="Reader",
        initial_fill=initial_prefetch_size,
        prefetch_queue_depth=1,
        file_list_include_preceding_frame=True,
    )

    sequence_mask = timestamps >= 0

    coin = fn.random.coin_flip(probability=0.5)
    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
    videos = fn.crop_mirror_normalize(  # type: ignore
        videos,  # type: ignore
        crop=resolution,
        dtype=types.FLOAT,  # type: ignore
        mean=mean,
        std=std,
        output_layout="FCHW",
        mirror=coin,
    )

    return videos, labels, sequence_mask


class LightningWrapper(DALIGenericIterator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __next__(self):  # type: ignore
        out = super().__next__()
        out = out[0]

        frames = out["frames"]
        sequence_mask = out["sequence_mask"]

        n_frames = (sequence_mask >= 0).sum(dim=-1).clamp(min=1)

        S = frames.size(1)
        batch_size = frames.size(0)
        device = frames.device
        range_tensor = torch.arange(S, device=device).unsqueeze(0).repeat(batch_size, 1)

        j = range_tensor % n_frames.unsqueeze(1)

        j_expanded = (
            j.unsqueeze(2)
            .unsqueeze(3)
            .unsqueeze(4)
            .expand(-1, -1, frames.size(2), frames.size(3), frames.size(4))
        )

        cyclic_frames = torch.gather(frames, 1, j_expanded)

        out["frames"] = cyclic_frames
        out["n_frames"] = n_frames
        out["sequence_mask"][:] = 1

        return out


class VideoDataset(pl.LightningDataModule):

    def __init__(self, pipeline_config: dict):
        super().__init__()

        self.pipeline_config = pipeline_config
        root = Path(pipeline_config["root"])
        del self.pipeline_config["root"]

        train_dir = root / "train"
        self.train_files = list(train_dir.glob("**/*.mp4"))
        self.train_labels = [
            int(str(f.parent).split("/")[-1]) for f in self.train_files
        ]

        val_dir = root / "validation"
        self.val_files = list(val_dir.glob("**/*.mp4"))
        self.val_labels = [int(str(f.parent).split("/")[-1]) for f in self.val_files]

        test_dir = root / "test"
        self.test_files = list(test_dir.glob("**/*.mp4"))
        self.test_labels = [int(str(f.parent).split("/")[-1]) for f in self.test_files]

    def setup(self, stage=None):  # type: ignore
        print("Setting up video dataset...")
        device_id = self.trainer.local_rank  # type: ignore
        shard_id = self.trainer.global_rank  # type: ignore
        num_shards = self.trainer.world_size  # type: ignore

        pipeline_train = video_pipe(
            **self.pipeline_config,
            filenames=self.train_files,
            labels=self.train_labels,
            device_id=device_id,
            shard_id=shard_id,
            num_shards=num_shards,
        )
        self.train_loader = LightningWrapper(
            pipeline_train,
            reader_name="Reader",
            output_map=["frames", "labels", "sequence_mask"],
            last_batch_policy=LastBatchPolicy.DROP,
        )

        # pipeline_val = video_pipe(
        #     **self.pipeline_config,
        #     filenames=self.val_files,
        #     labels=self.val_labels,
        #     device_id=device_id,
        #     shard_id=shard_id,
        #     num_shards=num_shards,
        # )
        # self.val_loader = LightningWrapper(
        #     pipeline_val,
        #     reader_name="Reader",
        #     output_map=["frames", "labels", "sequence_mask"],
        #     last_batch_policy=LastBatchPolicy.DROP,
        # )
        #
        # pipeline_test = video_pipe(
        #     **self.pipeline_config,
        #     filenames=self.test_files,
        #     labels=self.test_labels,
        #     device_id=device_id,
        #     shard_id=shard_id,
        #     num_shards=num_shards,
        # )
        # self.test_loader = LightningWrapper(
        #     pipeline_test,
        #     reader_name="Reader",
        #     output_map=["frames", "labels", "sequence_mask"],
        #     last_batch_policy=LastBatchPolicy.DROP,
        # )

        print("Video dataset setup complete.")

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader
