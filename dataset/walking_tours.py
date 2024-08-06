import lightning as pl


from nvidia.dali.plugin.base_iterator import LastBatchPolicy
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from nvidia.dali import pipeline_def  # type: ignore
import nvidia.dali.fn as fn
import nvidia.dali.types as types


initial_prefetch_size = 8


@pipeline_def  # type: ignore
def video_pipe(
    file_root="",
    filenames=[],
    resolution=(224, 224),
    sequence_length=16,
    stride=6,
    step=1,
    shuffle=True,
    shard_id=0,
    num_shards=1,
):
    print(f"file_root: {file_root}")
    videos = fn.readers.video_resize(
        device="gpu",
        file_root=file_root,
        filenames=filenames,
        resize_shorter=resolution[0],
        sequence_length=sequence_length,
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
    if isinstance(videos, list):
        videos = videos[0]

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

    return videos


class LightningWrapper(DALIGenericIterator):
    def __init__(self, *kargs, **kvargs):
        super().__init__(*kargs, **kvargs)

    def __next__(self):  # type: ignore
        out = super().__next__()
        return out[0]["data"]


class WalkingTours(pl.LightningDataModule):

    def __init__(self, pipeline_config: dict):
        super().__init__()

        self.pipeline_config = pipeline_config

    def setup(self, stage=None):  # type: ignore
        print("Setting up WalkingTours dataset...")
        device_id = self.trainer.local_rank  # type: ignore
        shard_id = self.trainer.global_rank  # type: ignore
        num_shards = self.trainer.world_size  # type: ignore

        wt_pipeline_train = video_pipe(
            **self.pipeline_config,
            device_id=device_id,
            shard_id=shard_id,
            num_shards=num_shards,
        )
        self.train_loader = LightningWrapper(
            wt_pipeline_train,
            reader_name="Reader",
            output_map=["data"],
            last_batch_policy=LastBatchPolicy.DROP,
        )

        print("WalkingTours dataset setup complete.")

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return None

    def test_dataloader(self):
        raise NotImplementedError
