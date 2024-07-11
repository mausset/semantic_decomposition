import lightning as pl


from nvidia.dali.plugin.base_iterator import LastBatchPolicy
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from nvidia.dali import pipeline_def  # type: ignore
import nvidia.dali.fn as fn
import nvidia.dali.types as types


initial_prefetch_size = 16


@pipeline_def  # type: ignore
def video_pipe(filenames, resolution=(224, 224), sequence_length=16, stride=6, step=1):
    videos = fn.readers.video_resize(
        device="gpu",
        filenames=filenames,
        resize_shorter=resolution[0],
        sequence_length=sequence_length,
        shard_id=0,
        stride=stride,
        step=step,
        num_shards=1,
        random_shuffle=True,
        name="Reader",
        initial_fill=initial_prefetch_size,
        prefetch_queue_depth=1,
    )

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


class WalkingTours(pl.LightningDataModule):

    def __init__(self, pipeline_config: dict):
        super().__init__()

        self.pipeline_config = pipeline_config
        print(pipeline_config["filenames"])

    def setup(self, stage=None):  # type: ignore
        print("Setting up WalkingTours dataset...")

        class LightningWrapper(DALIGenericIterator):
            def __init__(self, *kargs, **kvargs):
                super().__init__(*kargs, **kvargs)

            def __next__(self):  # type: ignore
                out = super().__next__()
                # DDP is used so only one pipeline per process
                # also we need to transform dict returned by
                # DALIClassificationIterator to iterable and squeeze the lables
                return out[0]["data"]

        wt_pipeline_train = video_pipe(**self.pipeline_config)
        self.train_loader = LightningWrapper(
            wt_pipeline_train,
            reader_name="Reader",
            output_map=["data"],
            last_batch_policy=LastBatchPolicy.DROP,
        )

        wt_pipeline_val = video_pipe(**self.pipeline_config)
        self.val_loader = LightningWrapper(
            wt_pipeline_val,
            reader_name="Reader",
            output_map=["data"],
            last_batch_policy=LastBatchPolicy.DROP,
        )
        print("WalkingTours dataset setup complete.")

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        raise NotImplementedError
