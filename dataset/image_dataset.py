import lightning.pytorch as pl
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.pipeline import pipeline_def
from nvidia.dali.plugin.base_iterator import LastBatchPolicy
from nvidia.dali.plugin.pytorch import DALIGenericIterator

INITIAL_FILL = 8


@pipeline_def
def ssl_image_pipeline(root, resolution):
    images, _ = fn.readers.file(
        file_root=root,
        random_shuffle=True,
        initial_fill=INITIAL_FILL,
        name="Reader",
    )
    images = fn.decoders.image(
        images,
        device="mixed",
    )

    coin = fn.random.coin_flip(probability=0.5)
    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
    transformed_images = fn.crop_mirror_normalize(
        images,
        crop=resolution,
        dtype=types.FLOAT,  # type: ignore
        mean=mean,
        std=std,
        output_layout="CHW",
        mirror=coin,
    )
    return transformed_images


class DALIImageDataset(pl.LightningDataModule):
    def __init__(self, root, resolution, batch_size, num_threads=4):
        super().__init__()
        self.root = root
        self.resolution = resolution
        self.batch_size = batch_size
        self.num_threads = num_threads

    def setup(self, stage=None):
        class LightningWrapper(DALIGenericIterator):
            def __init__(self, *kargs, **kvargs):
                super().__init__(*kargs, **kvargs)

            def __next__(self):  # type: ignore
                out = super().__next__()
                return out[0]["data"]

        pipeline = ssl_image_pipeline(
            root=self.root,
            resolution=self.resolution,
            batch_size=self.batch_size,
            num_threads=self.num_threads,
            device_id=0,
        )

        self.train_loader = LightningWrapper(
            pipeline,
            reader_name="Reader",
            output_map=["data"],
            last_batch_policy=LastBatchPolicy.DROP,
        )

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return None
