# Copied and modified from https://github.com/singhgautam/steve/blob/evaluate/download_movi_with_masks.py
import os
import argparse
from tqdm import tqdm
import tensorflow_datasets as tfds
import torchvision.utils as vutils

from torchvision import transforms


parser = argparse.ArgumentParser()

parser.add_argument('--out_path', default='data/movi_e/256x256/processed')
parser.add_argument('--data_dir', default='data/movi_e/256x256/1.0.0')

parser.add_argument('--version', default='1.0.0')
parser.add_argument('--image_size', type=int, default=256)

args = parser.parse_args()

ds, ds_info = tfds.load(f"movi_e/{args.image_size}x{args.image_size}:{args.version}",
    data_dir=args.data_dir,
    with_info=True,
    download=False,
) # type: ignore

to_tensor = transforms.ToTensor()

print('Please be patient; it is usually very slow.')
for split in ['train', 'validation', 'test']:
    print(f'Processing {split} split')
    split_iter = iter(tfds.as_numpy(ds[split])) # type: ignore
    for i, record in tqdm(enumerate(split_iter)):
        video = record['video']
        if split != 'train':
            masks = record["segmentations"]
        T, *_ = video.shape

        # setup dirs
        path_vid_imgs = os.path.join(args.out_path, split, "images", f"{i:08}")
        os.makedirs(path_vid_imgs, exist_ok=True)

        path_vid_masks = os.path.join(args.out_path, split, "masks", f"{i:08}")
        os.makedirs(path_vid_masks, exist_ok=True)

        for t in range(T):
            img = video[t]
            img = to_tensor(img)
            vutils.save_image(img, os.path.join(path_vid_imgs, f"{t:08}.png"))
            if split != 'train':
                mask = masks[t] # type: ignore
                mask = to_tensor(mask)
                vutils.save_image(mask, os.path.join(path_vid_masks, f"{t:08}.png"))

