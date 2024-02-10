import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import cv2
from tqdm import tqdm


def create_parser():
    parser = argparse.ArgumentParser(description="Preprocess Clevrer dataset")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/path/to/clevrer",
        help="path to clevrer dataset",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count(),
        help="number of workers for parallel processing",
    )
    return parser


def process_video(video_dir_data):
    data_dir, dir, vid = video_dir_data
    video_dir = os.path.join(data_dir, dir, vid)
    video_name = vid.split(".")[0]
    os.makedirs(os.path.join(data_dir, "video_frames", video_name), exist_ok=True)
    video = cv2.VideoCapture(video_dir)
    frame_count = 0
    while True:
        ret, frame = video.read()
        if not ret:
            break
        frame_name = f"frame_{frame_count}.jpg"
        frame_path = os.path.join(data_dir, "video_frames", video_name, frame_name)
        cv2.imwrite(frame_path, frame)
        frame_count += 1
    video.release()


def main(args):
    data_dir = args.data_dir
    workers = args.workers

    os.makedirs(os.path.join(data_dir, "video_frames"), exist_ok=True)
    videos_to_process = []

    for dir in os.listdir(data_dir):
        if (
            dir.endswith("frames")
            or dir.startswith("annotation")
            or dir.startswith(".DS_Store")
        ):
            continue

        for vid in os.listdir(os.path.join(data_dir, dir)):
            videos_to_process.append((data_dir, dir, vid))

    with ProcessPoolExecutor(max_workers=workers) as executor:
        list(
            tqdm(
                executor.map(process_video, videos_to_process),
                total=len(videos_to_process),
            )
        )

    print("Preprocessing Clevrer dataset is done!")


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
