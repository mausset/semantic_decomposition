import cv2
import os
import concurrent.futures
from pathlib import Path
import argparse
from tqdm import tqdm


# Function to resize and center crop a frame
def resize_and_crop(frame, target_size):
    h, w, _ = frame.shape
    scale = target_size / min(h, w)
    resized_frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

    h_resized, w_resized, _ = resized_frame.shape
    start_x = (w_resized - target_size) // 2
    start_y = (h_resized - target_size) // 2
    cropped_frame = resized_frame[
        start_y : start_y + target_size, start_x : start_x + target_size
    ]

    return cropped_frame


# Function to save a single frame to a directory
def save_frame(frame, frame_num, dir_path):
    frame_filename = os.path.join(dir_path, f"frame_{frame_num:06d}.jpg")
    cv2.imwrite(frame_filename, frame)


# Function to process a chunk of frames
def process_frames(
    start_frame,
    end_frame,
    video_path,
    dir_base,
    chunk_id,
    target_size,
    frame_offset,
    skip_n,
    progress,
):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_num = frame_offset
    dir_path = f"{dir_base}/chunk_{chunk_id:03d}"
    Path(dir_path).mkdir(parents=True, exist_ok=True)

    while start_frame < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        if (start_frame // skip_n) * skip_n == start_frame:
            frame = resize_and_crop(frame, target_size)
            save_frame(frame, frame_num, dir_path)
            frame_num += 1
        start_frame += 1
        progress.update(1)

    cap.release()


# Main function to decode video in parallel
def decode_videos(root_path, chunk_size=10000, target_size=448, skip_n=1):
    video_files = list(Path(root_path).rglob("*.mp4"))
    dir_base = os.path.join(root_path, "decoded")
    Path(dir_base).mkdir(parents=True, exist_ok=True)

    total_frames = 0
    video_infos = []
    for video_file in video_files:
        cap = cv2.VideoCapture(str(video_file))
        frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        video_infos.append((str(video_file), frames_count))
        total_frames += frames_count

    frame_offset = 0
    chunks = []
    chunk_id = 0

    for video_path, frames_count in video_infos:
        start_frame = 0
        while start_frame < frames_count:
            end_frame = min(
                start_frame + chunk_size - (frame_offset % chunk_size), frames_count
            )
            chunks.append(
                (
                    start_frame,
                    end_frame,
                    video_path,
                    dir_base,
                    chunk_id,
                    target_size,
                    frame_offset,
                    skip_n,
                )
            )
            frame_offset += end_frame - start_frame
            start_frame = end_frame
            if frame_offset % chunk_size == 0:
                chunk_id += 1

    with tqdm(total=total_frames, desc="Decoding videos", unit="frame") as progress:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(process_frames, *chunk, progress) for chunk in chunks
            ]
            for future in concurrent.futures.as_completed(futures):
                future.result()  # Ensure exceptions are raised if any


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Decode all MP4 videos in a directory into multiple directories of frames."
    )
    parser.add_argument(
        "root_path", type=str, help="Root directory to search for MP4 video files."
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=10000,
        help="Number of frames per directory. Default is 10,000.",
    )
    parser.add_argument(
        "--target_size",
        type=int,
        default=448,
        help="Target resolution for the smallest side of the frame. Default is 448.",
    )
    parser.add_argument(
        "--skip_n",
        type=int,
        default=1,
        help="Number of frames to skip between saves. Default is 1 (no skipping).",
    )

    args = parser.parse_args()

    decode_videos(args.root_path, args.chunk_size, args.target_size, args.skip_n)
