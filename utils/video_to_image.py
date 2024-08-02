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
    start_frame, end_frame, video_path, dir_base, chunk_id, target_size, progress
):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_num = start_frame
    dir_path = f"{dir_base}/chunk_{chunk_id:03d}"
    Path(dir_path).mkdir(parents=True, exist_ok=True)

    while frame_num < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        frame = resize_and_crop(frame, target_size)
        save_frame(frame, frame_num, dir_path)
        frame_num += 1
        progress.update(1)

    cap.release()


# Main function to decode video in parallel
def decode_video(video_path, dir_base, chunk_size=10000, target_size=448):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    chunks = [
        (i, min(i + chunk_size, total_frames), video_path, dir_base, idx, target_size)
        for idx, i in enumerate(range(0, total_frames, chunk_size))
    ]

    with tqdm(total=total_frames, desc="Decoding video", unit="frame") as progress:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(process_frames, *chunk, progress) for chunk in chunks
            ]
            for future in concurrent.futures.as_completed(futures):
                future.result()  # Ensure exceptions are raised if any


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Decode an MP4 video into multiple directories of frames."
    )
    parser.add_argument("video_path", type=str, help="Path to the MP4 video file.")
    parser.add_argument("output_dir", type=str, help="Directory to save the frames.")
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

    args = parser.parse_args()

    decode_video(args.video_path, args.output_dir, args.chunk_size, args.target_size)
