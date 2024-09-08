import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm


def process_video(input_file, output_file):
    """Runs the ffmpeg command to convert the input file to .mp4 using NVIDIA H.264 with yuv420p."""
    command = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-i",
        input_file,
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p",
        "-c:v",
        "h264_nvenc",  # Use NVIDIA H.264 encoder
        "-preset",
        "fast",  # Adjust this based on your desired speed/quality tradeoff
        "-cq",
        "28",  # Constant quality setting, adjust lower for better quality
        "-threads",
        "0",
        "-y",  # Overwrite output file if it exists
        output_file,
    ]
    subprocess.run(command, check=True)


def main(root_path, num_workers):
    root_path = Path(root_path).resolve()
    output_path = root_path.parent / "processed"
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all .webm files in the root directory
    video_files = list(root_path.rglob("*.webm"))

    if not video_files:
        print("No .webm files found in the specified root path.")
        return

    # Prepare tasks for parallel processing
    tasks = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for file in video_files:
            output_file = output_path / (file.stem + ".mp4")
            tasks.append(executor.submit(process_video, str(file), str(output_file)))

        # Use tqdm for a progress bar
        for _ in tqdm(as_completed(tasks), total=len(tasks), desc="Processing videos"):
            pass

    print(f"Processing complete! Processed files are in {output_path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python process_videos.py <root_path> <num_workers>")
        sys.exit(1)

    root_path = sys.argv[1]
    num_workers = int(sys.argv[2])

    main(root_path, num_workers)
