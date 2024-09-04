#!/bin/bash

# Check if the root directory is provided as an argument
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <root_dir>"
  exit 1
fi

root_dir="$1"
processed_dir="${root_dir}/processed"

# Create the processed directory if it doesn't exist
mkdir -p "$processed_dir"

# Iterate through subdirectories of the root directory
for subdir in "$root_dir"/*; do
  if [ -d "$subdir" ]; then
    # Check if the video file exists
    video_path="${subdir}/video.mp4"
    if [ -f "$video_path" ]; then
      # Create a corresponding subdirectory in the processed directory
      processed_subdir="${processed_dir}/$(basename "$subdir")"
      mkdir -p "$processed_subdir"

      # Define the output path for the processed video
      processed_video_path="${processed_subdir}/processed.mp4"

      # Apply the ffmpeg command
      ffmpeg -i "$video_path" -vf "scale=796:448,crop=448:448:174:0" -b:v 3M "$processed_video_path" -y
    else
      echo "Video file not found in $subdir"
    fi
  fi
done

echo "Processing complete."
