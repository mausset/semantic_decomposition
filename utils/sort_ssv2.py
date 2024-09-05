import os
import shutil
import json
import csv
import sys


def load_labels(labels_file):
    with open(labels_file, "r") as f:
        labels = json.load(f)
    return labels


def load_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def load_test_answers(test_answers_file):
    test_answers = {}
    with open(test_answers_file, "r") as f:
        reader = csv.reader(f, delimiter=";")
        for row in reader:
            test_answers[row[0]] = row[1]
    return test_answers


def clean_template(template):
    # Replace "[" and "]" with an empty string
    clean_temp = template.replace("[", "").replace("]", "")
    return clean_temp


def map_templates_to_ids(labels):
    template_to_id = {}
    for label, label_id in labels.items():
        clean_label = clean_template(label)
        template_to_id[clean_label] = label_id
    return template_to_id


def sort_videos(root_dir, labels_dir):
    # Load label mappings
    labels_file = os.path.join(labels_dir, "labels.json")
    labels = load_labels(labels_file)
    template_to_id = map_templates_to_ids(labels)

    # Define paths
    processed_dir = os.path.join(root_dir, "processed")
    sorted_dir = os.path.join(root_dir, "processed_sorted")

    # Create the sorted directory structure
    for split in ["train", "validation", "test"]:
        for label_id in template_to_id.values():
            os.makedirs(os.path.join(sorted_dir, split, label_id), exist_ok=True)

    # Sort train and validation videos
    for split in ["train", "validation"]:
        json_file = os.path.join(labels_dir, f"{split}.json")
        data = load_json(json_file)
        for entry in data:
            video_id = entry["id"]
            template = entry["template"]
            clean_template_str = clean_template(template)
            label_id = template_to_id[clean_template_str]
            src_video_path = os.path.join(processed_dir, f"{video_id}.mp4")
            dst_video_path = os.path.join(
                sorted_dir, split, label_id, f"{video_id}.mp4"
            )
            shutil.copy(src_video_path, dst_video_path)

    # Sort test videos
    test_json_file = os.path.join(labels_dir, "test.json")
    test_data = load_json(test_json_file)
    test_answers_file = os.path.join(labels_dir, "test-answers.csv")
    test_answers = load_test_answers(test_answers_file)

    for entry in test_data:
        video_id = entry["id"]
        label = test_answers[video_id]
        clean_label = clean_template(label)
        label_id = template_to_id[clean_label]
        src_video_path = os.path.join(processed_dir, f"{video_id}.mp4")
        dst_video_path = os.path.join(sorted_dir, "test", label_id, f"{video_id}.mp4")
        shutil.copy(src_video_path, dst_video_path)

    print("Sorting complete.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <root_directory>")
        sys.exit(1)

    root_dir = sys.argv[1]
    labels_dir = os.path.join(root_dir, "labels")
    sort_videos(root_dir, labels_dir)
