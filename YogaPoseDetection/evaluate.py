"""Leave-one-out evaluation of auto_detect_pose: for every image that
has at least one sibling in its pose folder, hold it out, rebuild that
pose's angle ranges from the remaining images only, and check whether
auto-detect correctly identifies it against the full set of poses.

Honest framing for the README: with only a handful of images per pose,
this is a small-sample estimate, not a rigorous benchmark - reported as
such, not inflated."""

import os

from extraction import DATASET_DIR, collect_dataset_angles, summarize_pose_angles
from feedback import auto_detect_pose


def evaluate(dataset_dir: str = DATASET_DIR) -> dict:
    per_pose = collect_dataset_angles(dataset_dir)
    testable_poses = {pose: images for pose, images in per_pose.items() if len(images) >= 2}

    correct = 0
    total = 0
    per_pose_results: dict[str, list[bool]] = {}

    for pose, images in testable_poses.items():
        per_pose_results[pose] = []
        for held_out_index in range(len(images)):
            held_out = images[held_out_index]
            remaining = {
                p: imgs if p != pose else imgs[:held_out_index] + imgs[held_out_index + 1:]
                for p, imgs in per_pose.items()
            }
            angle_data = summarize_pose_angles(remaining)
            predicted, _ = auto_detect_pose(held_out, angle_data)
            is_correct = predicted == pose
            per_pose_results[pose].append(is_correct)
            correct += int(is_correct)
            total += 1

    return {
        "overall_accuracy": correct / total if total else None,
        "total_evaluated": total,
        "per_pose_accuracy": {
            pose: sum(results) / len(results) for pose, results in per_pose_results.items()
        },
        "skipped_single_image_poses": [p for p, imgs in per_pose.items() if len(imgs) < 2],
    }


if __name__ == "__main__":
    import json

    results = evaluate()
    print(json.dumps(results, indent=2))
