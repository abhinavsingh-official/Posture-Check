"""Builds angles.json from the reference images in dataset/ - for each
pose folder, computes every joint angle on every image and records the
observed min/max, expanded by a fixed tolerance for real-time webcam use
(small poses/angles don't hold perfectly still frame to frame)."""

import json
import os
from collections import defaultdict

import cv2
import numpy as np
import mediapipe as mp

from geometry import JOINTS, calculate_angle, get_landmark_point

BASE_DIR = os.path.dirname(__file__)
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
OUTPUT_FILE = os.path.join(BASE_DIR, "angles.json")
AUTO_EXPAND = 25.0  # degrees to expand from the midpoint for webcam tolerance

mp_pose = mp.solutions.pose


def angles_for_image(pose_model, image_path: str) -> dict[str, float] | None:
    """One image's joint angles, or None if no pose was detected - the
    per-image unit both normal extraction and leave-one-out evaluation
    (evaluate.py) are built from."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    results = pose_model.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not results.pose_landmarks:
        return None

    lm = results.pose_landmarks.landmark
    angles = {}
    for joint, (p1, p2, p3) in JOINTS.items():
        try:
            a, b, c = get_landmark_point(lm, p1), get_landmark_point(lm, p2), get_landmark_point(lm, p3)
        except (KeyError, IndexError):
            continue
        angle = calculate_angle(a, b, c)
        if angle is not None:
            angles[joint] = angle
    return angles


def collect_dataset_angles(dataset_dir: str) -> dict[str, list[dict[str, float]]]:
    """Every usable image's per-image angles, grouped by pose folder -
    the raw material both extract_pose_angles (below) and evaluate.py
    build on, computed once rather than twice."""
    per_pose: dict[str, list[dict[str, float]]] = {}
    with mp_pose.Pose(static_image_mode=True) as pose_model:
        for pose_folder in sorted(os.listdir(dataset_dir)):
            folder_path = os.path.join(dataset_dir, pose_folder)
            if not os.path.isdir(folder_path):
                continue
            images = []
            for file in os.listdir(folder_path):
                if not file.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                angles = angles_for_image(pose_model, os.path.join(folder_path, file))
                if angles:
                    images.append(angles)
            per_pose[pose_folder] = images
    return per_pose


def summarize_pose_angles(per_pose: dict[str, list[dict[str, float]]]) -> dict:
    pose_angles = {}
    for pose_folder, images in per_pose.items():
        joint_angles = defaultdict(list)
        for angles in images:
            for joint, value in angles.items():
                joint_angles[joint].append(value)
        pose_angles[pose_folder] = _summarize(joint_angles)
        print(f"{pose_folder}: {len(images)} usable image(s), {len(pose_angles[pose_folder])} joint range(s)")
    return pose_angles


def extract_pose_angles(dataset_dir: str) -> dict:
    return summarize_pose_angles(collect_dataset_angles(dataset_dir))


def _summarize(joint_angles: dict[str, list[float]]) -> dict:
    summary = {}
    for joint, values in joint_angles.items():
        if not values:
            continue
        arr = np.array(values)
        mid = (arr.min() + arr.max()) / 2
        summary[joint] = {
            "angle_min": round(mid - AUTO_EXPAND, 1),
            "angle_max": round(mid + AUTO_EXPAND, 1),
            "tip": f"Adjust your {joint.replace('_', ' ')}.",
        }
    return summary


if __name__ == "__main__":
    pose_angles = extract_pose_angles(DATASET_DIR)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(pose_angles, f, indent=4)
    print(f"\nExtraction complete. Saved to {OUTPUT_FILE}")
