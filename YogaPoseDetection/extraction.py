import os
import cv2
import json
import numpy as np
import mediapipe as mp
from collections import defaultdict

# Paths
BASE_DIR = os.path.dirname(__file__)
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
OUTPUT_FILE = os.path.join(BASE_DIR, "angles.json")

# Flexible angle expansion for realistic webcam use
AUTO_EXPAND = 25.0   # degrees to expand from midpoint

mp_pose = mp.solutions.pose

# Joint definitions
JOINTS = {
    "left_knee": ("LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"),
    "right_knee": ("RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"),
    "left_elbow": ("LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"),
    "right_elbow": ("RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"),
    "left_shoulder": ("LEFT_ELBOW", "LEFT_SHOULDER", "LEFT_HIP"),
    "right_shoulder": ("RIGHT_ELBOW", "RIGHT_SHOULDER", "RIGHT_HIP"),
    "left_hip": ("RIGHT_HIP", "LEFT_HIP", "LEFT_KNEE"),
    "right_hip": ("LEFT_HIP", "RIGHT_HIP", "RIGHT_KNEE"),
    "spine": ("LEFT_HIP", "RIGHT_HIP", "NOSE"),
    "neck": ("LEFT_SHOULDER", "NOSE", "RIGHT_SHOULDER"),
}

# Angle calculation
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    v1, v2 = a - b, c - b
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0:
        return None
    value = np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(value)))

# Safe coordinate fetch
def get_point(landmarks, name):
    lm = landmarks[mp_pose.PoseLandmark[name].value]
    return [lm.x, lm.y, lm.z]

POSE_ANGLES = {}

# Process dataset
with mp_pose.Pose(static_image_mode=True) as pose:
    for pose_folder in sorted(os.listdir(DATASET_DIR)):
        folder_path = os.path.join(DATASET_DIR, pose_folder)
        if not os.path.isdir(folder_path):
            continue

        print(f"\nProcessing: {pose_folder}")

        joint_angles = defaultdict(list)

        for file in os.listdir(folder_path):
            if not file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img = cv2.imread(os.path.join(folder_path, file))
            if img is None:
                continue

            results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if not results.pose_landmarks:
                continue

            lm = results.pose_landmarks.landmark

            for joint, (p1, p2, p3) in JOINTS.items():
                try:
                    a, b, c = get_point(lm, p1), get_point(lm, p2), get_point(lm, p3)
                    angle = calculate_angle(a, b, c)
                    if angle is not None:
                        joint_angles[joint].append(angle)
                except:
                    pass

        # Summaries
        summary = {}

        for joint, values in joint_angles.items():
            if len(values) == 0:
                continue

            arr = np.array(values)
            min_angle, max_angle = arr.min(), arr.max()

            # Midpoint-based expansion for real-time tolerance
            mid = (min_angle + max_angle) / 2
            expanded_min = mid - AUTO_EXPAND
            expanded_max = mid + AUTO_EXPAND

            summary[joint] = {
                "angle_min": round(expanded_min, 1),
                "angle_max": round(expanded_max, 1),
                "tip": f"Adjust your {joint.replace('_', ' ')}."
            }

        POSE_ANGLES[pose_folder] = summary
        print(f"Saved {len(summary)} joint ranges.")

# Save JSON output
with open(OUTPUT_FILE, "w") as f:
    json.dump(POSE_ANGLES, f, indent=4)

print(f"\nExtraction complete. Saved to {OUTPUT_FILE}")
