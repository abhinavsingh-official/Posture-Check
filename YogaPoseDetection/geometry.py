"""Shared landmark/angle math - used by both extraction.py (building
angles.json from the dataset) and the live analysis/workout modes, so
there's one copy of this instead of two drifting ones."""

import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose
POSE_LMS = mp_pose.PoseLandmark

JOINTS = {
    "left_knee": ("LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"),
    "right_knee": ("RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"),
    "left_elbow": ("LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"),
    "right_elbow": ("RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"),
    "left_shoulder": ("LEFT_ELBOW", "LEFT_SHOULDER", "LEFT_HIP"),
    "right_shoulder": ("RIGHT_ELBOW", "RIGHT_SHOULDER", "RIGHT_HIP"),
    "left_hip": ("RIGHT_HIP", "LEFT_HIP", "LEFT_KNEE"),
    "right_hip": ("LEFT_HIP", "RIGHT_HIP", "RIGHT_KNEE"),
    "neck": ("LEFT_SHOULDER", "NOSE", "RIGHT_SHOULDER"),
    "spine": ("LEFT_HIP", "RIGHT_HIP", "NOSE"),
}


def calculate_angle(a, b, c) -> float | None:
    a, b, c = np.array(a), np.array(b), np.array(c)
    v1, v2 = a - b, c - b
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0:
        return None
    value = np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(value)))


def get_landmark_point(landmarks, name: str) -> list[float]:
    lm = landmarks[POSE_LMS[name].value]
    return [lm.x, lm.y, lm.z]


def get_all_angles(landmarks, joints: dict = JOINTS) -> dict[str, float]:
    angles = {}
    for joint_name, (p1, p2, p3) in joints.items():
        try:
            a = get_landmark_point(landmarks, p1)
            b = get_landmark_point(landmarks, p2)
            c = get_landmark_point(landmarks, p3)
        except (KeyError, IndexError):
            continue
        angle = calculate_angle(a, b, c)
        if angle is not None:
            angles[joint_name] = angle
    return angles


def full_body_visible(landmarks, threshold: float = 0.80) -> bool:
    visible = [lm for lm in landmarks if 0 < lm.x < 1 and 0 < lm.y < 1]
    return len(visible) / len(landmarks) >= threshold
