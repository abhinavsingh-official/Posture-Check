"""Compares a user's live joint angles against the reference ranges in
angles.json and produces corrective hints."""

import numpy as np

GOOD_SCORE_THRESHOLD = 10  # average degrees-off before a pose counts as "correct"


def compare_pose(user_angles: dict[str, float], pose_name: str, angle_data: dict) -> tuple[float, list[str]]:
    data = angle_data[pose_name]
    diffs = []
    hints = []
    for joint, val in user_angles.items():
        if joint not in data:
            continue
        mn, mx = data[joint]["angle_min"], data[joint]["angle_max"]
        if val < mn:
            diff = mn - val
            diffs.append(diff)
            hints.append(f"{joint.replace('_', ' ').title()}: increase by {diff:.1f}")
        elif val > mx:
            diff = val - mx
            diffs.append(diff)
            hints.append(f"{joint.replace('_', ' ').title()}: decrease by {diff:.1f}")
        else:
            diffs.append(0)
    score = float(np.mean(diffs)) if diffs else float("inf")
    return score, hints


def auto_detect_pose(user_angles: dict[str, float], angle_data: dict) -> tuple[str | None, float]:
    best_pose, best_score = None, float("inf")
    for pose_name in angle_data:
        score, _ = compare_pose(user_angles, pose_name, angle_data)
        if score < best_score:
            best_pose, best_score = pose_name, score
    return best_pose, best_score


def is_pose_correct(score: float) -> bool:
    return score < GOOD_SCORE_THRESHOLD
