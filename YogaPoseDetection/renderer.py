"""All OpenCV drawing: the MediaPipe skeleton overlay (color-coded per
joint - green if within range, red if not) and the dark-mode side panel."""

import cv2
import numpy as np
import mediapipe as mp

from geometry import JOINTS, POSE_LMS

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

GREEN = (120, 255, 120)
RED = (80, 80, 255)
WHITE = (255, 255, 255)
ACCENT = (255, 180, 120)
DEFAULT_LANDMARK_COLOR = (200, 200, 200)

# Which skeleton connections (landmark index pairs) belong to each named
# joint, so a bad joint's hint can highlight the actual lines on screen.
_JOINT_CONNECTIONS: dict[str, list[tuple[int, int]]] = {}
for _joint, (_p1, _p2, _p3) in JOINTS.items():
    i1, i2, i3 = POSE_LMS[_p1].value, POSE_LMS[_p2].value, POSE_LMS[_p3].value
    _JOINT_CONNECTIONS[_joint] = [(i1, i2), (i2, i3)]


def draw_skeleton(frame, landmarks, bad_joints: set[str]) -> None:
    """Draws the full MediaPipe skeleton, with any joint named in
    `bad_joints` rendered red instead of green."""
    bad_connections = {
        conn for joint in bad_joints for conn in _JOINT_CONNECTIONS.get(joint, [])
    }

    connection_spec = {}
    for connection in mp_pose.POSE_CONNECTIONS:
        is_bad = connection in bad_connections or (connection[1], connection[0]) in bad_connections
        color = RED if is_bad else GREEN
        connection_spec[connection] = mp_drawing.DrawingSpec(color=color, thickness=3)

    mp_drawing.draw_landmarks(
        frame,
        landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(color=DEFAULT_LANDMARK_COLOR, thickness=2, circle_radius=3),
        connection_drawing_spec=connection_spec,
    )


def bad_joints_from_hints(hints: list[str]) -> set[str]:
    """Hints are rendered as "Left Knee: increase by 12.0" - recover the
    joint key ("left_knee") so draw_skeleton can highlight it."""
    return {hint.split(":")[0].strip().lower().replace(" ", "_") for hint in hints}


def draw_side_panel(title: str, status_line: str, hints: list[str], live_angles: dict, height: int, width: int = 380):
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel[:] = (25, 25, 25)

    y = 45
    cv2.putText(panel, title, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)
    y += 35
    cv2.putText(panel, status_line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREEN, 2)
    y += 40

    if live_angles:
        cv2.putText(panel, "Live Angles:", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ACCENT, 2)
        y += 30
        for joint, value in live_angles.items():
            cv2.putText(panel, f"{joint}: {int(value)}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1)
            y += 22
        y += 15

    if hints:
        cv2.putText(panel, "Hints:", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ACCENT, 2)
        y += 28
        for hint in hints:
            cv2.putText(panel, hint, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1)
            y += 22

    return panel


def compose_frame(frame, panel):
    panel = cv2.resize(panel, (panel.shape[1], frame.shape[0]))
    return np.hstack([frame, panel])
