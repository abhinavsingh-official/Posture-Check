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


# Sticker icons are drawn procedurally with OpenCV primitives rather than
# loaded from image files - no external asset/licensing question, and it
# keeps the whole gesture -> sticker pipeline self-contained in code.
STICKER_COLORS = {
    "wave": (120, 220, 255),
    "fist_burst": (80, 80, 255),
    "sparkle": (255, 220, 120),
    "thumbs_up": (120, 255, 150),
}


def _draw_sparkle(canvas, center, radius, color):
    cx, cy = center
    for angle_deg in range(0, 360, 45):
        angle = np.deg2rad(angle_deg)
        x2 = int(cx + radius * np.cos(angle))
        y2 = int(cy + radius * np.sin(angle))
        cv2.line(canvas, (cx, cy), (x2, y2), color, 3)
    cv2.circle(canvas, center, max(radius // 3, 4), WHITE, -1)


def _draw_wave(canvas, center, radius, color):
    cx, cy = center
    for i in range(5):
        x = cx - radius + i * (radius // 2)
        cv2.ellipse(canvas, (x, cy), (radius // 6, radius // 2), 0, 180, 360, color, 4)


def _draw_fist_burst(canvas, center, radius, color):
    cx, cy = center
    cv2.circle(canvas, center, radius // 2, color, -1)
    for angle_deg in range(0, 360, 30):
        angle = np.deg2rad(angle_deg)
        x1 = int(cx + (radius // 2) * np.cos(angle))
        y1 = int(cy + (radius // 2) * np.sin(angle))
        x2 = int(cx + radius * np.cos(angle))
        y2 = int(cy + radius * np.sin(angle))
        cv2.line(canvas, (x1, y1), (x2, y2), color, 2)


def _draw_thumbs_up(canvas, center, radius, color):
    cx, cy = center
    cv2.rectangle(canvas, (cx - radius // 3, cy), (cx + radius // 3, cy + radius), color, -1)
    cv2.circle(canvas, (cx, cy - radius // 3), radius // 3, color, -1)
    cv2.rectangle(canvas, (cx - radius // 6, cy - radius), (cx + radius // 6, cy - radius // 3), color, -1)


_STICKER_DRAWERS = {
    "wave": _draw_wave,
    "fist_burst": _draw_fist_burst,
    "sparkle": _draw_sparkle,
    "thumbs_up": _draw_thumbs_up,
}


def draw_sticker(frame, position: tuple[int, int], sticker: str, radius: int = 60, alpha: float = 0.85) -> None:
    """Composites a procedural sticker icon at `position`, alpha-blended
    onto `frame` in place. `sticker` is one of gestures.GESTURE_STICKERS'
    values; an unrecognized name is a no-op rather than an error, since a
    gesture with no mapped sticker is a normal (not exceptional) case."""
    drawer = _STICKER_DRAWERS.get(sticker)
    if drawer is None:
        return
    color = STICKER_COLORS.get(sticker, ACCENT)

    overlay = frame.copy()
    cv2.circle(overlay, position, radius + 15, (30, 30, 30), -1)
    drawer(overlay, position, radius, color)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, dst=frame)
