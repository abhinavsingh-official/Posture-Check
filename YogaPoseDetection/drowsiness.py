"""Eye-aspect-ratio (EAR) drowsiness/blink detection - the same metric
from the classic Soukupova & Cech formula most EAR tutorials use, but
computed from MediaPipe Face Mesh landmarks instead of dlib's 68-point
model. This project already depends on MediaPipe for everything else
(Pose, Hands); pulling in dlib just for this one mode would mean a second,
heavier face-landmark dependency (plus a separate ~100MB .dat model file
to download) for a formula MediaPipe's own landmarks already support.

EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|) for six points around an eye
(p1/p4 the corners, p2/p3/p5/p6 the upper/lower lid) - it drops sharply
when the eye closes since the vertical distances shrink while the
horizontal one doesn't."""

from dataclasses import dataclass

import numpy as np

# MediaPipe Face Mesh landmark indices for a 6-point eye outline, in the
# same corner/upper/upper/corner/lower/lower order the EAR formula expects.
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

EAR_THRESHOLD = 0.21
CONSECUTIVE_FRAMES_FOR_ALERT = 15  # roughly half a second at ~30fps


def _dist(a, b) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def eye_aspect_ratio(eye_points: list[tuple[float, float]]) -> float:
    p1, p2, p3, p4, p5, p6 = eye_points
    vertical = _dist(p2, p6) + _dist(p3, p5)
    horizontal = _dist(p1, p4)
    if horizontal == 0:
        return 1.0  # degenerate frame - treat as "eyes open" rather than divide by zero
    return vertical / (2.0 * horizontal)


def average_ear(face_landmarks, frame_width: int, frame_height: int) -> float:
    def points(indices):
        return [
            (face_landmarks.landmark[i].x * frame_width, face_landmarks.landmark[i].y * frame_height)
            for i in indices
        ]

    left_ear = eye_aspect_ratio(points(LEFT_EYE))
    right_ear = eye_aspect_ratio(points(RIGHT_EYE))
    return (left_ear + right_ear) / 2.0


@dataclass
class DrowsinessMonitor:
    threshold: float = EAR_THRESHOLD
    frames_required: int = CONSECUTIVE_FRAMES_FOR_ALERT
    consecutive_closed_frames: int = 0
    total_blinks: int = 0
    _was_closed: bool = False

    def update(self, ear: float) -> bool:
        """Returns True if the alert threshold has just been crossed this
        frame. Also tracks a simple blink count (a short closed streak that
        never reached the drowsiness threshold still counts as one blink)."""
        is_closed = ear < self.threshold

        if is_closed:
            self.consecutive_closed_frames += 1
        else:
            if self._was_closed and 0 < self.consecutive_closed_frames < self.frames_required:
                self.total_blinks += 1
            self.consecutive_closed_frames = 0

        self._was_closed = is_closed
        return self.consecutive_closed_frames >= self.frames_required
