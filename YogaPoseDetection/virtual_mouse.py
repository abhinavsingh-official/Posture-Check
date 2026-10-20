"""Control the real system cursor with one hand - index fingertip moves
the pointer, a closed-finger pinch clicks. Reuses geometry.calculate_angle
(already built for pose joint angles) for the finger-curl check instead of
duplicating the same three-point angle formula under a different name, as
the inspiration project does.

This is the one mode in this app that reaches outside the browser/camera
sandbox and actually moves your real cursor - it's opt-in behind a
separate confirmation, not just a camera-start checkbox, and a short
no-hand timeout auto-disables it so a dropped hand-tracking lock doesn't
leave the cursor stuck mid-screen."""

import time
from dataclasses import dataclass

import pyautogui
from pynput.mouse import Button, Controller

from geometry import calculate_angle

pyautogui.FAILSAFE = True  # moving the real cursor to a screen corner aborts pyautogui actions

CLICK_ANGLE_THRESHOLD = 50  # degrees; a finger curled tighter than this counts as "closed"
AUTO_DISABLE_AFTER_SECONDS = 3  # no-hand timeout before cursor control turns itself off

mouse = Controller()


@dataclass
class HandLandmarks2D:
    """Plain (x, y) pairs for the 21 MediaPipe hand landmarks - decoupled
    from the MediaPipe landmark object itself so the click-detection logic
    below is trivial to unit test with plain tuples."""
    points: list[tuple[float, float]]

    def __getitem__(self, index: int) -> tuple[float, float]:
        return self.points[index]

    @classmethod
    def from_mediapipe(cls, hand_landmarks) -> "HandLandmarks2D":
        return cls([(lm.x, lm.y) for lm in hand_landmarks.landmark])


def _finger_angle(landmarks: HandLandmarks2D, mcp: int, pip: int, tip: int) -> float | None:
    return calculate_angle(landmarks[mcp], landmarks[pip], landmarks[tip])


def is_index_extended(landmarks: HandLandmarks2D) -> bool:
    angle = _finger_angle(landmarks, 5, 6, 8)
    return angle is not None and angle > 90


def is_pinch_click(landmarks: HandLandmarks2D) -> bool:
    """A click is an index finger curled tight (not extended) while the
    thumb and index-finger-base stay apart - distinguishes a deliberate
    curl-to-click from just resting the hand in a loose fist."""
    index_angle = _finger_angle(landmarks, 5, 6, 8)
    if index_angle is None:
        return False
    thumb_index_base_dist = _distance(landmarks[4], landmarks[5])
    return index_angle < CLICK_ANGLE_THRESHOLD and thumb_index_base_dist > 0.1


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def move_cursor(landmarks: HandLandmarks2D, screen_width: int, screen_height: int) -> None:
    index_tip = landmarks[8]
    pyautogui.moveTo(int(index_tip[0] * screen_width), int(index_tip[1] * screen_height), _pause=False)


def click() -> None:
    mouse.press(Button.left)
    mouse.release(Button.left)


class VirtualMouseController:
    """Wraps the per-frame decision (move vs. click vs. idle) plus the
    no-hand auto-disable timer, so app.py's camera loop only has to call
    one method per frame instead of re-deriving this state machine."""

    def __init__(self):
        self.last_click_time = 0.0
        self.last_hand_seen = time.time()
        self.click_cooldown = 0.6

    def update(self, landmarks: HandLandmarks2D | None, screen_width: int, screen_height: int) -> str:
        if landmarks is None:
            if time.time() - self.last_hand_seen > AUTO_DISABLE_AFTER_SECONDS:
                return "idle (no hand)"
            return "no hand"

        self.last_hand_seen = time.time()

        if is_pinch_click(landmarks):
            now = time.time()
            if now - self.last_click_time > self.click_cooldown:
                click()
                self.last_click_time = now
                return "click"
            return "click (cooldown)"

        if is_index_extended(landmarks):
            move_cursor(landmarks, screen_width, screen_height)
            return "moving"

        return "hand idle"
