"""Draw in the air by moving your index fingertip - a different interaction
model from the gesture classifier (gestures.py classifies a static hand
shape; this tracks one point continuously over time).

Pen state uses a pinch (thumb tip close to index fingertip) to lift the
pen, rather than the toolbar-relative y-position checks typical of similar
tutorials - lifting the pen by pinching is the same mental model as a real
pen, and it means the toolbar only has to handle color/clear selection, not
also double as a pen-up zone."""

from dataclasses import dataclass, field

import cv2
import numpy as np

TOOLBAR_HEIGHT = 70
COLORS = {
    "blue": (255, 120, 80),
    "green": (120, 255, 120),
    "red": (80, 80, 255),
    "yellow": (80, 220, 255),
}
PINCH_THRESHOLD = 40  # pixels; thumb-to-index distance below this = pen lifted


@dataclass
class AirCanvasState:
    current_color: str = "blue"
    strokes: dict[str, list[list[tuple[int, int]]]] = field(
        default_factory=lambda: {color: [[]] for color in COLORS}
    )

    def handle_point(self, index_tip: tuple[int, int], thumb_tip: tuple[int, int], frame_width: int) -> str:
        """Updates stroke state for one frame, returns a short status
        string ("drawing", "pen up", or "toolbar") for the UI to show."""
        x, y = index_tip
        pinch_distance = _distance(index_tip, thumb_tip)

        if y < TOOLBAR_HEIGHT:
            self._handle_toolbar(x, frame_width)
            return "toolbar"

        if pinch_distance < PINCH_THRESHOLD:
            # Pen lifted - end the current stroke so the next drawn point
            # starts a new line instead of connecting across the gap.
            if self.strokes[self.current_color][-1]:
                self.strokes[self.current_color].append([])
            return "pen up"

        self.strokes[self.current_color][-1].append((x, y))
        return "drawing"

    def _handle_toolbar(self, x: int, frame_width: int) -> None:
        zone_width = frame_width / (len(COLORS) + 1)
        zone = int(x / zone_width)
        if zone == 0:
            self.clear()
        else:
            names = list(COLORS.keys())
            if zone - 1 < len(names):
                self.current_color = names[zone - 1]

    def clear(self) -> None:
        self.strokes = {color: [[]] for color in COLORS}


def _distance(a: tuple[int, int], b: tuple[int, int]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def draw_toolbar(frame, active_color: str) -> None:
    width = frame.shape[1]
    zone_width = width // (len(COLORS) + 1)
    cv2.rectangle(frame, (0, 0), (zone_width, TOOLBAR_HEIGHT), (40, 40, 40), -1)
    cv2.putText(frame, "CLEAR", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    for i, (name, color) in enumerate(COLORS.items(), start=1):
        x0 = i * zone_width
        cv2.rectangle(frame, (x0, 0), (x0 + zone_width, TOOLBAR_HEIGHT), color, -1)
        if name == active_color:
            cv2.rectangle(frame, (x0, 0), (x0 + zone_width, TOOLBAR_HEIGHT), (255, 255, 255), 3)


def draw_strokes(frame, strokes: dict[str, list[list[tuple[int, int]]]]) -> None:
    for color_name, color_strokes in strokes.items():
        color = COLORS[color_name]
        for stroke in color_strokes:
            for i in range(1, len(stroke)):
                cv2.line(frame, stroke[i - 1], stroke[i], color, 4)
