"""Hand-sign classification - wraps a vendored pretrained TFLite model
(see models/gesture/ATTRIBUTION.md) rather than training one from scratch;
the new work here is turning a classified gesture into an actual on-screen
response (renderer.draw_sticker), which the source project doesn't do at
all - it only prints the label as text.

The landmark preprocessing (flatten to wrist-relative coordinates, then
normalize by the largest absolute value) matches the vendored model's own
training preprocessing exactly - the model's accuracy depends on getting
this normalization bit-for-bit consistent with how it was trained."""

import csv
from pathlib import Path

import numpy as np
from ai_edge_litert.interpreter import Interpreter

MODEL_DIR = Path(__file__).resolve().parent / "models" / "gesture"
MODEL_PATH = MODEL_DIR / "keypoint_classifier.tflite"
LABELS_PATH = MODEL_DIR / "keypoint_classifier_label.csv"

# Each recognized gesture maps to a distinct on-screen sticker response -
# see renderer.draw_sticker for what each one actually looks like.
GESTURE_STICKERS = {
    "Open": "wave",
    "Close": "fist_burst",
    "Pointer": "sparkle",
    "OK": "thumbs_up",
}


def load_labels() -> list[str]:
    with open(LABELS_PATH, encoding="utf-8-sig") as f:
        return [row[0] for row in csv.reader(f)]


class GestureClassifier:
    def __init__(self):
        self.interpreter = Interpreter(model_path=str(MODEL_PATH), num_threads=1)
        self.interpreter.allocate_tensors()
        self.input_index = self.interpreter.get_input_details()[0]["index"]
        self.output_index = self.interpreter.get_output_details()[0]["index"]
        self.labels = load_labels()

    def classify(self, hand_landmarks, frame_width: int, frame_height: int) -> str:
        """hand_landmarks: a MediaPipe NormalizedLandmarkList for one hand.
        frame_width/height matter here, not just a formality - the
        vendored model was trained on landmarks converted to pixel space
        *before* computing wrist-relative, max-normalized coordinates, so
        skipping that pixel conversion (and using raw 0-1 landmark.x/y
        directly) would feed it differently-scaled input than it saw
        during training whenever the frame isn't square."""
        flattened = preprocess_landmarks(hand_landmarks, frame_width, frame_height)
        self.interpreter.set_tensor(self.input_index, np.array([flattened], dtype=np.float32))
        self.interpreter.invoke()
        result = self.interpreter.get_tensor(self.output_index)
        label_index = int(np.argmax(np.squeeze(result)))
        return self.labels[label_index]


def preprocess_landmarks(hand_landmarks, frame_width: int, frame_height: int) -> list[float]:
    points = [[lm.x * frame_width, lm.y * frame_height] for lm in hand_landmarks.landmark]
    base_x, base_y = points[0]
    relative = [[x - base_x, y - base_y] for x, y in points]
    flat = [v for point in relative for v in point]
    max_value = max(abs(v) for v in flat) or 1.0
    return [v / max_value for v in flat]
