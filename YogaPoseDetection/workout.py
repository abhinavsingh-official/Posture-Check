"""Rep counting for dynamic exercises - a different problem from yoga's
"hold a static pose": here we track a joint angle crossing down/up
thresholds and count one rep per full cycle, the standard approach used
by most AI-trainer rep counters."""

from dataclasses import dataclass, field

from geometry import calculate_angle, get_landmark_point

# Each exercise tracks one representative angle. "down_below"/"up_above"
# are crossed in sequence to count a rep - this is a simple state
# machine (up -> down -> up), not a raw single-threshold check, so a
# noisy frame near the threshold doesn't double-count.
EXERCISES: dict[str, dict] = {
    "squat": {
        "joint": ("LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"),
        "down_below": 100,
        "up_above": 160,
        "label": "Knee angle",
    },
    "pushup": {
        "joint": ("LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"),
        "down_below": 90,
        "up_above": 160,
        "label": "Elbow angle",
    },
    "bicep_curl": {
        "joint": ("LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"),
        "down_below": 50,
        "up_above": 150,
        "label": "Elbow angle",
    },
}


@dataclass
class RepCounter:
    exercise: str
    count: int = 0
    state: str = "up"  # "up" or "down"
    last_angle: float | None = field(default=None)

    def update(self, landmarks) -> float | None:
        spec = EXERCISES[self.exercise]
        p1, p2, p3 = spec["joint"]
        try:
            a = get_landmark_point(landmarks, p1)
            b = get_landmark_point(landmarks, p2)
            c = get_landmark_point(landmarks, p3)
        except (KeyError, IndexError):
            return None
        angle = calculate_angle(a, b, c)
        if angle is None:
            return None
        self.last_angle = angle

        if self.state == "up" and angle < spec["down_below"]:
            self.state = "down"
        elif self.state == "down" and angle > spec["up_above"]:
            self.state = "up"
            self.count += 1
        return angle
