"""Rep counting for dynamic exercises - a different problem from yoga's
"hold a static pose": here we track a joint angle crossing down/up
thresholds and count one rep per full cycle, the standard approach used
by most AI-trainer rep counters.

Counting reps alone doesn't catch bad form, so each exercise also has a
form check - a second angle or position check that's independent of the
rep-counting state machine, so a badly-formed rep still gets counted
(you did complete a rep) but comes with a warning."""

from dataclasses import dataclass, field

from geometry import calculate_angle, get_landmark_point

# Each exercise tracks one representative angle for rep counting.
# "down_below"/"up_above" are crossed in sequence to count a rep - a
# simple state machine (up -> down -> up), not a raw single-threshold
# check, so a noisy frame near the threshold doesn't double-count.
#
# "form_check" is a second, independent check for a common mistake in
# that exercise - these thresholds are reasonable approximations for a
# demo, not biomechanically validated advice.
EXERCISES: dict[str, dict] = {
    "squat": {
        "joint": ("LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"),
        "down_below": 100,
        "up_above": 160,
        "label": "Knee angle",
        "form_check": "knee_valgus",
    },
    "pushup": {
        "joint": ("LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"),
        "down_below": 90,
        "up_above": 160,
        "label": "Elbow angle",
        "form_check": "elbow_flare",
    },
    "bicep_curl": {
        "joint": ("LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"),
        "down_below": 50,
        "up_above": 150,
        "label": "Elbow angle",
        "form_check": "upper_arm_drift",
    },
}

# How far the knee can drift inward of the ankle (as a fraction of
# normalized frame width) before it reads as valgus (knees caving in).
KNEE_VALGUS_THRESHOLD = 0.04
# Shoulder-hip-elbow angle beyond which the elbow counts as flared out
# from the torso during a push-up.
ELBOW_FLARE_THRESHOLD = 75
# Same angle, for bicep curls - the upper arm swinging forward off the
# torso instead of staying still.
UPPER_ARM_DRIFT_THRESHOLD = 40


def _check_knee_valgus(landmarks) -> str | None:
    try:
        knee = get_landmark_point(landmarks, "LEFT_KNEE")
        ankle = get_landmark_point(landmarks, "LEFT_ANKLE")
    except (KeyError, IndexError):
        return None
    if ankle[0] - knee[0] > KNEE_VALGUS_THRESHOLD:
        return "Knee caving inward - track it over your toes"
    return None


def _check_torso_angle(landmarks, threshold: float, warning: str) -> str | None:
    try:
        hip = get_landmark_point(landmarks, "LEFT_HIP")
        shoulder = get_landmark_point(landmarks, "LEFT_SHOULDER")
        elbow = get_landmark_point(landmarks, "LEFT_ELBOW")
    except (KeyError, IndexError):
        return None
    angle = calculate_angle(hip, shoulder, elbow)
    if angle is not None and angle > threshold:
        return warning
    return None


def check_form(exercise: str, landmarks) -> str | None:
    form_check = EXERCISES[exercise]["form_check"]
    if form_check == "knee_valgus":
        return _check_knee_valgus(landmarks)
    if form_check == "elbow_flare":
        return _check_torso_angle(landmarks, ELBOW_FLARE_THRESHOLD, "Elbows flaring out - keep them closer to your body")
    if form_check == "upper_arm_drift":
        return _check_torso_angle(landmarks, UPPER_ARM_DRIFT_THRESHOLD, "Upper arm drifting forward - keep it still")
    return None


@dataclass
class RepCounter:
    exercise: str
    count: int = 0
    state: str = "up"  # "up" or "down"
    last_angle: float | None = field(default=None)
    form_warnings: int = 0

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
