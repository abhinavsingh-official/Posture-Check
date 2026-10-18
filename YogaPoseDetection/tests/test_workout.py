from geometry import POSE_LMS
from workout import RepCounter, check_form


class _FakeLandmark:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = x, y, z


def _landmarks(**named_points):
    """A fake MediaPipe landmark list with only the named points set to
    something meaningful - everything else sits at the origin, which is
    fine since each test only exercises the joints it cares about."""
    points = [_FakeLandmark() for _ in range(33)]
    for name, (x, y, z) in named_points.items():
        points[POSE_LMS[name].value] = _FakeLandmark(x, y, z)
    return points


def test_rep_counter_counts_one_full_down_up_cycle():
    counter = RepCounter(exercise="squat")
    # Standing (angle ~180): hip/knee/ankle roughly in a line.
    standing = _landmarks(LEFT_HIP=(0, 0, 0), LEFT_KNEE=(0, 1, 0), LEFT_ANKLE=(0, 2, 0))
    counter.update(standing)
    assert counter.state == "up"
    assert counter.count == 0

    # Squatting down: knee bent to roughly a right angle.
    squatting = _landmarks(LEFT_HIP=(0, 0, 0), LEFT_KNEE=(0, 1, 0), LEFT_ANKLE=(1, 1, 0))
    counter.update(squatting)
    assert counter.state == "down"
    assert counter.count == 0

    # Back up: counts the rep.
    counter.update(standing)
    assert counter.state == "up"
    assert counter.count == 1


def test_rep_counter_does_not_count_without_a_real_down_phase():
    counter = RepCounter(exercise="squat")
    standing = _landmarks(LEFT_HIP=(0, 0, 0), LEFT_KNEE=(0, 1, 0), LEFT_ANKLE=(0, 2, 0))
    counter.update(standing)
    counter.update(standing)
    counter.update(standing)
    assert counter.count == 0


def test_rep_counter_missing_landmarks_returns_none_without_crashing():
    counter = RepCounter(exercise="squat")
    assert counter.update([]) is None


def test_check_form_flags_knee_valgus():
    # Knee drifted well inward of the ankle.
    landmarks = _landmarks(LEFT_KNEE=(0.40, 0.5, 0), LEFT_ANKLE=(0.50, 0.9, 0))
    warning = check_form("squat", landmarks)
    assert warning is not None
    assert "knee" in warning.lower()


def test_check_form_no_warning_when_knee_tracks_ankle():
    landmarks = _landmarks(LEFT_KNEE=(0.50, 0.5, 0), LEFT_ANKLE=(0.50, 0.9, 0))
    assert check_form("squat", landmarks) is None


def test_lateral_raise_counts_a_rep():
    counter = RepCounter(exercise="lateral_raise")
    # Arm down at the side: hip-shoulder-elbow nearly in line (~173 deg).
    arm_down = _landmarks(LEFT_HIP=(0, 0, 0), LEFT_SHOULDER=(0, -1, 0), LEFT_ELBOW=(0.1, -1.8, 0))
    counter.update(arm_down)
    assert counter.state == "up"

    # Arm raised to shoulder height: a right-angle bend at the shoulder (~90 deg).
    arm_raised = _landmarks(LEFT_HIP=(0, 0, 0), LEFT_SHOULDER=(0, -1, 0), LEFT_ELBOW=(1, -1, 0))
    counter.update(arm_raised)
    assert counter.state == "down"

    counter.update(arm_down)
    assert counter.state == "up"
    assert counter.count == 1


def test_check_form_flags_raise_overshoot():
    # Elbow above the shoulder (smaller y) - overshot the raise.
    landmarks = _landmarks(LEFT_SHOULDER=(0, -1, 0), LEFT_ELBOW=(0, -2, 0))
    warning = check_form("lateral_raise", landmarks)
    assert warning is not None
    assert "shoulder height" in warning.lower()


def test_check_form_no_warning_at_shoulder_height():
    landmarks = _landmarks(LEFT_SHOULDER=(0, -1, 0), LEFT_ELBOW=(1, -1, 0))
    assert check_form("lateral_raise", landmarks) is None
