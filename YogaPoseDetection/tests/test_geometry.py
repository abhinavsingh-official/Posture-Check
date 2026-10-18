import math

from geometry import calculate_angle, full_body_visible


def test_calculate_angle_right_angle():
    # A right angle at b=(0,0,0): a is straight up, c is straight right.
    angle = calculate_angle([0, 1, 0], [0, 0, 0], [1, 0, 0])
    assert math.isclose(angle, 90.0, abs_tol=0.01)


def test_calculate_angle_straight_line():
    angle = calculate_angle([-1, 0, 0], [0, 0, 0], [1, 0, 0])
    assert math.isclose(angle, 180.0, abs_tol=0.01)


def test_calculate_angle_zero_length_vector_returns_none():
    # a == b, so v1 has zero length - undefined angle, not a crash.
    assert calculate_angle([0, 0, 0], [0, 0, 0], [1, 0, 0]) is None


class _FakeLandmark:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def test_full_body_visible_true_when_all_in_frame():
    landmarks = [_FakeLandmark(0.5, 0.5) for _ in range(10)]
    assert full_body_visible(landmarks) is True


def test_full_body_visible_false_when_mostly_out_of_frame():
    landmarks = [_FakeLandmark(1.5, 1.5) for _ in range(10)]
    assert full_body_visible(landmarks) is False


def test_full_body_visible_respects_threshold():
    # 8/10 in frame = 0.8, right at the default threshold.
    landmarks = [_FakeLandmark(0.5, 0.5) for _ in range(8)] + [_FakeLandmark(-1, -1) for _ in range(2)]
    assert full_body_visible(landmarks, threshold=0.80) is True
    assert full_body_visible(landmarks, threshold=0.81) is False
