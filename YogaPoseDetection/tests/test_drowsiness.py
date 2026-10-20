from drowsiness import DrowsinessMonitor, eye_aspect_ratio


def test_open_eye_has_high_ear():
    # Corners far apart, lids far apart too - a "normal open eye" shape.
    eye = [(0, 0), (10, -5), (20, -5), (30, 0), (20, 5), (10, 5)]
    assert eye_aspect_ratio(eye) > 0.3


def test_closed_eye_has_low_ear():
    # Corners the same distance apart, but lids collapsed onto each other.
    eye = [(0, 0), (10, 0), (20, 0), (30, 0), (20, 0), (10, 0)]
    assert eye_aspect_ratio(eye) == 0.0


def test_monitor_does_not_alert_on_a_normal_blink():
    monitor = DrowsinessMonitor(threshold=0.2, frames_required=15)
    for _ in range(3):  # a quick blink - well under the alert threshold
        assert monitor.update(0.1) is False
    assert monitor.update(0.3) is False
    assert monitor.total_blinks == 1


def test_monitor_alerts_on_sustained_eye_closure():
    monitor = DrowsinessMonitor(threshold=0.2, frames_required=15)
    alerted = False
    for _ in range(20):
        alerted = monitor.update(0.1) or alerted
    assert alerted is True


def test_monitor_resets_after_eyes_reopen():
    monitor = DrowsinessMonitor(threshold=0.2, frames_required=5)
    for _ in range(5):
        monitor.update(0.1)
    assert monitor.update(0.3) is False
    assert monitor.consecutive_closed_frames == 0
