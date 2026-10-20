from virtual_mouse import HandLandmarks2D, VirtualMouseController, is_index_extended, is_pinch_click


def _landmarks(overrides: dict) -> HandLandmarks2D:
    points = [(0.5, 0.5)] * 21
    points = list(points)
    for index, point in overrides.items():
        points[index] = point
    return HandLandmarks2D(points)


def test_extended_index_finger_detected():
    # Straight line mcp(5) -> pip(6) -> tip(8): fully extended, angle near 180.
    landmarks = _landmarks({5: (0.5, 0.5), 6: (0.5, 0.4), 8: (0.5, 0.3)})
    assert is_index_extended(landmarks)


def test_curled_index_finger_not_extended():
    # A sharp bend at the pip joint - tip folds back toward the mcp.
    landmarks = _landmarks({5: (0.5, 0.5), 6: (0.5, 0.4), 8: (0.55, 0.45)})
    assert not is_index_extended(landmarks)


def test_pinch_click_requires_curled_finger_and_separated_thumb():
    landmarks = _landmarks({4: (0.3, 0.3), 5: (0.5, 0.5), 6: (0.5, 0.4), 8: (0.55, 0.45)})
    assert is_pinch_click(landmarks)


def test_no_click_when_thumb_too_close_to_index_base():
    landmarks = _landmarks({4: (0.5, 0.5), 5: (0.51, 0.51), 6: (0.5, 0.4), 8: (0.55, 0.45)})
    assert not is_pinch_click(landmarks)


def test_controller_reports_no_hand_then_idle_after_timeout():
    controller = VirtualMouseController()
    controller.last_hand_seen = 0  # force "a long time ago"
    status = controller.update(None, 1920, 1080)
    assert status == "idle (no hand)"


def test_controller_reports_no_hand_before_timeout():
    import time
    controller = VirtualMouseController()
    controller.last_hand_seen = time.time()
    assert controller.update(None, 1920, 1080) == "no hand"
