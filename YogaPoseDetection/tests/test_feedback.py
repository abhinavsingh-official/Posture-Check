from feedback import auto_detect_pose, compare_pose, is_pose_correct

ANGLE_DATA = {
    "tree-pose": {
        "left_knee": {"angle_min": 150, "angle_max": 180, "tip": "straighten your standing leg"},
        "right_hip": {"angle_min": 40, "angle_max": 70, "tip": "lift your bent knee higher"},
    },
    "chair-pose": {
        "left_knee": {"angle_min": 80, "angle_max": 110, "tip": "bend your knees more"},
        "right_hip": {"angle_min": 80, "angle_max": 110, "tip": "sit back further"},
    },
}


def test_compare_pose_in_range_scores_zero_with_no_hints():
    score, hints = compare_pose({"left_knee": 165, "right_hip": 55}, "tree-pose", ANGLE_DATA)
    assert score == 0
    assert hints == []


def test_compare_pose_out_of_range_produces_hint_and_nonzero_score():
    score, hints = compare_pose({"left_knee": 120, "right_hip": 55}, "tree-pose", ANGLE_DATA)
    assert score > 0
    assert len(hints) == 1
    assert "increase" in hints[0].lower()


def test_compare_pose_above_range_says_decrease():
    score, hints = compare_pose({"left_knee": 175, "right_hip": 90}, "tree-pose", ANGLE_DATA)
    assert "decrease" in hints[0].lower()


def test_compare_pose_ignores_joints_not_in_reference():
    score, hints = compare_pose({"left_knee": 165, "neck": 999}, "tree-pose", ANGLE_DATA)
    assert score == 0
    assert hints == []


def test_auto_detect_pose_picks_the_closer_match():
    # Squarely in chair-pose's range, nowhere near tree-pose's.
    predicted, score = auto_detect_pose({"left_knee": 95, "right_hip": 95}, ANGLE_DATA)
    assert predicted == "chair-pose"
    assert score == 0


def test_is_pose_correct_threshold():
    assert is_pose_correct(5.0) is True
    assert is_pose_correct(15.0) is False
