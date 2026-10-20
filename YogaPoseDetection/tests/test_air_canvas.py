from air_canvas import AirCanvasState


def test_drawing_appends_points_to_current_stroke():
    state = AirCanvasState()
    state.handle_point((100, 200), (300, 400), frame_width=640)  # thumb far away - pen down
    state.handle_point((105, 205), (300, 400), frame_width=640)
    assert state.strokes["blue"][-1] == [(100, 200), (105, 205)]


def test_pinch_lifts_pen_and_starts_new_stroke():
    state = AirCanvasState()
    state.handle_point((100, 200), (300, 400), frame_width=640)
    status = state.handle_point((100, 200), (105, 205), frame_width=640)  # thumb now close - pinch
    assert status == "pen up"
    assert state.strokes["blue"] == [[(100, 200)], []]

    state.handle_point((150, 250), (400, 450), frame_width=640)  # pen down again, far thumb
    assert state.strokes["blue"][-1] == [(150, 250)]


def test_toolbar_zone_clears_canvas():
    state = AirCanvasState()
    state.handle_point((100, 200), (300, 400), frame_width=640)
    assert state.strokes["blue"][-1] == [(100, 200)]

    state.handle_point((10, 10), (300, 400), frame_width=640)  # top-left = clear zone
    assert state.strokes["blue"] == [[]]


def test_toolbar_zone_switches_color():
    state = AirCanvasState()
    # 640 / 5 zones = 128px wide each: clear=0-128, blue=128-256, green=256-384, red=384-512, yellow=512-640
    state.handle_point((300, 10), (300, 400), frame_width=640)
    assert state.current_color == "green"
