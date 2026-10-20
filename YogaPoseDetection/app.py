"""Streamlit frontend - the real UI upgrade over the raw OpenCV window
analysis.py uses. Same underlying logic (geometry/feedback/renderer/
workout), just a different display loop: st.image() instead of
cv2.imshow(), since Streamlit can't show an OpenCV window directly.

Uses a checkbox, not Start/Stop buttons, to drive the camera loop - a
Streamlit button only reads as True for the one script run right after
it's clicked, so it can start a loop but can never interrupt one
already running. Unchecking a checkbox is a new widget interaction,
which Streamlit can act on even while a previous run's loop is still
executing - the standard, documented pattern for a live camera feed.

Every status message analysis.py's OpenCV window shows ("Stand
properly in frame", "Move back", "Please start your yoga") is mirrored
here as a real on-screen banner - the raw camera feed is never shown
with no explanation for why nothing else is happening."""

import time

import cv2
import mediapipe as mp
import pandas as pd
import pyautogui
import streamlit as st

import geometry
import feedback
import renderer
import storage
import workout
from air_canvas import AirCanvasState, draw_strokes, draw_toolbar
from analysis import load_angle_data, smooth_angles
from drowsiness import DrowsinessMonitor, average_ear
from gestures import GESTURE_STICKERS, GestureClassifier
from voice import VoiceFeedback
from virtual_mouse import HandLandmarks2D, VirtualMouseController
from workout import EXERCISES, RepCounter

st.set_page_config(page_title="Posture-Check", layout="wide")
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh

HOLD_TIME_REQUIRED = 5
AUTO_DETECT_DELAY = 5
START_MESSAGE_DURATION = 3
SMOOTHING_FRAMES = 7
CAMERA_WARMUP_ATTEMPTS = 15
PROCESS_WIDTH = 640  # frames are downscaled to this width before MediaPipe runs on them
STICKER_SUSTAIN_FRAMES = 20  # how many frames a sticker stays on screen once triggered

STYLE = """
<style>
.block-container { padding-top: 2rem; max-width: 1200px; }
h1 { font-weight: 700; letter-spacing: -0.02em; color: #e52521; }
div[data-testid="stMetric"] {
    background: #13213d; border: 1px solid #2a5ca8; border-radius: 10px;
    padding: 0.9rem 1rem;
}
div[data-testid="stMetricValue"] { font-size: 1.6rem; }
.pc-card {
    background: #13213d; border: 1px solid #2a5ca8; border-radius: 10px;
    padding: 1rem 1.2rem; margin-bottom: 0.8rem;
}
.pc-banner {
    background: #13213d; border: 1px solid #2a5ca8; border-radius: 12px;
    padding: 3rem 1.5rem; text-align: center; color: #c9d1d9; font-size: 1.05rem;
}
.pc-status-good { color: #3fb950; font-weight: 600; }
.pc-status-bad { color: #e52521; font-weight: 600; }
.pc-feature-card {
    background: #13213d; border: 1px solid #2a5ca8; border-radius: 12px;
    padding: 1.3rem 1.4rem; height: 100%;
}
.pc-feature-card h4 { margin-top: 0; color: #5b9bf0; }
.pc-stat { font-size: 2rem; font-weight: 700; color: #e52521; }
.pc-stat-label { color: #8b949e; font-size: 0.9rem; }
.pc-card ul, .pc-feature-card ul { margin: 0.4rem 0 0 0; padding-left: 1.2rem; color: #c9d1d9; }
.pc-card li, .pc-feature-card li { margin-bottom: 0.35rem; }
.pc-section-intro { color: #8b949e; font-size: 0.98rem; max-width: 760px; margin-bottom: 0.6rem; }
.pc-sidebar-note { color: #8b949e; font-size: 0.88rem; margin-bottom: 0.8rem; }
</style>
"""


def _open_camera() -> cv2.VideoCapture | None:
    cap = cv2.VideoCapture(0)
    for _ in range(CAMERA_WARMUP_ATTEMPTS):
        if not cap.isOpened():
            break
        ret, _ = cap.read()
        if ret:
            return cap
        time.sleep(0.1)
    cap.release()
    return None


def _downscale(frame):
    h, w = frame.shape[:2]
    if w <= PROCESS_WIDTH:
        return frame
    scale = PROCESS_WIDTH / w
    return cv2.resize(frame, (PROCESS_WIDTH, int(h * scale)))


def _camera_error():
    st.error(
        "Could not read from the camera. It opened but never returned a frame after "
        f"{CAMERA_WARMUP_ATTEMPTS} attempts - check that no other app is using it, and that "
        "this terminal app has camera permission (System Settings > Privacy & Security > Camera)."
    )


def _progress_bar(label: str, value: float, maximum: float) -> None:
    st.markdown(f"**{label}**")
    st.progress(min(value / maximum, 1.0))


def _empty_state(frame_slot, text: str) -> None:
    frame_slot.markdown(f'<div class="pc-banner">{text}</div>', unsafe_allow_html=True)


def home_tab():
    st.markdown(
        '<div class="pc-card" style="padding:2rem 2.2rem;">'
        "<h2 style='margin-top:0;'>A webcam, and six different ways to use it</h2>"
        "<p style='color:#c9d1d9; font-size:1.05rem; max-width:780px; line-height:1.6;'>"
        "This started as a yoga form-checker. It's grown into a small playground of what "
        "classical computer vision and MediaPipe can actually do with just a webcam - pose "
        "correction, rep counting, hand-gesture recognition, drawing in thin air, controlling "
        "your cursor with a finger, and watching for drowsy eyes. Pick a tab above."
        "</p>"
        "<ul>"
        "<li><b>Yoga</b> and <b>Workout</b> - the original two, full-body pose tracking</li>"
        "<li><b>Gesture Stickers</b>, <b>Air Canvas</b>, <b>Virtual Mouse</b> - hand tracking, "
        "three different ways</li>"
        "<li><b>Drowsiness Monitor</b> - face tracking, watching your eyes instead of your hands</li>"
        "</ul>"
        "</div>",
        unsafe_allow_html=True,
    )

    sessions = storage.recent_sessions()
    st.markdown("#### Where things stand right now")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f'<div class="pc-feature-card"><div class="pc-stat">{len(load_angle_data())}</div>'
            '<div class="pc-stat-label">Yoga poses with reference data</div></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f'<div class="pc-feature-card"><div class="pc-stat">{len(EXERCISES)}</div>'
            '<div class="pc-stat-label">Tracked workout exercises</div></div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            '<div class="pc-feature-card"><div class="pc-stat">6</div>'
            '<div class="pc-stat-label">Camera modes total</div></div>',
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f'<div class="pc-feature-card"><div class="pc-stat">{len(sessions)}</div>'
            '<div class="pc-stat-label">Sessions logged on this machine</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("#### What each tab does")
    row1 = st.columns(3)
    row2 = st.columns(3)
    with row1[0]:
        st.markdown(
            '<div class="pc-feature-card"><h4>Yoga</h4>'
            "<ul>"
            "<li>Auto-detect a pose, or pick one yourself</li>"
            "<li>Skeleton turns red/green per joint</li>"
            "<li>Tells you exactly what to fix, e.g. \"Left Knee: increase by 12°\"</li>"
            "<li>Counts a hold once you've held good form a few seconds</li>"
            "</ul></div>",
            unsafe_allow_html=True,
        )
    with row1[1]:
        st.markdown(
            '<div class="pc-feature-card"><h4>Workout</h4>'
            "<ul>"
            "<li>Six exercises: squats, push-ups, curls, lunges, presses, raises</li>"
            "<li>Counts reps without double-counting on a jittery frame</li>"
            "<li>Flags the common mistake for each exercise - caving knees, flared elbows, "
            "and so on</li>"
            "</ul></div>",
            unsafe_allow_html=True,
        )
    with row1[2]:
        st.markdown(
            '<div class="pc-feature-card"><h4>Gesture Stickers</h4>'
            "<ul>"
            "<li>Recognizes four hand signs: open palm, closed fist, pointing, OK</li>"
            "<li>Each one pops a different sticker onto the feed</li>"
            "<li>Runs on a pretrained hand-sign classifier, not something trained here</li>"
            "</ul></div>",
            unsafe_allow_html=True,
        )
    with row2[0]:
        st.markdown(
            '<div class="pc-feature-card"><h4>Air Canvas</h4>'
            "<ul>"
            "<li>Draw by moving your index fingertip</li>"
            "<li>Pinch thumb and finger together to lift the pen</li>"
            "<li>Pick a color or clear the canvas from the on-screen toolbar</li>"
            "</ul></div>",
            unsafe_allow_html=True,
        )
    with row2[1]:
        st.markdown(
            '<div class="pc-feature-card"><h4>Virtual Mouse</h4>'
            "<ul>"
            "<li>Point to move your real cursor, curl your index finger to click</li>"
            "<li>Cursor control is a separate opt-in - camera preview runs first</li>"
            "<li>Auto-disables itself a few seconds after your hand leaves frame</li>"
            "</ul></div>",
            unsafe_allow_html=True,
        )
    with row2[2]:
        st.markdown(
            '<div class="pc-feature-card"><h4>Drowsiness Monitor</h4>'
            "<ul>"
            "<li>Tracks eye-aspect-ratio - how open your eyes are, not just open/closed</li>"
            "<li>Alerts after eyes stay closed too long, counts blinks separately</li>"
            "<li>The same technique real driver-drowsiness systems use</li>"
            "</ul></div>",
            unsafe_allow_html=True,
        )

    st.markdown("#### How it actually works")
    h1, h2 = st.columns(2)
    with h1:
        st.markdown(
            '<div class="pc-feature-card"><h4>Under the hood</h4>'
            '<p class="pc-section-intro" style="margin-bottom:0.6rem;">Three different MediaPipe '
            "trackers, each feeding a different kind of logic:</p>"
            "<ul>"
            "<li><b>Pose</b> (33 body landmarks) - joint angles via vector math, for Yoga and "
            "Workout</li>"
            "<li><b>Hands</b> (21 landmarks per hand) - a pretrained classifier for Gesture "
            "Stickers, raw fingertip tracking for Air Canvas and Virtual Mouse</li>"
            "<li><b>Face Mesh</b> (468 landmarks) - eye-aspect-ratio for the Drowsiness "
            "Monitor</li>"
            "</ul></div>",
            unsafe_allow_html=True,
        )
    with h2:
        st.markdown(
            '<div class="pc-feature-card"><h4>Known limits</h4>'
            "<ul>"
            "<li>Yoga auto-detect tested at 88.9% accuracy overall - a few poses are weaker, "
            "see the README for the full breakdown</li>"
            "<li>Form checks and the drowsiness threshold are reasonable defaults, not "
            "medically validated advice</li>"
            "<li>Every hand/face mode assumes one person, mostly facing the camera</li>"
            "</ul></div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        '<p style="color:#8b949e; margin-top:1.6rem;">Try it - pick any tab above to start.</p>',
        unsafe_allow_html=True,
    )


def yoga_tab():
    angle_data = load_angle_data()
    pose_list = list(angle_data.keys())

    col1, col2 = st.columns([3, 1], gap="large")
    with col2:
        st.markdown(
            '<p class="pc-sidebar-note">Auto-detect figures out your pose after a short hold. '
            "Manual skips the guessing if you want to practice one pose.</p>",
            unsafe_allow_html=True,
        )
        st.markdown('<div class="pc-card">', unsafe_allow_html=True)
        detect_mode = st.radio("Mode", ["Auto-detect", "Manual"])
        selected_pose = None if detect_mode == "Auto-detect" else st.selectbox("Pose", pose_list)
        live_angles_on = st.checkbox("Show live angles")
        voice_on = st.checkbox("Voice feedback", key="yoga_voice")
        st.markdown("</div>", unsafe_allow_html=True)
        running = st.checkbox("Start camera", key="yoga_running")
        pose_metric = st.empty()
        hold_slot = st.empty()
        hints_box = st.empty()

    with col1:
        frame_slot = st.empty()
        if not running:
            _empty_state(frame_slot, "Check “Start camera” on the right to begin")

    if not running:
        return

    cap = _open_camera()
    if cap is None:
        _camera_error()
        return

    voice = VoiceFeedback(enabled=voice_on)
    smoothing_buffer: list[dict[str, float]] = []
    detection_start_time = None
    hold_start = None
    session_start = time.time()
    loop_start = time.time()

    try:
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose_model:
            while st.session_state.get("yoga_running"):
                ret, frame = cap.read()
                if not ret:
                    st.warning("Lost the camera feed mid-session.")
                    break
                frame = cv2.flip(frame, 1)
                process_frame = _downscale(frame)
                results = pose_model.process(cv2.cvtColor(process_frame, cv2.COLOR_BGR2RGB))

                if not results.pose_landmarks:
                    if time.time() - loop_start <= START_MESSAGE_DURATION:
                        _empty_state(frame_slot, "Please start your yoga")
                    else:
                        _empty_state(frame_slot, "Stand properly in frame")
                    continue

                landmarks = results.pose_landmarks
                lm = landmarks.landmark
                if not geometry.full_body_visible(lm):
                    _empty_state(frame_slot, "Move back – full body not visible")
                    continue

                if time.time() - loop_start <= START_MESSAGE_DURATION:
                    _empty_state(frame_slot, "Please start your yoga")
                    continue

                user_angles = geometry.get_all_angles(lm)
                smoothing_buffer.append(user_angles)
                if len(smoothing_buffer) > SMOOTHING_FRAMES:
                    smoothing_buffer.pop(0)
                smoothed = smooth_angles(smoothing_buffer, geometry.JOINTS)

                active_pose = selected_pose
                if detect_mode == "Auto-detect":
                    if detection_start_time is None:
                        detection_start_time = time.time()
                    elapsed = time.time() - detection_start_time
                    if elapsed < AUTO_DETECT_DELAY:
                        _empty_state(frame_slot, f"Hold still... detecting pose in {int(AUTO_DETECT_DELAY - elapsed)}s")
                        continue
                    active_pose, _ = feedback.auto_detect_pose(smoothed, angle_data)

                score, hints = feedback.compare_pose(smoothed, active_pose, angle_data)
                correct = feedback.is_pose_correct(score)
                hold_start = hold_start if correct and hold_start else (time.time() if correct else None)
                hold_time = (time.time() - hold_start) if hold_start else 0

                if voice_on and hints:
                    voice.say(hints[0])

                bad_joints = renderer.bad_joints_from_hints(hints)
                renderer.draw_skeleton(process_frame, landmarks, bad_joints)
                frame_slot.image(process_frame, channels="BGR")
                pose_metric.metric("Pose", active_pose or "None")
                with hold_slot.container():
                    _progress_bar("Hold time", hold_time, HOLD_TIME_REQUIRED)
                if hints:
                    hints_box.markdown(
                        '<div class="pc-card pc-status-bad">' + "<br>".join(hints) + "</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    hints_box.markdown(
                        '<div class="pc-card pc-status-good">Looking good!</div>', unsafe_allow_html=True
                    )

                if hold_time >= HOLD_TIME_REQUIRED:
                    storage.log_session(storage.SessionRecord(
                        mode="yoga", name=active_pose, duration_seconds=time.time() - session_start,
                    ))
                    st.success(f"Held {active_pose} correctly for {HOLD_TIME_REQUIRED}s!")
                    break
    except Exception as e:  # noqa: BLE001 - surface it in the UI, never a silent blank screen
        st.error(f"Session stopped on an error: {e}")
    finally:
        cap.release()


def workout_tab():
    col1, col2 = st.columns([3, 1], gap="large")
    with col2:
        st.markdown(
            '<p class="pc-sidebar-note">Pick the exercise you\'re doing - reps and form '
            "warnings only make sense if it matches what's on screen.</p>",
            unsafe_allow_html=True,
        )
        st.markdown('<div class="pc-card">', unsafe_allow_html=True)
        exercise = st.selectbox("Exercise", list(EXERCISES.keys()), format_func=lambda e: e.replace("_", " ").title())
        voice_on = st.checkbox("Voice feedback", key="workout_voice")
        st.markdown("</div>", unsafe_allow_html=True)
        running = st.checkbox("Start camera", key="workout_running")
        rep_metric = st.empty()
        warning_box = st.empty()

    with col1:
        frame_slot = st.empty()
        if not running:
            _empty_state(frame_slot, "Check “Start camera” on the right to begin")

    if not running:
        return

    cap = _open_camera()
    if cap is None:
        _camera_error()
        return

    voice = VoiceFeedback(enabled=voice_on)
    counter = RepCounter(exercise=exercise)
    session_start = time.time()

    try:
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose_model:
            while st.session_state.get("workout_running"):
                ret, frame = cap.read()
                if not ret:
                    st.warning("Lost the camera feed mid-session.")
                    break
                frame = cv2.flip(frame, 1)
                process_frame = _downscale(frame)
                results = pose_model.process(cv2.cvtColor(process_frame, cv2.COLOR_BGR2RGB))

                if not results.pose_landmarks:
                    _empty_state(frame_slot, "Stand properly in frame")
                    continue

                if not geometry.full_body_visible(results.pose_landmarks.landmark):
                    _empty_state(frame_slot, "Move back – full body not visible")
                    continue

                counter.update(results.pose_landmarks.landmark)
                form_warning = workout.check_form(exercise, results.pose_landmarks.landmark)
                if form_warning:
                    counter.form_warnings += 1
                    if voice_on:
                        voice.say(form_warning)
                renderer.draw_skeleton(process_frame, results.pose_landmarks, set())
                if form_warning:
                    warning_box.markdown(
                        f'<div class="pc-card pc-status-bad">{form_warning}</div>', unsafe_allow_html=True
                    )
                else:
                    warning_box.markdown(
                        '<div class="pc-card pc-status-good">Good form</div>', unsafe_allow_html=True
                    )

                frame_slot.image(process_frame, channels="BGR")
                rep_metric.metric("Reps", counter.count)
    except Exception as e:  # noqa: BLE001
        st.error(f"Session stopped on an error: {e}")
    finally:
        cap.release()
        storage.log_session(storage.SessionRecord(
            mode="workout", name=exercise, duration_seconds=time.time() - session_start,
            reps=counter.count, form_warnings=counter.form_warnings,
        ))
        st.success(f"Session saved: {counter.count} reps, {counter.form_warnings} form warning(s).")


@st.cache_resource
def _load_gesture_classifier() -> GestureClassifier:
    return GestureClassifier()


def gesture_tab():
    st.markdown(
        '<p class="pc-sidebar-note">Show one hand to the camera - an open palm, a closed fist, '
        "a pointing finger, or an OK sign. Each one pops up a different sticker.</p>",
        unsafe_allow_html=True,
    )
    running = st.checkbox("Start camera", key="gesture_running")
    gesture_label = st.empty()
    frame_slot = st.empty()
    if not running:
        _empty_state(frame_slot, "Check “Start camera” above to begin")
        return

    cap = _open_camera()
    if cap is None:
        _camera_error()
        return

    classifier = _load_gesture_classifier()
    sticker_name = None
    sticker_frames_left = 0

    try:
        with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.5) as hands:
            while st.session_state.get("gesture_running"):
                ret, frame = cap.read()
                if not ret:
                    st.warning("Lost the camera feed mid-session.")
                    break
                frame = cv2.flip(frame, 1)
                process_frame = _downscale(frame)
                results = hands.process(cv2.cvtColor(process_frame, cv2.COLOR_BGR2RGB))
                height, width = process_frame.shape[:2]

                if not results.multi_hand_landmarks:
                    gesture_label.markdown(
                        '<div class="pc-card">Show a hand to the camera</div>', unsafe_allow_html=True
                    )
                    frame_slot.image(process_frame, channels="BGR")
                    continue

                hand_landmarks = results.multi_hand_landmarks[0]
                label = classifier.classify(hand_landmarks, width, height)
                sticker = GESTURE_STICKERS.get(label)
                if sticker:
                    sticker_name = sticker
                    sticker_frames_left = STICKER_SUSTAIN_FRAMES

                mp.solutions.drawing_utils.draw_landmarks(
                    process_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

                if sticker_frames_left > 0 and sticker_name:
                    wrist = hand_landmarks.landmark[0]
                    position = (int(wrist.x * width), int(wrist.y * height))
                    renderer.draw_sticker(process_frame, position, sticker_name)
                    sticker_frames_left -= 1

                frame_slot.image(process_frame, channels="BGR")
                gesture_label.markdown(
                    f'<div class="pc-card"><b>Detected:</b> {label}</div>', unsafe_allow_html=True
                )
    except Exception as e:  # noqa: BLE001
        st.error(f"Session stopped on an error: {e}")
    finally:
        cap.release()


def air_canvas_tab():
    st.markdown(
        '<p class="pc-sidebar-note">Point with your index finger to draw. Pinch your thumb and '
        "index finger together to lift the pen. The top bar picks a color or clears the canvas.</p>",
        unsafe_allow_html=True,
    )
    running = st.checkbox("Start camera", key="canvas_running")
    status_slot = st.empty()
    frame_slot = st.empty()
    if not running:
        _empty_state(frame_slot, "Check “Start camera” above to begin")
        return

    cap = _open_camera()
    if cap is None:
        _camera_error()
        return

    canvas_state = AirCanvasState()

    try:
        with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.5) as hands:
            while st.session_state.get("canvas_running"):
                ret, frame = cap.read()
                if not ret:
                    st.warning("Lost the camera feed mid-session.")
                    break
                frame = cv2.flip(frame, 1)
                process_frame = _downscale(frame)
                results = hands.process(cv2.cvtColor(process_frame, cv2.COLOR_BGR2RGB))
                height, width = process_frame.shape[:2]

                status = "no hand"
                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    index_tip = hand_landmarks.landmark[8]
                    thumb_tip = hand_landmarks.landmark[4]
                    index_point = (int(index_tip.x * width), int(index_tip.y * height))
                    thumb_point = (int(thumb_tip.x * width), int(thumb_tip.y * height))
                    status = canvas_state.handle_point(index_point, thumb_point, width)

                draw_strokes(process_frame, canvas_state.strokes)
                draw_toolbar(process_frame, canvas_state.current_color)
                frame_slot.image(process_frame, channels="BGR")
                status_slot.markdown(
                    f'<div class="pc-card">Status: <b>{status}</b> - color: <b>{canvas_state.current_color}</b></div>',
                    unsafe_allow_html=True,
                )
    except Exception as e:  # noqa: BLE001
        st.error(f"Session stopped on an error: {e}")
    finally:
        cap.release()


def virtual_mouse_tab():
    st.markdown(
        '<p class="pc-sidebar-note">This mode moves your real system cursor. Point with your '
        "index finger to move it; curl the index finger (while keeping your thumb away from "
        "your palm) to click. Camera preview runs first - cursor control is a separate, "
        "explicit opt-in below it.</p>",
        unsafe_allow_html=True,
    )
    running = st.checkbox("Start camera (preview only)", key="mouse_running")
    cursor_enabled = False
    if running:
        cursor_enabled = st.checkbox(
            "⚠ Enable cursor control (moves your real mouse)", key="mouse_cursor_enabled"
        )
    status_slot = st.empty()
    frame_slot = st.empty()
    if not running:
        _empty_state(frame_slot, "Check “Start camera” above to begin")
        return

    cap = _open_camera()
    if cap is None:
        _camera_error()
        return

    controller = VirtualMouseController()
    screen_width, screen_height = pyautogui.size()

    try:
        with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.5) as hands:
            while st.session_state.get("mouse_running"):
                ret, frame = cap.read()
                if not ret:
                    st.warning("Lost the camera feed mid-session.")
                    break
                frame = cv2.flip(frame, 1)
                process_frame = _downscale(frame)
                results = hands.process(cv2.cvtColor(process_frame, cv2.COLOR_BGR2RGB))

                landmarks = None
                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    landmarks = HandLandmarks2D.from_mediapipe(hand_landmarks)
                    mp.solutions.drawing_utils.draw_landmarks(
                        process_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )

                if cursor_enabled:
                    status = controller.update(landmarks, screen_width, screen_height)
                else:
                    status = "cursor control off - preview only"

                frame_slot.image(process_frame, channels="BGR")
                status_slot.markdown(f'<div class="pc-card">Status: <b>{status}</b></div>', unsafe_allow_html=True)
    except Exception as e:  # noqa: BLE001
        st.error(f"Session stopped on an error: {e}")
    finally:
        cap.release()


def drowsiness_tab():
    st.markdown(
        '<p class="pc-sidebar-note">Watches your eye-aspect-ratio (how open your eyes are, '
        "relative to their width) and alerts if your eyes stay closed too long - the same "
        "technique used in driver-drowsiness systems.</p>",
        unsafe_allow_html=True,
    )
    running = st.checkbox("Start camera", key="drowsy_running")
    status_slot = st.empty()
    frame_slot = st.empty()
    if not running:
        _empty_state(frame_slot, "Check “Start camera” above to begin")
        return

    cap = _open_camera()
    if cap is None:
        _camera_error()
        return

    monitor = DrowsinessMonitor()
    voice = VoiceFeedback(enabled=True)

    try:
        with mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
            while st.session_state.get("drowsy_running"):
                ret, frame = cap.read()
                if not ret:
                    st.warning("Lost the camera feed mid-session.")
                    break
                frame = cv2.flip(frame, 1)
                process_frame = _downscale(frame)
                results = face_mesh.process(cv2.cvtColor(process_frame, cv2.COLOR_BGR2RGB))
                height, width = process_frame.shape[:2]

                if not results.multi_face_landmarks:
                    _empty_state(frame_slot, "Face not visible - move into frame")
                    continue

                face_landmarks = results.multi_face_landmarks[0]
                ear = average_ear(face_landmarks, width, height)
                alert = monitor.update(ear)

                if alert:
                    cv2.putText(process_frame, "DROWSINESS ALERT", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                    voice.say("Wake up")

                frame_slot.image(process_frame, channels="BGR")
                status_slot.markdown(
                    f'<div class="pc-card {"pc-status-bad" if alert else "pc-status-good"}">'
                    f"EAR: {ear:.2f} - blinks: {monitor.total_blinks}"
                    f'{" - ALERT: eyes closed too long" if alert else ""}</div>',
                    unsafe_allow_html=True,
                )
    except Exception as e:  # noqa: BLE001
        st.error(f"Session stopped on an error: {e}")
    finally:
        cap.release()


def history_tab():
    sessions = storage.recent_sessions()
    if not sessions:
        st.info("No sessions recorded yet - complete a Yoga or Workout session first.")
        return

    df = pd.DataFrame(sessions)
    workout_df = df[df["mode"] == "workout"]
    yoga_df = df[df["mode"] == "yoga"]

    st.markdown("#### Personal bests")
    b1, b2, b3 = st.columns(3)
    with b1:
        best_reps = int(workout_df["reps"].max()) if not workout_df.empty else 0
        st.markdown(
            f'<div class="pc-feature-card"><div class="pc-stat">{best_reps}</div>'
            '<div class="pc-stat-label">Most reps in a single session</div></div>',
            unsafe_allow_html=True,
        )
    with b2:
        longest_hold = yoga_df["duration_seconds"].max() if not yoga_df.empty else 0
        st.markdown(
            f'<div class="pc-feature-card"><div class="pc-stat">{longest_hold:.0f}s</div>'
            '<div class="pc-stat-label">Longest yoga session</div></div>',
            unsafe_allow_html=True,
        )
    with b3:
        st.markdown(
            f'<div class="pc-feature-card"><div class="pc-stat">{len(df)}</div>'
            '<div class="pc-stat-label">Total sessions logged</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("#### Full history")
    st.dataframe(df, use_container_width=True)
    if not workout_df.empty:
        for exercise_name in workout_df["name"].unique():
            st.subheader(f"{exercise_name.replace('_', ' ').title()} - reps over time")
            trend = workout_df[workout_df["name"] == exercise_name][["started_at", "reps"]]
            st.line_chart(trend.set_index("started_at"))

    st.download_button(
        "Export session history (CSV)",
        df.to_csv(index=False).encode("utf-8"),
        file_name="posture_check_sessions.csv",
        mime="text/csv",
    )


def main():
    st.markdown(STYLE, unsafe_allow_html=True)
    st.title("Posture-Check")
    st.caption("Six real-time computer vision modes, one webcam - pose, hands, and face tracking.")
    tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
        ["Home", "Yoga", "Workout", "Gesture Stickers", "Air Canvas", "Virtual Mouse",
         "Drowsiness Monitor", "Session History"]
    )
    with tab0:
        home_tab()
    with tab1:
        yoga_tab()
    with tab2:
        workout_tab()
    with tab3:
        gesture_tab()
    with tab4:
        air_canvas_tab()
    with tab5:
        virtual_mouse_tab()
    with tab6:
        drowsiness_tab()
    with tab7:
        history_tab()


if __name__ == "__main__":
    main()
