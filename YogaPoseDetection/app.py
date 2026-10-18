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
import streamlit as st

import geometry
import feedback
import renderer
import storage
import workout
from analysis import load_angle_data, smooth_angles
from voice import VoiceFeedback
from workout import EXERCISES, RepCounter

st.set_page_config(page_title="Posture-Check", layout="wide")
mp_pose = mp.solutions.pose

HOLD_TIME_REQUIRED = 5
AUTO_DETECT_DELAY = 5
START_MESSAGE_DURATION = 3
SMOOTHING_FRAMES = 7
CAMERA_WARMUP_ATTEMPTS = 15
PROCESS_WIDTH = 640  # frames are downscaled to this width before MediaPipe runs on them

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
        "<h2 style='margin-top:0;'>A webcam that checks your form</h2>"
        "<p style='color:#c9d1d9; font-size:1.05rem; max-width:760px; line-height:1.6;'>"
        "Point your webcam at yourself and get told, in real time, whether you're actually "
        "doing the move right - not just whether you moved."
        "</p>"
        "<ul>"
        "<li><b>Yoga</b> - matches your pose against a reference, joint by joint</li>"
        "<li><b>Workout</b> - counts reps and catches the usual form mistakes</li>"
        "</ul>"
        "</div>",
        unsafe_allow_html=True,
    )

    sessions = storage.recent_sessions()
    st.markdown("#### Where things stand right now")
    col1, col2, col3 = st.columns(3)
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
            f'<div class="pc-feature-card"><div class="pc-stat">{len(sessions)}</div>'
            '<div class="pc-stat-label">Sessions logged on this machine</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("#### What each tab does")
    c1, c2, c3 = st.columns(3)
    with c1:
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
    with c2:
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
    with c3:
        st.markdown(
            '<div class="pc-feature-card"><h4>Session History</h4>'
            "<ul>"
            "<li>Every session saves locally (SQLite) - nothing online</li>"
            "<li>Reps-over-time chart per exercise</li>"
            "<li>One-click CSV export</li>"
            "</ul></div>",
            unsafe_allow_html=True,
        )

    st.markdown("#### How it actually works")
    h1, h2 = st.columns(2)
    with h1:
        st.markdown(
            '<div class="pc-feature-card"><h4>Under the hood</h4>'
            '<p class="pc-section-intro" style="margin-bottom:0.6rem;">No neural network makes '
            "the pass/fail call - it's just geometry:</p>"
            "<ul>"
            "<li>MediaPipe tracks 33 body landmarks per frame</li>"
            "<li>10 joint angles, computed with basic vector math</li>"
            "<li>Smoothed across a few frames so it doesn't flicker</li>"
            "<li>Yoga compares to reference ranges from sample images; workout uses "
            "hand-tuned thresholds</li>"
            "</ul></div>",
            unsafe_allow_html=True,
        )
    with h2:
        st.markdown(
            '<div class="pc-feature-card"><h4>Known limits</h4>'
            "<ul>"
            "<li>Auto-detect tested at 88.9% accuracy overall - a few poses are weaker, see "
            "the README for the full breakdown</li>"
            "<li>Form checks are reasonable guesses, not real coaching advice</li>"
            "<li>Works best with one person facing the camera</li>"
            "</ul></div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        '<p style="color:#8b949e; margin-top:1.6rem;">Try it - head to <b>Yoga</b> or '
        "<b>Workout</b> above.</p>",
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


def history_tab():
    sessions = storage.recent_sessions()
    if not sessions:
        st.info("No sessions recorded yet - complete a Yoga or Workout session first.")
        return

    df = pd.DataFrame(sessions)
    st.dataframe(df, use_container_width=True)

    workout_df = df[df["mode"] == "workout"]
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
    st.caption("Real-time pose correction and rep counting, powered by actual joint-angle math.")
    tab0, tab1, tab2, tab3 = st.tabs(["Home", "Yoga", "Workout", "Session History"])
    with tab0:
        home_tab()
    with tab1:
        yoga_tab()
    with tab2:
        workout_tab()
    with tab3:
        history_tab()


if __name__ == "__main__":
    main()
