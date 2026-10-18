"""Streamlit frontend - the real UI upgrade over the raw OpenCV window
analysis.py uses. Same underlying logic (geometry/feedback/renderer/
workout), just a different display loop: st.image() instead of
cv2.imshow(), since Streamlit can't show an OpenCV window directly."""

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
SMOOTHING_FRAMES = 7


def yoga_tab():
    angle_data = load_angle_data()
    pose_list = list(angle_data.keys())

    col1, col2 = st.columns([3, 1])
    with col2:
        detect_mode = st.radio("Mode", ["Auto-detect", "Manual"])
        selected_pose = None if detect_mode == "Auto-detect" else st.selectbox("Pose", pose_list)
        live_angles_on = st.checkbox("Show live angles")
        voice_on = st.checkbox("Voice feedback", key="yoga_voice")
        run = st.button("Start session", key="yoga_start")
        pose_metric = st.empty()
        hold_metric = st.empty()
        hints_box = st.empty()

    with col1:
        frame_slot = st.empty()

    if not run:
        return

    voice = VoiceFeedback(enabled=voice_on)
    cap = cv2.VideoCapture(0)
    smoothing_buffer: list[dict[str, float]] = []
    detection_start_time = None
    hold_start = None
    session_start = time.time()

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose_model:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            results = pose_model.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if not results.pose_landmarks:
                frame_slot.image(frame, channels="BGR")
                continue

            landmarks = results.pose_landmarks
            lm = landmarks.landmark
            if not geometry.full_body_visible(lm):
                frame_slot.image(frame, channels="BGR")
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
                    pose_metric.metric("Pose", f"Detecting... {int(AUTO_DETECT_DELAY - elapsed)}s")
                    frame_slot.image(frame, channels="BGR")
                    continue
                active_pose, _ = feedback.auto_detect_pose(smoothed, angle_data)

            score, hints = feedback.compare_pose(smoothed, active_pose, angle_data)
            correct = feedback.is_pose_correct(score)
            hold_start = hold_start if correct and hold_start else (time.time() if correct else None)
            hold_time = (time.time() - hold_start) if hold_start else 0

            if voice and hints:
                voice.say(hints[0])

            bad_joints = renderer.bad_joints_from_hints(hints)
            renderer.draw_skeleton(frame, landmarks, bad_joints)
            frame_slot.image(frame, channels="BGR")
            pose_metric.metric("Pose", active_pose or "None")
            hold_metric.metric("Hold time", f"{hold_time:.1f}s / {HOLD_TIME_REQUIRED}s")
            hints_box.warning("\n".join(hints)) if hints else hints_box.success("Looking good!")

            if hold_time >= HOLD_TIME_REQUIRED:
                storage.log_session(storage.SessionRecord(
                    mode="yoga", name=active_pose, duration_seconds=time.time() - session_start,
                ))
                st.success(f"Held {active_pose} correctly for {HOLD_TIME_REQUIRED}s!")
                break

    cap.release()


def workout_tab():
    col1, col2 = st.columns([3, 1])
    with col2:
        exercise = st.selectbox("Exercise", list(EXERCISES.keys()))
        voice_on = st.checkbox("Voice feedback", key="workout_voice")
        run = st.button("Start session", key="workout_start")
        stop = st.button("Stop & save", key="workout_stop")
        rep_metric = st.empty()
        warning_box = st.empty()

    with col1:
        frame_slot = st.empty()

    if not run:
        return

    voice = VoiceFeedback(enabled=voice_on)
    counter = RepCounter(exercise=exercise)
    session_start = time.time()
    cap = cv2.VideoCapture(0)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose_model:
        while cap.isOpened() and not stop:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            results = pose_model.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if results.pose_landmarks:
                counter.update(results.pose_landmarks.landmark)
                form_warning = workout.check_form(exercise, results.pose_landmarks.landmark)
                if form_warning:
                    counter.form_warnings += 1
                    if voice:
                        voice.say(form_warning)
                renderer.draw_skeleton(frame, results.pose_landmarks, set())
                warning_box.warning(form_warning) if form_warning else warning_box.success("Good form")

            frame_slot.image(frame, channels="BGR")
            rep_metric.metric("Reps", counter.count)

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
    st.title("Posture-Check")
    tab1, tab2, tab3 = st.tabs(["Yoga", "Workout", "Session History"])
    with tab1:
        yoga_tab()
    with tab2:
        workout_tab()
    with tab3:
        history_tab()


if __name__ == "__main__":
    main()
