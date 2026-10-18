"""Entry point - wires together landmark extraction (geometry.py), pose
comparison (feedback.py), rep counting (workout.py), and rendering
(renderer.py). Two modes: Yoga (hold a static pose for N seconds) and
Workout (count reps of a dynamic exercise)."""

import json
import time

import cv2
import mediapipe as mp

import geometry
import feedback
import renderer
from workout import EXERCISES, RepCounter

FULL_BODY_THRESHOLD = 0.80
AUTO_DETECT_DELAY = 5
SMOOTHING_FRAMES = 7
START_MESSAGE_DURATION = 5
HOLD_TIME_REQUIRED = 5
PANEL_WIDTH = 380

mp_pose = mp.solutions.pose


def load_angle_data(path: str = "angles.json") -> dict:
    with open(path) as f:
        return json.load(f)


def smooth_angles(buffer: list[dict[str, float]], joints: dict) -> dict[str, float]:
    smoothed = {}
    for joint in joints:
        values = [frame[joint] for frame in buffer if joint in frame]
        if values:
            smoothed[joint] = sum(values) / len(values)
    return smoothed


def run_yoga_mode(cap, pose_model, angle_data: dict, auto_detect: bool, live_angles_on: bool):
    pose_list = list(angle_data.keys())
    selected_pose = None
    if not auto_detect:
        print("\nAvailable poses:")
        for i, name in enumerate(pose_list):
            print(f"{i + 1}) {name}")
        idx = int(input("Select pose number: "))
        selected_pose = pose_list[idx - 1]

    smoothing_buffer: list[dict[str, float]] = []
    detection_start_time = None
    start_message_time = time.time()
    hold_start = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        results = pose_model.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_h = frame.shape[0]

        if not results.pose_landmarks:
            cv2.putText(frame, "Stand properly in frame", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            panel = renderer.draw_side_panel("None", "", [], {}, frame_h, PANEL_WIDTH)
            cv2.imshow("Yoga", renderer.compose_frame(frame, panel))
            if cv2.waitKey(1) == 27:
                return
            continue

        landmarks = results.pose_landmarks
        lm = landmarks.landmark

        if not geometry.full_body_visible(lm, FULL_BODY_THRESHOLD):
            cv2.putText(frame, "Move back - full body not visible", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            panel = renderer.draw_side_panel("None", "", [], {}, frame_h, PANEL_WIDTH)
            cv2.imshow("Yoga", renderer.compose_frame(frame, panel))
            if cv2.waitKey(1) == 27:
                return
            continue

        if time.time() - start_message_time <= START_MESSAGE_DURATION:
            cv2.putText(frame, "Please start your yoga", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            panel = renderer.draw_side_panel("None", "", [], {}, frame_h, PANEL_WIDTH)
            cv2.imshow("Yoga", renderer.compose_frame(frame, panel))
            if cv2.waitKey(1) == 27:
                return
            continue

        user_angles = geometry.get_all_angles(lm)
        smoothing_buffer.append(user_angles)
        if len(smoothing_buffer) > SMOOTHING_FRAMES:
            smoothing_buffer.pop(0)
        smoothed = smooth_angles(smoothing_buffer, geometry.JOINTS)

        if auto_detect:
            if detection_start_time is None:
                detection_start_time = time.time()
            elapsed = time.time() - detection_start_time
            if elapsed < AUTO_DETECT_DELAY:
                remaining = AUTO_DETECT_DELAY - int(elapsed)
                cv2.putText(frame, f"Hold still... detecting pose in {remaining}s", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                panel = renderer.draw_side_panel("Detecting...", "", [], {}, frame_h, PANEL_WIDTH)
                cv2.imshow("Yoga", renderer.compose_frame(frame, panel))
                if cv2.waitKey(1) == 27:
                    return
                continue
            selected_pose, _ = feedback.auto_detect_pose(smoothed, angle_data)

        score, hints = feedback.compare_pose(smoothed, selected_pose, angle_data)
        correct = feedback.is_pose_correct(score)

        hold_start = hold_start if correct and hold_start else (time.time() if correct else None)
        hold_time = (time.time() - hold_start) if hold_start else 0

        if hold_time >= HOLD_TIME_REQUIRED:
            print("\nPose held correctly for 5 seconds. Done!")
            return

        bad_joints = renderer.bad_joints_from_hints(hints)
        renderer.draw_skeleton(frame, landmarks, bad_joints)
        panel = renderer.draw_side_panel(
            selected_pose or "None", f"Hold: {hold_time:.1f}s", hints,
            smoothed if live_angles_on else {}, frame_h, PANEL_WIDTH,
        )
        cv2.imshow("Yoga", renderer.compose_frame(frame, panel))
        if cv2.waitKey(1) == 27:
            return


def run_workout_mode(cap, pose_model, exercise: str):
    counter = RepCounter(exercise=exercise)
    label = EXERCISES[exercise]["label"]

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        results = pose_model.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_h = frame.shape[0]

        if not results.pose_landmarks:
            cv2.putText(frame, "Stand properly in frame", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            panel = renderer.draw_side_panel(exercise.replace("_", " ").title(), f"Reps: {counter.count}", [], {}, frame_h, PANEL_WIDTH)
            cv2.imshow("Yoga", renderer.compose_frame(frame, panel))
            if cv2.waitKey(1) == 27:
                return
            continue

        angle = counter.update(results.pose_landmarks.landmark)
        renderer.draw_skeleton(frame, results.pose_landmarks, set())
        live_angles = {label: angle} if angle is not None else {}
        panel = renderer.draw_side_panel(
            exercise.replace("_", " ").title(), f"Reps: {counter.count}", [f"State: {counter.state}"], live_angles, frame_h, PANEL_WIDTH,
        )
        cv2.imshow("Yoga", renderer.compose_frame(frame, panel))
        if cv2.waitKey(1) == 27:
            return


def main():
    print("\nSelect Mode:")
    print("1) Yoga - Auto Detect Pose")
    print("2) Yoga - Manual Pose Selection")
    print("3) Workout - Rep Counter")
    mode = input("Enter choice (1/2/3): ").strip()

    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Yoga", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Yoga", 1900, 1050)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose_model:
        if mode in ("1", "2"):
            angle_data = load_angle_data()
            live_choice = input("\nEnable live angle display? (y/n): ").strip().lower() == "y"
            run_yoga_mode(cap, pose_model, angle_data, auto_detect=(mode == "1"), live_angles_on=live_choice)
        elif mode == "3":
            print("\nAvailable exercises:")
            names = list(EXERCISES.keys())
            for i, name in enumerate(names):
                print(f"{i + 1}) {name}")
            idx = int(input("Select exercise number: "))
            run_workout_mode(cap, pose_model, names[idx - 1])
        else:
            print("Invalid choice.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
