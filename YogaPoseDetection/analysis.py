import cv2
import time
import json
import numpy as np
import mediapipe as mp

LIVE_ANGLE_DISPLAY = False
FULL_BODY_THRESHOLD = 0.80
AUTO_DETECT_DELAY = 5
SMOOTHING_FRAMES = 7
START_MESSAGE_DURATION = 5
HOLD_TIME_REQUIRED = 5

with open("angles.json", "r") as f:
    ANGLE_DATA = json.load(f)

POSE_LIST = list(ANGLE_DATA.keys())

mp_pose = mp.solutions.pose
POSE_LMS = mp_pose.PoseLandmark

JOINTS = {
    "left_knee": ("LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"),
    "right_knee": ("RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"),
    "left_elbow": ("LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"),
    "right_elbow": ("RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"),
    "left_shoulder": ("LEFT_ELBOW", "LEFT_SHOULDER", "LEFT_HIP"),
    "right_shoulder": ("RIGHT_ELBOW", "RIGHT_SHOULDER", "RIGHT_HIP"),
    "left_hip": ("RIGHT_HIP", "LEFT_HIP", "LEFT_KNEE"),
    "right_hip": ("LEFT_HIP", "RIGHT_HIP", "RIGHT_KNEE"),
    "neck": ("LEFT_SHOULDER", "NOSE", "RIGHT_SHOULDER"),
    "spine": ("LEFT_HIP", "RIGHT_HIP", "NOSE"),
}

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    v1, v2 = a - b, c - b
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0:
        return None
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / denom, -1, 1))))

def get_landmark_point(landmarks, name):
    lm = landmarks[POSE_LMS[name].value]
    return [lm.x, lm.y, lm.z]

def get_all_angles(landmarks):
    angles = {}
    for j, (p1, p2, p3) in JOINTS.items():
        try:
            a = get_landmark_point(landmarks, p1)
            b = get_landmark_point(landmarks, p2)
            c = get_landmark_point(landmarks, p3)
            ang = calculate_angle(a, b, c)
            if ang is not None:
                angles[j] = ang
        except:
            pass
    return angles

def compare_pose(user_angles, pose_name):
    data = ANGLE_DATA[pose_name]
    diffs = []
    hints = []
    for joint, val in user_angles.items():
        if joint not in data:
            continue
        mn = data[joint]["angle_min"]
        mx = data[joint]["angle_max"]
        if val < mn:
            diff = mn - val
            diffs.append(diff)
            hints.append(f"{joint.replace('_',' ').title()}: increase by {diff:.1f}")
        elif val > mx:
            diff = val - mx
            diffs.append(diff)
            hints.append(f"{joint.replace('_',' ').title()}: decrease by {diff:.1f}")
        else:
            diffs.append(0)
    score = np.mean(diffs) if diffs else 999
    return score, hints

def auto_detect_pose(user_angles):
    best_pose = None
    best_score = 999
    for pose in POSE_LIST:
        score, _ = compare_pose(user_angles, pose)
        if score < best_score:
            best_score = score
            best_pose = pose
    return best_pose, best_score

def full_body_visible(landmarks):
    visible = [lm for lm in landmarks if 0 < lm.x < 1 and 0 < lm.y < 1]
    return len(visible) / len(landmarks) >= FULL_BODY_THRESHOLD

# -------------------------
# Dark-mode PANEL (hints only)
# -------------------------
def draw_side_panel(selected_pose, hold_time, feedback, live_angles):
    panel = np.zeros((720, 380, 3), dtype=np.uint8)
    panel[:] = (25, 25, 25)  # dark background

    white = (255, 255, 255)
    accent = (120, 180, 255)
    good = (120, 255, 120)

    font_title = 0.8
    font_txt = 0.6
    font_small = 0.55

    y = 45

    cv2.putText(panel, f"Pose: {selected_pose}", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, font_title, white, 2)
    y += 35

    cv2.putText(panel, f"Hold: {hold_time:.1f}s", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, font_title, good, 2)
    y += 40

    if live_angles:
        cv2.putText(panel, "Live Angles:", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_txt, accent, 2)
        y += 30
        for j, v in live_angles.items():
            cv2.putText(panel, f"{j}: {int(v)}", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_small, white, 1)
            y += 22
        y += 15

    cv2.putText(panel, "Hints:", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, font_txt, accent, 2)
    y += 28

    for f in feedback:
        cv2.putText(panel, f, (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_small, white, 1)
        y += 22

    return panel

# -------------------------
# MAIN
# -------------------------
def main():

    print("\nSelect Mode:")
    print("1) Auto Detect Pose")
    print("2) Manual Selection")
    mode = input("Enter choice (1/2): ").strip()

    selected_pose = None
    if mode == "2":
        print("\nAvailable poses:")
        for i, p in enumerate(POSE_LIST):
            print(f"{i+1}) {p}")
        idx = int(input("Select pose number: "))
        selected_pose = POSE_LIST[idx - 1]

    live_choice = input("\nEnable live angle display? (y/n): ").strip().lower()
    global LIVE_ANGLE_DISPLAY
    LIVE_ANGLE_DISPLAY = (live_choice == "y")

    cap = cv2.VideoCapture(0)

    cv2.namedWindow("Yoga", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Yoga", 1900, 1050)

    smoothing_buffer = []
    detection_start_time = None
    start_message_time = time.time()
    hold_start = None

    with mp_pose.Pose(min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose_model:

        while True:

            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)

            results = pose_model.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            frame_h = frame.shape[0]
            panel_w = 380

            # No landmarks detected
            if not results.pose_landmarks:
                cv2.putText(frame, "Stand properly in frame",
                            (30, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (255, 255, 255), 2)
                panel = draw_side_panel("None", 0, [], {})
                panel = cv2.resize(panel, (panel_w, frame_h))
                combined = np.hstack([frame, panel])
                cv2.imshow("Yoga", combined)
                if cv2.waitKey(1) == 27:
                    break
                continue

            lm = results.pose_landmarks.landmark

            # Full body not visible
            if not full_body_visible(lm):
                cv2.putText(frame, "Move back so your full body is visible",
                            (30, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255, 255, 255), 2)
                panel = draw_side_panel("None", 0, [], {})
                panel = cv2.resize(panel, (panel_w, frame_h))
                combined = np.hstack([frame, panel])
                cv2.imshow("Yoga", combined)
                if cv2.waitKey(1) == 27:
                    break
                continue

            # Angles + smoothing
            user_angles = get_all_angles(lm)
            smoothing_buffer.append(user_angles)
            if len(smoothing_buffer) > SMOOTHING_FRAMES:
                smoothing_buffer.pop(0)

            smoothed = {}
            for j in JOINTS.keys():
                vals = [d.get(j) for d in smoothing_buffer if d.get(j)]
                if vals:
                    smoothed[j] = sum(vals) / len(vals)

            # Initial "start yoga" message (on main screen only)
            if time.time() - start_message_time <= START_MESSAGE_DURATION:
                cv2.putText(frame, "Please start your yoga",
                            (30, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (255, 255, 255), 2)
                panel = draw_side_panel("None", 0, [], {})
                panel = cv2.resize(panel, (panel_w, frame_h))
                combined = np.hstack([frame, panel])
                cv2.imshow("Yoga", combined)
                if cv2.waitKey(1) == 27:
                    break
                continue

            # Auto-detect mode
            if mode == "1":
                if detection_start_time is None:
                    detection_start_time = time.time()

                elapsed = time.time() - detection_start_time
                if elapsed < AUTO_DETECT_DELAY:
                    remaining = AUTO_DETECT_DELAY - int(elapsed)
                    cv2.putText(frame, f"Hold still... detecting pose in {remaining}s",
                                (30, 40), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (255, 255, 255), 2)
                    panel = draw_side_panel("Detecting...", 0, [], {})
                    panel = cv2.resize(panel, (panel_w, frame_h))
                    combined = np.hstack([frame, panel])
                    cv2.imshow("Yoga", combined)
                    if cv2.waitKey(1) == 27:
                        break
                    continue

                autodetected_pose, _ = auto_detect_pose(smoothed)
                selected_pose = autodetected_pose

            # Manual mode already has selected_pose

            score, hints = compare_pose(smoothed, selected_pose)

            # Hold timer
            if score < 10:
                if hold_start is None:
                    hold_start = time.time()
            else:
                hold_start = None

            hold_time = (time.time() - hold_start) if hold_start else 0

            if hold_time >= HOLD_TIME_REQUIRED:
                print("\nPose held correctly for 5 seconds. Exiting...")
                break

            panel = draw_side_panel(
                selected_pose,
                hold_time,
                hints,
                smoothed if LIVE_ANGLE_DISPLAY else {}
            )

            panel = cv2.resize(panel, (panel_w, frame_h))
            combined = np.hstack([frame, panel])

            cv2.imshow("Yoga", combined)
            if cv2.waitKey(1) == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
