# Posture-Check

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white) ![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white) ![MediaPipe](https://img.shields.io/badge/MediaPipe-0097A7?style=for-the-badge&logo=google&logoColor=white) ![SQLite](https://img.shields.io/badge/SQLite-003B57?style=for-the-badge&logo=sqlite&logoColor=white) ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

A computer vision playground built on one webcam, powered by three
MediaPipe trackers (Pose, Hands, Face Mesh) feeding six real-time modes:
yoga form correction with live joint-angle feedback and auto pose
detection, workout rep counting across six exercises with form checks,
hand-gesture recognition that triggers procedurally-drawn sticker overlays,
an air canvas you draw in with your fingertip, a hand-tracked virtual
mouse, and a drowsiness monitor built on eye-aspect-ratio tracking. Every
Yoga/Workout session is logged to SQLite with a reps-over-time trend and a
personal-bests summary, and feedback is spoken through offline
text-to-speech.

## Screenshots

![Home tab](screenshots/home.png)
![Yoga tab](screenshots/yoga.png)
![Workout tab](screenshots/workout.png)
![Sticker icons](screenshots/sticker_icons.png)
![Air Canvas tab](screenshots/air_canvas.png)
![Virtual Mouse tab](screenshots/virtual_mouse.png)
![Drowsiness Monitor tab](screenshots/drowsiness_monitor.png)

## How to run

```bash
pip install -r requirements.txt
cd YogaPoseDetection
streamlit run app.py
```

On macOS, Virtual Mouse needs Accessibility permission granted to the
terminal/IDE (System Settings → Privacy & Security → Accessibility).

Tests: `cd YogaPoseDetection && pytest tests/`
