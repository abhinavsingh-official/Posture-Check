# Posture-Check

A computer vision playground built on one webcam: pose correction and rep counting for yoga and
workouts, plus hand-gesture recognition, drawing in the air, cursor control, and a drowsiness
monitor - six modes, three different MediaPipe trackers (Pose, Hands, Face Mesh), one shared app.

This started as a two-mode yoga/workout form-checker. It grew into this because the underlying
question - "what can you actually build with a webcam and MediaPipe, beyond a skeleton overlay?"
- turned out to have more interesting answers than just one.

## Screenshots

**Home** - real stats pulled live from the data on disk, and a breakdown of all six modes:

![Home tab](screenshots/home.png)

**Yoga** - auto-detect vs. manual pose selection, live angle display, voice feedback:

![Yoga tab](screenshots/yoga.png)

**Workout** - pick an exercise, see its rep-counting joint and form check explained up front:

![Workout tab](screenshots/workout.png)

**Gesture Stickers** - a pretrained hand-sign classifier triggers a procedurally-drawn sticker
(no PNG assets - these are drawn with OpenCV primitives):

![Sticker icons](screenshots/sticker_icons.png)

**Air Canvas** - draw with your fingertip, pinch to lift the pen, pick a color from the toolbar:

![Air Canvas tab](screenshots/air_canvas.png)

**Virtual Mouse** - real hand-landmark tracking feeding cursor control (shown here with cursor
control off - preview-only is the default):

![Virtual Mouse tab](screenshots/virtual_mouse.png)

**Drowsiness Monitor** - a real computed eye-aspect-ratio against a real face:

![Drowsiness Monitor tab](screenshots/drowsiness_monitor.png)

## How it works

Three different MediaPipe trackers feed six different pieces of logic:

**Pose (33 body landmarks) - Yoga and Workout.** For each of 10 tracked joints, the angle between
its three defining landmarks is computed via the standard vector dot-product formula (`arccos` of
the normalized dot product). Yoga compares live angles against per-pose reference ranges mined
from sample images (`extraction.py`); a live angle outside range becomes a corrective hint
("Left Knee: increase by 12.0°"), and holding every angle in range for 5 seconds completes the
pose. Workout counts reps of six exercises (squats, push-ups, bicep curls, lunges, shoulder
presses, lateral raises) through a simple up/down state machine on one representative joint angle
per exercise, with a second, independent check flagging common form mistakes.

**Hands (21 landmarks per hand) - three different uses of the same tracker:**
- *Gesture Stickers* runs a vendored, pretrained hand-sign classifier
  ([kinivi/hand-gesture-recognition-mediapipe](https://github.com/kinivi/hand-gesture-recognition-mediapipe),
  Apache-2.0 - see `YogaPoseDetection/models/gesture/ATTRIBUTION.md`) that recognizes four static
  hand signs (open palm, closed fist, pointing, OK). Each one triggers a different
  procedurally-drawn sticker overlay, alpha-blended onto the feed - the overlay rendering itself
  is original code, built on top of the unmodified classifier.
- *Air Canvas* tracks the index fingertip continuously and draws a line while it moves, using a
  pinch (thumb close to the fingertip) to lift the pen - the same mental model as a real pen,
  rather than a toolbar zone doubling as a pen-up/pen-down switch.
- *Virtual Mouse* maps the index fingertip to the real system cursor and a curled index finger to
  a click, reusing `geometry.calculate_angle` (already built for pose joint angles) for the
  finger-curl check instead of a separate formula. Cursor control is a deliberate, separate opt-in
  behind camera preview - moving your actual mouse is not something that should happen by default
  just because a tab is open - and it auto-disables itself a few seconds after your hand leaves
  the frame.

**Face Mesh (468 landmarks) - Drowsiness Monitor.** Computes eye-aspect-ratio (EAR) - the ratio of
how open each eye is to how wide it is - using MediaPipe's own face landmarks rather than the
dlib + 68-point model most EAR tutorials use; this project already depends on MediaPipe for
everything else, so a second, heavier face-landmark dependency (plus a separate ~100MB model file)
wasn't worth it for a formula MediaPipe's own landmarks already support just as well. Eyes closed
for about half a second triggers an alert; shorter closures are counted as ordinary blinks.

## Features

- **Yoga**: auto-detect or manual pose selection, live angle display, a skeleton overlay colored
  red/green per joint, 5-second hold-to-complete
- **Workout**: rep counting for six exercises with real-time form warnings, not just a rep count
- **Gesture Stickers**: real-time hand-sign recognition with an on-screen sticker response
- **Air Canvas**: draw in the air, pinch-to-lift-pen, a 4-color toolbar
- **Virtual Mouse**: hand-tracked cursor control, opt-in and auto-disabling for safety
- **Drowsiness Monitor**: eye-aspect-ratio tracking with a blink counter and a sustained-closure
  alert, voiced through the same offline text-to-speech used elsewhere in this app
- **Session history**: every Yoga/Workout session is logged to a local SQLite database, with a
  reps-over-time trend, personal-bests summary, and a CSV export
- **Voice feedback**: offline text-to-speech, no internet/API required
- **Two ways to run the original two modes**: the Streamlit app (recommended) or the original
  OpenCV window (`analysis.py`) for Yoga/Workout specifically

## How accurate is yoga auto-detect, really?

Leave-one-out evaluation (`evaluate.py`): for every reference image with at least one sibling in
its pose folder, hold it out, rebuild that pose's angle range from the remaining images, and check
whether auto-detect correctly identifies the held-out image against all 21 poses.

**Result: 88.9% accuracy overall** (81 images evaluated). Per-pose, most poses hit 100%, but a few
are noticeably weaker - bow-pose (33%), and locust/reverse-warrior/sphinx/triangle/upward-salute
(67% each) - which tracks with those poses having only 3 reference images each.

## A real, known data limitation

Reference images per pose range from 3 to 7 - thin enough that `extraction.py` has to widen each
joint's derived range by a fixed ±25° rather than deriving that tolerance from genuine statistical
spread. The `dataset/` folder isn't included in this repo (the images aren't mine to redistribute)
- drop your own into `dataset/<pose-name>/` and re-run `extraction.py` to improve on this.

## Project structure

```
YogaPoseDetection/
  geometry.py        # shared landmark/angle math (reused by Yoga, Workout, and Virtual Mouse)
  feedback.py         # yoga pose comparison
  renderer.py           # skeleton overlay, side panel, and sticker-overlay drawing
  workout.py             # rep counting + form checks
  gestures.py              # hand-sign classification (Gesture Stickers)
  air_canvas.py              # finger-drawing state machine (Air Canvas)
  virtual_mouse.py             # hand-tracked cursor control (Virtual Mouse)
  drowsiness.py                  # eye-aspect-ratio tracking (Drowsiness Monitor)
  storage.py                       # session history (SQLite)
  voice.py                           # offline text-to-speech
  extraction.py                        # builds angles.json from dataset/ images
  evaluate.py                            # leave-one-out accuracy evaluation
  analysis.py                              # OpenCV-window entry point (Yoga/Workout only)
  app.py                                     # Streamlit entry point - all six modes
  angles.json                                  # derived joint-angle ranges (included)
  models/gesture/                                # vendored pretrained hand-sign classifier
  dataset/                                         # NOT included - see note above
  tests/                                             # pytest coverage for the pure logic
```

## Setup

```bash
pip install -r requirements.txt
```

On macOS, if you plan to use Virtual Mouse, grant the terminal/IDE running Streamlit Accessibility
permission (System Settings → Privacy & Security → Accessibility) - `pyautogui`/`pynput` need it
to move the real cursor.

## Run it

**Streamlit app (recommended - all six modes):**

```bash
cd YogaPoseDetection
streamlit run app.py
```

(Streamlit looks for its theme config relative to the current directory, not the script's own
location - running it from the repo root instead would still work, but without the custom theme.)

**Original OpenCV window (Yoga/Workout only):**

```bash
cd YogaPoseDetection
python analysis.py
```

**Run the tests:**

```bash
cd YogaPoseDetection
pytest tests/
```

## Limitations

- Every mode assumes one person, mostly facing the camera - multiple people or extreme angles
  will confuse all three trackers, not just Pose
- Form checks, the drowsiness EAR threshold, and the gesture set (4 static hand signs) are
  reasonable defaults, not medically validated or exhaustive
- Virtual Mouse's click gesture (a curled index finger) can false-trigger if your hand rests in a
  loose fist shape - it's tuned for a deliberate motion, not passive holding
- The thin yoga reference dataset (see above) is the main lever for improving yoga accuracy further

## What I'd improve with more time

- A proper evaluation set for yoga, captured specifically for testing (not reused training images)
- More gesture classes for Gesture Stickers (the vendored model only recognizes four)
- A calibration step for Drowsiness Monitor's EAR threshold, since eye shape varies enough between
  people that one fixed threshold is a compromise, not a precise fit for everyone
- Multi-hand support for Air Canvas (two-handed drawing, or one hand drawing while the other picks
  colors without reaching for the toolbar)
