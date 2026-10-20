`keypoint_classifier.tflite` and `keypoint_classifier_label.csv` are vendored, unmodified, from
[kinivi/hand-gesture-recognition-mediapipe](https://github.com/kinivi/hand-gesture-recognition-mediapipe)
(itself a translation/fork of Kazuhito Takahashi's original work), licensed Apache License 2.0.
See `LICENSE-gesture-model.txt` in this directory for the full license text.

The model is a small MLP trained on normalized MediaPipe hand-landmark coordinates to classify
four static hand signs: Open, Close, Pointer, OK. This project uses it unmodified; the sticker/
emoji overlay compositing (`gestures.py`, `renderer.py`) that renders a response to each
recognized gesture is original code written for this project.
