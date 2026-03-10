# Hand Gesture Recognition

A real-time hand gesture recognition system built with Python, OpenCV, and MediaPipe.
Detects ASL alphabet, finger counting, and common gestures using webcam input — no ML model training required.

---

## Features

- Two-hand support — tracks both hands simultaneously
- Finger counting — counts fingers per hand and shows combined total (0–10)
- ASL Alphabet — recognizes A–Z fingerspelling in real time
- Basic gestures — Open Palm, Fist, Thumbs Up, Peace Sign, Pointing, Rock On, Call Me
- Clean UI — info panels that move away from your hand, color-coded per hand

---

## Demo

| Gesture | Result |
|--------|--------|
| ✋ Open hand | `Open Palm` — Fingers: 5 |
| ✊ Closed fist | `Fist` — Fingers: 0 |
| 👍 Thumb up | `Thumbs Up` — Fingers: 1 |
| ✌️ Two fingers | `Peace Sign` — Fingers: 2 |
| 🤟 ASL B | `ASL: B` — Fingers: 4 |

---

## Getting Started

### Requirements
- Python 3.11 (MediaPipe does not support Python 3.12+ yet)
- Webcam

### Installation

```bash
pip install opencv-python mediapipe==0.10.14
```

### Run

```bash
python main.py
```

Press **Q** to quit.

---

## Project Structure

```
hand-gesture-recognition/
├── main.py            # Main application
├── requirements.txt   # Dependencies
└── README.md
```

---

## How It Works

No ML model training is used. Gesture detection is based purely on MediaPipe hand landmark positions:

- 21 landmarks are detected per hand (fingertips, joints, wrist)
- Finger extension is determined by comparing tip vs PIP joint y-coordinates
- Thumb detection uses distance from thumb tip to pinky MCP (palm width reference) for accuracy at any hand angle
- ASL letters are identified using spatial relationships between landmarks (distances, angles, relative positions)

---

## Dependencies

| Package | Version |
|---------|---------|
| opencv-python | latest |
| mediapipe | 0.10.14 |

---

## License

MIT License — free to use and modify.

---

## Author

**neric-joel** — [GitHub](https://github.com/neric-joel)
