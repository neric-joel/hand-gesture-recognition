"""
Advanced Hand Gesture Recognition
- Two hands, finger counting (0-10), ASL A-Z, basic gestures
- Thumb detection uses distance from thumb tip to pinky MCP (palm width ref)
Requires: Python 3.11, mediapipe==0.10.14, opencv-python
Run: py -3.11 main.py
"""

import cv2
import mediapipe as mp

mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles  = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75
)

FINGER_TIPS = [8, 12, 16, 20]
FINGER_PIPS = [6, 10, 14, 18]
FINGER_MCPS = [5,  9, 13, 17]
THUMB_TIP   = 4
THUMB_IP    = 3
THUMB_MCP   = 2
INDEX_MCP   = 5
PINKY_MCP   = 17
WRIST       = 0


def lm_dist(lm, a, b):
    return ((lm[a].x - lm[b].x)**2 + (lm[a].y - lm[b].y)**2) ** 0.5


def fingers_extended(lm, handedness):
    """
    Thumb: compare thumb tip distance to pinky MCP vs thumb IP distance to
    pinky MCP. If tip is further away = thumb is extended outward.
    This uses the palm width as a natural reference scale, making it
    robust for both left and right hands at any orientation.
    """
    # Palm reference: distance from index MCP to pinky MCP (palm width)
    palm_width = lm_dist(lm, INDEX_MCP, PINKY_MCP)

    # Thumb tip vs thumb IP — both measured from pinky MCP
    tip_to_ref = lm_dist(lm, THUMB_TIP, PINKY_MCP)
    ip_to_ref  = lm_dist(lm, THUMB_IP,  PINKY_MCP)

    # Thumb is extended if tip is clearly further from pinky MCP than IP joint
    # Use palm_width to normalize the threshold
    thumb = tip_to_ref > ip_to_ref + palm_width * 0.15

    # Four fingers: tip y above PIP y = extended
    four = [lm[tip].y < lm[pip].y for tip, pip in zip(FINGER_TIPS, FINGER_PIPS)]

    return [thumb] + four


def count_fingers(extended):
    return sum(extended)


def detect_basic_gesture(extended):
    t, i, m, r, p = extended
    if t and i and m and r and p:                             return "Open Palm"
    if not t and not i and not m and not r and not p:         return "Fist"
    if t and not i and not m and not r and not p:             return "Thumbs Up"
    if not t and i and m and not r and not p:                 return "Peace Sign"
    if not t and i and not m and not r and not p:             return "Pointing"
    if not t and i and not m and not r and p:                 return "Rock On"
    if t and not i and not m and not r and p:                 return "Call Me"
    return None


def detect_asl(lm, extended, handedness):
    t, i, m, r, p = extended
    tip_x = [lm[tip].x for tip in FINGER_TIPS]

    def near(a, b, thresh=0.07):
        return lm_dist(lm, a, b) < thresh

    # A: Fist, thumb beside index (not tucked under)
    if not i and not m and not r and not p and not t:
        if lm[THUMB_TIP].y < lm[5].y:
            return "A"

    # B: All 4 fingers up, thumb folded across palm
    if i and m and r and p and not t:
        return "B"

    # C: All fingers half-curled into C shape
    if not i and not m and not r and not p and not t:
        half = all(
            lm[tip].y > lm[pip].y and lm[tip].y < lm[mcp].y
            for tip, pip, mcp in zip(FINGER_TIPS, FINGER_PIPS, FINGER_MCPS)
        )
        if half:
            return "C"

    # D: Index up, thumb touches middle finger
    if i and not m and not r and not p:
        if near(THUMB_TIP, 12, 0.08):
            return "D"

    # E: All curled, thumb tucked under fingers
    if not i and not m and not r and not p and not t:
        if lm[THUMB_TIP].y > lm[6].y:
            return "E"

    # F: Index+thumb circle, other 3 up
    if not i and m and r and p:
        if near(THUMB_TIP, 8, 0.07):
            return "F"

    # G: Index pointing sideways
    if i and not m and not r and not p and not t:
        if abs(lm[8].x - lm[6].x) > abs(lm[8].y - lm[6].y) * 1.5:
            return "G"

    # H: Index + middle pointing sideways
    if i and m and not r and not p and not t:
        if abs(lm[8].x - lm[6].x) > abs(lm[8].y - lm[6].y):
            return "H"

    # I: Only pinky up
    if not i and not m and not r and p and not t:
        return "I"

    # K: Index + middle up, thumb raised between them
    if i and m and not r and not p and t:
        if lm[THUMB_TIP].y < lm[6].y:
            return "K"

    # L: Index up + thumb out
    if i and not m and not r and not p and t:
        if abs(lm[8].y - lm[5].y) > 0.1:
            return "L"

    # M: Three fingers over thumb
    if not i and not m and not r and not p and not t:
        if near(6, THUMB_TIP, 0.1) and near(10, THUMB_TIP, 0.1):
            return "M"

    # N: Two fingers over thumb
    if not i and not m and not r and not p and not t:
        if near(6, THUMB_TIP, 0.09) and not near(14, THUMB_TIP, 0.09):
            return "N"

    # O: Fingers curve to meet thumb
    if near(8, THUMB_TIP, 0.07) and near(12, THUMB_TIP, 0.09):
        return "O"

    # R: Index + middle crossed
    if i and m and not r and not p and not t:
        if abs(tip_x[0] - tip_x[1]) < 0.035:
            return "R"

    # S: Fist, thumb over fingers
    if not i and not m and not r and not p and not t:
        if lm[THUMB_TIP].x > lm[8].x - 0.05:
            return "S"

    # T: Thumb between index and middle
    if not i and not m and not r and not p and not t:
        if near(THUMB_TIP, 6, 0.07):
            return "T"

    # U: Index + middle close together
    if i and m and not r and not p and not t:
        if abs(tip_x[0] - tip_x[1]) < 0.045:
            return "U"

    # V: Index + middle spread apart
    if i and m and not r and not p and not t:
        if abs(tip_x[0] - tip_x[1]) >= 0.045:
            return "V"

    # W: Index + middle + ring up
    if i and m and r and not p and not t:
        return "W"

    # X: Index hooked
    if not i and not m and not r and not p and not t:
        if lm[8].y > lm[6].y and lm[8].y < lm[5].y:
            return "X"

    # Y: Thumb + pinky out
    if not i and not m and not r and p and t:
        return "Y"

    # Z: Index pointing up (static fallback)
    if i and not m and not r and not p and not t:
        return "Z"

    return None


def get_label(lm, handedness):
    ext     = fingers_extended(lm, handedness)
    count   = count_fingers(ext)
    asl     = detect_asl(lm, ext, handedness)
    if asl:
        return f"ASL: {asl}", count
    gesture = detect_basic_gesture(ext)
    if gesture:
        return gesture, count
    return "Unknown", count


PANEL_W = 230
PANEL_H = 115
MARGIN  = 18


def best_panel_position(bbox, frame_w, frame_h):
    hx1, hy1, hx2, hy2 = bbox
    hand_cx = (hx1 + hx2) / 2
    hand_cy = (hy1 + hy2) / 2
    px = frame_w - PANEL_W - MARGIN if hand_cx < frame_w / 2 else MARGIN
    py = frame_h - PANEL_H - MARGIN - 30 if hand_cy < frame_h / 2 else MARGIN + 65
    return int(px), int(py)


def draw_panel(frame, x, y, lines, color):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + PANEL_W, y + PANEL_H), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv2.rectangle(frame, (x, y), (x + PANEL_W, y + PANEL_H), color, 2)
    line_h = PANEL_H // (len(lines) + 1)
    for idx, (text, scale, thick) in enumerate(lines):
        ty = y + (idx + 1) * line_h + 6
        cv2.putText(frame, text, (x + 12, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_banner(frame, text, w, color=(0, 255, 255)):
    fs = 1.1
    th = 3
    tw, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, th)[0]
    tx = (w - tw) // 2
    overlay = frame.copy()
    cv2.rectangle(overlay, (tx - 14, 10), (tx + tw + 14, 58), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.putText(frame, text, (tx, 48),
                cv2.FONT_HERSHEY_SIMPLEX, fs, color, th, cv2.LINE_AA)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Hand Gesture Recognition | Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res   = hands.process(rgb)

        hand_data     = []
        total_fingers = 0

        if res.multi_hand_landmarks:
            for idx, hand_lm in enumerate(res.multi_hand_landmarks):
                handedness = res.multi_handedness[idx].classification[0].label

                mp_drawing.draw_landmarks(
                    frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style()
                )

                lm    = hand_lm.landmark
                xs    = [int(p.x * w) for p in lm]
                ys    = [int(p.y * h) for p in lm]
                bbox  = (min(xs), min(ys), max(xs), max(ys))

                label, count  = get_label(lm, handedness)
                total_fingers += count
                hand_data.append((label, count, handedness, bbox))

        for label, count, handedness, bbox in hand_data:
            color  = (80, 255, 120) if handedness == "Right" else (255, 180, 60)
            px, py = best_panel_position(bbox, w, h)
            draw_panel(frame, px, py, [
                (f"{handedness} Hand", 0.6,  2),
                (label,                0.95, 2),
                (f"Fingers: {count}",  0.72, 2),
            ], color)

        if len(hand_data) == 2:
            draw_banner(frame, f"Total: {total_fingers}", w)
        elif len(hand_data) == 1:
            draw_banner(frame, f"Fingers: {hand_data[0][1]}", w)
        else:
            cv2.putText(frame, "No hand detected", (20, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (110, 110, 110), 2, cv2.LINE_AA)

        cv2.putText(frame, "Green=Right  Orange=Left  |  Q=Quit",
                    (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (160, 160, 160), 1, cv2.LINE_AA)

        cv2.imshow("Hand Gesture Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()