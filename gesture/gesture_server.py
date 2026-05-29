# Streams gesture tokens over TCP. Uses MediaPipe Tasks + $1 recognizer + trajectory.

import os
import socket
import time
import collections

import cv2
from dollarpy import Point

import mediapipe_hands_helper as mph
import textdollar

GESTURE_HOST = "127.0.0.1"
GESTURE_PORT = 5001

MIN_GESTURE_CONFIDENCE = 0.3
MIN_SECONDS_BETWEEN_SENDS = 0.5
RECOGNIZE_EVERY_FRAMES = 23
CAMERA_INDEX = int(os.environ.get("GESTURE_CAMERA_INDEX", "0"))
ACCEPT_TIMEOUT_SEC = 0.05

TRAJECTORY_WINDOW = 20
MIN_DISPLACEMENT_PX = 60
TRAJECTORY_DIRECTION_RATIO = 1.4


class TrajectoryAnalyzer:
    """Tracks index fingertip over sliding window to determine swipe direction."""

    def __init__(self, window_size=TRAJECTORY_WINDOW):
        self.positions = collections.deque(maxlen=window_size)
        self.last_direction = None
        self.last_direction_time = 0.0

    def update(self, hand_landmarks, frame_w, frame_h):
        if hand_landmarks and len(hand_landmarks) >= 9:
            # Landmark 8 = index fingertip
            lm = hand_landmarks[8]
            px = int(lm.x * frame_w)
            py = int(lm.y * frame_h)
            self.positions.append((px, py, time.time()))

    def get_direction(self):
        if len(self.positions) < self.positions.maxlen // 2:
            return None, 0

        # Use the oldest and newest positions in the window
        old_x, old_y, old_t = self.positions[0]
        new_x, new_y, new_t = self.positions[-1]

        # Need at least some time elapsed (not instant)
        if new_t - old_t < 0.15:
            return None, 0

        dx = new_x - old_x
        dy = new_y - old_y

        abs_dx = abs(dx)
        abs_dy = abs(dy)

        total_displacement = (dx * dx + dy * dy) ** 0.5

        if total_displacement < MIN_DISPLACEMENT_PX:
            return None, 0

        # Determine dominant direction
        if abs_dy > abs_dx * TRAJECTORY_DIRECTION_RATIO:
            # Vertical swipe
            if dy < 0:
                return "UP", total_displacement
            else:
                return "DOWN", total_displacement
        elif abs_dx > abs_dy * TRAJECTORY_DIRECTION_RATIO:
            # Horizontal swipe
            if dx < 0:
                return "LEFTSWIPE", total_displacement
            else:
                return "RIGHTSWIPE", total_displacement

        return None, 0

    def clear(self):
        """Clear the trajectory buffer (after sending a gesture)."""
        self.positions.clear()
        self.last_direction = None


def label_to_token(label):
    """Turn dollarpy template name into one line for the socket."""
    if not label:
        return None
    name = str(label).strip().lower()
    normalized = name.replace("_", "").replace("-", "").replace(" ", "")
    if name.startswith("up"):
        return "UP"
    if name.startswith("down"):
        return "DOWN"
    if (
        name.startswith("left")
        or normalized.startswith("swipeleft")
        or normalized.startswith("leftswipe")
    ):
        return "LEFTSWIPE"
    if (
        name.startswith("right")
        or normalized.startswith("swiperight")
        or normalized.startswith("rightswipe")
    ):
        return "RIGHTSWIPE"
    return None


def main():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((GESTURE_HOST, GESTURE_PORT))
    srv.listen(1)
    srv.settimeout(ACCEPT_TIMEOUT_SEC)

    landmarker = mph.create_hand_landmarker()
    recognizer = textdollar.recognizer
    trajectory = TrajectoryAnalyzer()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("ERROR: Cannot open camera index", CAMERA_INDEX, "- try CAMERA_INDEX = 1 in the script.")
        landmarker.close()
        srv.close()
        return

    cv2.namedWindow("Gesture server", cv2.WINDOW_AUTOSIZE)

    print(
        "Gesture server on",
        GESTURE_HOST + ":" + str(GESTURE_PORT),
        "- camera should open now. Start TuioDemo to connect. q = quit.",
    )
    print("Hybrid mode: $1 recognizer + trajectory analyzer active.")

    conn = None
    writer = None
    all_points = []
    framecent = 0
    last_send_time = 0.0
    timestamp_ms = 0
    last_sent_token = None

    try:
        while True:
            if writer is None:
                try:
                    conn, addr = srv.accept()
                    print("Client connected from", addr)
                    writer = conn.makefile("w", encoding="utf-8", newline="\n")
                except socket.timeout:
                    pass

            ret, frame = cap.read()
            if not ret:
                print("Camera read failed, exiting.")
                break

            frame = cv2.resize(frame, (640, 480))
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            framecent += 1
            timestamp_ms += 33

            if writer is None:
                cv2.putText(
                    frame,
                    "Waiting for TuioDemo (port %d)..." % GESTURE_PORT,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 165, 255),
                    2,
                )

            hand = mph.detect_hands(landmarker, frame, timestamp_ms)

            if not hand:
                trajectory.clear()
                cv2.imshow("Gesture server", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            # --- Feed both systems ---

            # 1. Feed $1 recognizer (all 21 landmarks)
            for lm in hand:
                px = int(lm.x * w)
                py = int(lm.y * h)
                all_points.append(Point(px, py, 1))

            # 2. Feed trajectory analyzer (index fingertip only)
            trajectory.update(hand, w, h)

            # --- Hybrid decision every N frames ---
            token_to_send = None

            if framecent % RECOGNIZE_EVERY_FRAMES == 0:
                framecent = 0

                # Get $1 result
                dollar_result = recognizer.recognize(all_points)
                all_points = all_points[-210:]

                dollar_token = None
                dollar_conf = dollar_result[1]
                if dollar_conf >= MIN_GESTURE_CONFIDENCE:
                    dollar_token = label_to_token(dollar_result[0])

                # Get trajectory result
                traj_direction, traj_displacement = trajectory.get_direction()

                # --- Hybrid scoring ---
                if dollar_token and traj_direction:
                    if dollar_token == traj_direction:
                        # Both agree -- high confidence, send it
                        token_to_send = traj_direction
                        print(
                            "HYBRID [AGREE]:",
                            traj_direction,
                            "| $1:",
                            dollar_result[0],
                            "conf=%.2f" % dollar_conf,
                            "| traj: %.0fpx" % traj_displacement,
                        )
                    else:
                        # They disagree -- trust trajectory for direction
                        # because $1 normalizes rotation
                        token_to_send = traj_direction
                        print(
                            "HYBRID [TRAJ WINS]:",
                            traj_direction,
                            "| $1 said:",
                            dollar_token,
                            "conf=%.2f" % dollar_conf,
                            "| traj: %.0fpx" % traj_displacement,
                        )
                elif traj_direction and traj_displacement >= MIN_DISPLACEMENT_PX * 1.3:
                    # $1 didn't fire but trajectory is very clear
                    token_to_send = traj_direction
                    print(
                        "HYBRID [TRAJ ONLY]:",
                        traj_direction,
                        "| traj: %.0fpx" % traj_displacement,
                    )
                elif dollar_token:
                    # $1 fired but trajectory didn't -- still use $1
                    token_to_send = dollar_token
                    print(
                        "HYBRID [$1 ONLY]:",
                        dollar_token,
                        "| $1:",
                        dollar_result[0],
                        "conf=%.2f" % dollar_conf,
                    )

            # --- Send result ---
            if token_to_send and writer is not None:
                now = time.time()
                if now - last_send_time >= MIN_SECONDS_BETWEEN_SENDS:
                    try:
                        writer.write(token_to_send + "\n")
                        writer.flush()
                        last_send_time = now
                        last_sent_token = token_to_send
                        trajectory.clear()
                        print(">>> Sent:", token_to_send)
                    except (BrokenPipeError, ConnectionResetError, OSError) as ex:
                        print("Client disconnected:", ex)
                        try:
                            writer.close()
                        except OSError:
                            pass
                        try:
                            conn.close()
                        except OSError:
                            pass
                        writer = None
                        conn = None

            # --- Draw overlay ---
            mph.draw_hands_on_bgr(frame, hand)

            # Draw trajectory trail (green line showing fingertip path)
            pts = list(trajectory.positions)
            for i in range(1, len(pts)):
                cv2.line(
                    frame,
                    (pts[i - 1][0], pts[i - 1][1]),
                    (pts[i][0], pts[i][1]),
                    (0, 255, 0),
                    2,
                )

            # Show last sent gesture on screen
            if last_sent_token and time.time() - last_send_time < 1.5:
                cv2.putText(
                    frame,
                    "Gesture: " + last_sent_token,
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                )

            cv2.imshow("Gesture server", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        if writer is not None:
            try:
                writer.close()
            except OSError:
                pass
        if conn is not None:
            try:
                conn.close()
            except OSError:
                pass
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()
        srv.close()


if __name__ == "__main__":
    main()