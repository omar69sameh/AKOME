"""Gesture recognition TCP server — laser/hand modes with dollarpy $P recognizer."""

from __future__ import annotations

import json
import os
import socketserver
import sys
import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from dollarpy import Recognizer, Template, Point

# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.normpath(os.path.join(_HERE, ".."))
sys.path.insert(0, _PROJECT)

LASER_TEMPLATES = os.path.join(_HERE, "templates.json")
HAND_TEMPLATES = os.path.join(_PROJECT, "hand_recorder", "hand_templates.json")

# ---------------------------------------------------------------------------
# network
# ---------------------------------------------------------------------------
HOST = os.environ.get("DRAW_ANSWER_LASER_HOST", "127.0.0.1")
PORT = int(os.environ.get("DRAW_ANSWER_LASER_PORT", "5002"))

# ---------------------------------------------------------------------------
# camera / processing constants
# ---------------------------------------------------------------------------
FRAME_W, FRAME_H = 640, 480
MIN_BRIGHT_THRESH = 200          # laser mode
LASER_RADIUS = 8
TRAIL_MAX_POINTS = 700
GESTURE_MIN_PTS = 10

COLOR_TRAIL_HEAD = (0, 255, 255)
COLOR_TRAIL_TAIL = (0, 120, 200)
COLOR_DOT = (0, 255, 0)

# ---------------------------------------------------------------------------
# hand landmarker (lazy)
# ---------------------------------------------------------------------------
_hand_landmarker = None
_hand_ts_ms = 0
_hand_smooth_x = None
_hand_smooth_y = None


def _get_hand_landmarker():
    global _hand_landmarker
    if _hand_landmarker is not None:
        return _hand_landmarker
    import mediapipe as mp
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    model_path = os.path.join(_PROJECT, "gesture", "hand_landmarker.task")
    if not os.path.isfile(model_path):
        print("[draw_answer_server] hand_landmarker.task NOT found at", model_path)
        return None
    _hand_landmarker = HandLandmarker.create_from_options(
        HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )
    print("[draw_answer_server] MediaPipe HandLandmarker loaded")
    return _hand_landmarker


def detect_hand_tip(frame_bgr):
    global _hand_ts_ms, _hand_smooth_x, _hand_smooth_y
    lm = _get_hand_landmarker()
    if lm is None:
        return None
    import mediapipe as mp
    _hand_ts_ms += 33
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = lm.detect_for_video(mp_image, _hand_ts_ms)
    if result.hand_landmarks and len(result.hand_landmarks) > 0:
        h, w, _ = frame_bgr.shape
        raw_x = int(result.hand_landmarks[0][8].x * w)
        raw_y = int(result.hand_landmarks[0][8].y * h)
        if _hand_smooth_x is None:
            _hand_smooth_x, _hand_smooth_y = raw_x, raw_y
        else:
            alpha = 0.35
            _hand_smooth_x = int(_hand_smooth_x * (1 - alpha) + raw_x * alpha)
            _hand_smooth_y = int(_hand_smooth_y * (1 - alpha) + raw_y * alpha)
        return _hand_smooth_x, _hand_smooth_y
    return None

# ---------------------------------------------------------------------------
# Object mode removed — laser and hand only
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# shape canonicalisation
# ---------------------------------------------------------------------------
def canonical_shape_name(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    s = raw.strip().lower()
    if s == "square":
        return "square"
    if s == "circle":
        return "circle"
    if s == "triangle":
        return "triangle"
    return None

# ---------------------------------------------------------------------------
# template loading
# ---------------------------------------------------------------------------
def _load_templates_from(path: str) -> List[Template]:
    templates: List[Template] = []
    if not os.path.isfile(path):
        print(f"[draw_answer_server] templates file NOT found: {path}")
        return templates
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            nm = item.get("name") or ""
            canon = canonical_shape_name(nm)
            if not canon:
                continue
            pts_list = item.get("points") or []
            if len(pts_list) < 5:
                continue
            pts = [Point(float(x), float(y), 0) for x, y in pts_list]
            templates.append(Template(canon, pts))
    except Exception as ex:
        print(f"[draw_answer_server] Failed to load {path}:", ex)
    return templates


_RECOGNIZER_LOCK = threading.Lock()
_RECOGNIZER_CACHE: Dict[str, Optional[Recognizer]] = {}


def get_recognizer(mode: str) -> Optional[Recognizer]:
    global _RECOGNIZER_CACHE
    with _RECOGNIZER_LOCK:
        if mode in _RECOGNIZER_CACHE:
            return _RECOGNIZER_CACHE[mode]
        path_map = {"laser": LASER_TEMPLATES, "hand": HAND_TEMPLATES}
        path = path_map.get(mode, LASER_TEMPLATES)
        tpls = _load_templates_from(path)
        if not tpls:
            print(f"[draw_answer_server] WARNING: no circle/square/triangle templates for mode '{mode}'")
            _RECOGNIZER_CACHE[mode] = None
            return None
        r = Recognizer(tpls)
        print(f"[draw_answer_server] Recognizer ({mode}): {len(tpls)} templates")
        _RECOGNIZER_CACHE[mode] = r
        return r

# ---------------------------------------------------------------------------
# laser detection (unchanged)
# ---------------------------------------------------------------------------
def detect_laser(blur, thresh: float):
    _, max_val, _, max_loc = cv2.minMaxLoc(blur)
    return (max_loc, max_val) if max_val >= thresh else (None, max_val)


# ---------------------------------------------------------------------------
# trail helpers
# ---------------------------------------------------------------------------
def draw_trail_canvas(
    canvas,
    stamped: deque,
    show_glow_outline: bool,
    now: float,
    trail_max_age: float,
) -> None:
    h, w = canvas.shape[:2]
    if show_glow_outline:
        overlay = canvas.copy()
        cv2.rectangle(overlay, (8, 8), (w - 8, h - 8), (100, 200, 255), 18)
        cv2.addWeighted(overlay, 0.22, canvas, 0.78, 0, canvas)
        cv2.rectangle(canvas, (8, 8), (w - 8, h - 8), (180, 255, 255), 2)

    xy = [(int(x), int(y)) for t, x, y in stamped if now - t <= trail_max_age]
    n = len(xy)
    if n < 2:
        return
    for i in range(1, n):
        alpha = i / float(max(1, n - 1))
        b = int(COLOR_TRAIL_TAIL[0] + alpha * (COLOR_TRAIL_HEAD[0] - COLOR_TRAIL_TAIL[0]))
        g = int(COLOR_TRAIL_TAIL[1] + alpha * (COLOR_TRAIL_HEAD[1] - COLOR_TRAIL_TAIL[1]))
        r = int(COLOR_TRAIL_TAIL[2] + alpha * (COLOR_TRAIL_HEAD[2] - COLOR_TRAIL_TAIL[2]))
        thickness = max(2, int(4 * alpha))
        cv2.line(canvas, xy[i - 1], xy[i], (b, g, r), thickness, cv2.LINE_AA)


def flash_trail_on_frame(frame, stamped: deque, matched: str):
    flash_colors = {
        "circle": (80, 220, 80),
        "square": (80, 180, 255),
        "triangle": (200, 180, 255),
    }
    col = flash_colors.get(matched, COLOR_TRAIL_HEAD)
    pts = [(int(x), int(y)) for _t, x, y in stamped]
    for i in range(1, len(pts)):
        cv2.line(frame, pts[i - 1], pts[i], col, 5, cv2.LINE_AA)


def resample_points(xy, n=48):
    """Resample a stroke to *n* equally-spaced points for more consistent $P matching."""
    if len(xy) < 2:
        return xy
    seg_lens = [((xy[i][0] - xy[i-1][0])**2 + (xy[i][1] - xy[i-1][1])**2) ** 0.5 for i in range(1, len(xy))]
    total = sum(seg_lens)
    if total == 0:
        return xy
    step = total / n
    out = [xy[0]]
    acc = 0.0
    si = 0
    for i in range(1, len(xy)):
        acc += seg_lens[i-1]
        while acc >= step and len(out) < n:
            excess = acc - step
            r = 1.0 - excess / seg_lens[i-1] if seg_lens[i-1] > 0 else 0.0
            out.append((int(xy[i-1][0] + r * (xy[i][0] - xy[i-1][0])),
                        int(xy[i-1][1] + r * (xy[i][1] - xy[i-1][1]))))
            acc -= step
    if len(out) < n and xy:
        out.append(xy[-1])
    return out[:n]


def recognize_trail(
    recognizer: Recognizer, xy_points: List[Tuple[int, int]], min_pts: int
) -> Tuple[Optional[str], float]:
    if len(xy_points) < min_pts:
        return None, 0.0
    pts = [Point(float(x), float(y), 0) for x, y in xy_points]
    try:
        name, score = recognizer.recognize(pts)
        cn = canonical_shape_name(str(name) if name is not None else "")
        return cn, float(score)
    except Exception as ex:
        print("[recognize_trail]", ex)
        return None, 0.0


# ---------------------------------------------------------------------------
# per-round runner — dispatches by mode
# ---------------------------------------------------------------------------
def run_round(params: Dict[str, Any]) -> Dict[str, Any]:
    rid = params.get("round_id", 0)
    mode = str(params.get("mode", "laser")).strip().lower()
    idle_gap = float(params.get("idle_gap_sec", 0.85))
    trail_max_age = float(params.get("trail_max_age_sec", 999.0))
    if trail_max_age <= 0:
        trail_max_age = 999.0
    min_score = float(params.get("min_score", 0.32))
    min_pts = int(params.get("gesture_min_pts", GESTURE_MIN_PTS))
    cam_index = int(params.get("camera_index", 0))
    thresh = float(params.get("thresh", MIN_BRIGHT_THRESH))
    max_round = params.get("max_round_sec", None)
    show_glow = bool(params.get("show_canvas_glow", True))
    debug_det = bool(params.get("debug_detection", False))

    deadline: Optional[float] = None
    if max_round is not None:
        try:
            deadline = time.time() + float(max_round)
        except (TypeError, ValueError):
            deadline = None

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        return {"round_id": rid, "ok": False, "shape": None, "score": 0.0, "reason": "no_camera"}

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    recognizer = get_recognizer(mode)
    if recognizer is None:
        cap.release()
        return {"round_id": rid, "ok": False, "shape": None, "score": 0.0,
                "reason": f"no_templates_for_mode_{mode}"}

    stamped: deque = deque(maxlen=TRAIL_MAX_POINTS)
    last_move_t = time.time()
    last_loc = None
    stroke_active = False

    outcome_reason = ""
    final_shape: Optional[str] = None
    final_score: float = 0.0

    win = f"Draw the Answer — {mode}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    dbg_win = "Detection debug"
    if debug_det:
        cv2.namedWindow(dbg_win, cv2.WINDOW_NORMAL)

    try:
        while True:
            now = time.time()
            ok, frame = cap.read()
            if not ok:
                outcome_reason = "frame_error"
                break

            frame = cv2.resize(frame, (FRAME_W, FRAME_H))
            frame = cv2.flip(frame, 1)   # mirror so right hand → right side
            loc = None

            if mode == "laser":
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (5, 5), 0)
                loc, _bright = detect_laser(blur, thresh)
            elif mode == "hand":
                loc = detect_hand_tip(frame)
            # object mode removed

            if loc:
                # Hand jitter filter: ignore sub‑pixel wobbles
                if mode == "hand" and last_loc and abs(loc[0] - last_loc[0]) + abs(loc[1] - last_loc[1]) < 4:
                    loc = last_loc
                stamped.append((now, loc[0], loc[1]))
                stroke_active = True
                if last_loc is None or (loc[0] != last_loc[0] or loc[1] != last_loc[1]):
                    last_move_t = now
                last_loc = loc
                cv2.circle(frame, loc, LASER_RADIUS + 3, (0, 180, 255), 2, cv2.LINE_AA)
                cv2.circle(frame, loc, LASER_RADIUS, COLOR_DOT, -1, cv2.LINE_AA)
            elif stroke_active:
                stroke_active = False
                xy_only = [(x, y) for t, x, y in stamped if now - t <= trail_max_age]
                if len(xy_only) >= min_pts:
                    xy_only = resample_points(xy_only, 48)
                    sh, sc = recognize_trail(recognizer, xy_only, min_pts)
                    if sh is not None and sc >= min_score:
                        outcome_reason = "recognized"
                        final_shape, final_score = sh, sc
                        snap = deque(stamped)
                        flash = frame.copy()
                        flash_trail_on_frame(flash, snap, sh)
                        cv2.imshow(win, flash)
                        cv2.waitKey(200)
                        break
                stamped.clear()
                last_loc = None
                continue

            gap = now - last_move_t
            if gap >= idle_gap and len(stamped) >= min_pts:
                xy_only = [(x, y) for t, x, y in stamped if now - t <= trail_max_age]
                if len(xy_only) >= min_pts:
                    xy_only = resample_points(xy_only, 48)
                    sh, sc = recognize_trail(recognizer, xy_only, min_pts)
                    if sh is not None and sc >= min_score:
                        outcome_reason = "recognized"
                        final_shape, final_score = sh, sc
                        snap = deque(stamped)
                        flash = frame.copy()
                        flash_trail_on_frame(flash, snap, sh)
                        cv2.imshow(win, flash)
                        cv2.waitKey(200)
                        break
                stamped.clear()

            trimmed = deque((t, x, y) for t, x, y in stamped if now - t <= trail_max_age)
            stamped.clear()
            stamped.extend(trimmed)

            canvas = frame.copy()
            draw_trail_canvas(canvas, stamped, show_glow, now, trail_max_age)
            prompt = f"Draw circle, square, or triangle (mode: {mode})"
            cv2.putText(
                canvas, prompt, (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (245, 245, 245), 1, cv2.LINE_AA,
            )
            cv2.imshow(win, canvas)

            if debug_det and mode == "laser":
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (5, 5), 0)
                dbg = cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)
                if loc:
                    cv2.circle(dbg, loc, 5, (0, 0, 255), 2)
                cv2.imshow(dbg_win, dbg)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                outcome_reason = "stopped"
                break

            if deadline is not None and time.time() >= deadline:
                final_shape, final_score = None, 0.0
                outcome_reason = "timeout"
                break
    finally:
        cap.release()
        try:
            cv2.destroyWindow(win)
        except Exception:
            pass
        if debug_det:
            try:
                cv2.destroyWindow(dbg_win)
            except Exception:
                pass

    if outcome_reason == "recognized":
        return {
            "round_id": rid, "ok": True, "shape": final_shape,
            "score": final_score, "reason": outcome_reason,
        }
    if outcome_reason == "unknown_shape":
        return {"round_id": rid, "ok": True, "shape": None, "score": 0.0, "reason": "unknown_shape"}
    oc = outcome_reason or "stopped"
    return {"round_id": rid, "ok": True, "shape": final_shape, "score": final_score, "reason": oc}


# ---------------------------------------------------------------------------
# TCP server
# ---------------------------------------------------------------------------
class _Handler(socketserver.StreamRequestHandler):
    def handle(self):
        print("[draw_answer_server] Client connected:", self.client_address)
        fh = self.rfile
        try:
            while True:
                line = fh.readline()
                if not line:
                    break
                try:
                    text = line.decode("utf-8").strip()
                except Exception:
                    continue
                if not text:
                    continue
                try:
                    msg = json.loads(text)
                except json.JSONDecodeError:
                    resp = {"ok": False, "error": "bad_json"}
                    self.wfile.write((json.dumps(resp) + "\n").encode("utf-8"))
                    self.wfile.flush()
                    continue

                cmd = msg.get("cmd") or ""
                if cmd == "PING":
                    self.wfile.write((json.dumps({"ok": True, "ping": True}) + "\n").encode("utf-8"))
                    self.wfile.flush()
                    continue

                if cmd == "ROUND":
                    out = run_round(msg if isinstance(msg, dict) else {})
                    self.wfile.write((json.dumps(out) + "\n").encode("utf-8"))
                    self.wfile.flush()
                    continue

                resp = {"ok": False, "error": "unknown_cmd"}
                self.wfile.write((json.dumps(resp) + "\n").encode("utf-8"))
                self.wfile.flush()
        except (ConnectionResetError, BrokenPipeError):
            pass
        print("[draw_answer_server] Client disconnected")


class ThreadingServer(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    srv = ThreadingServer((HOST, PORT), _Handler)
    print(f"[draw_answer_server] Listening on tcp://{HOST}:{PORT}")
    print("[draw_answer_server] Modes: laser / hand")
    print("[draw_answer_server] Press Ctrl+C to stop.")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[draw_answer_server] Shutdown.")
        srv.shutdown()


if __name__ == "__main__":
    main()
