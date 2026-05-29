import argparse
import time
import socket
import threading
from pathlib import Path

import cv2
from ultralytics import YOLO

OBJECT_PORT = 5003
_object_clients = []
_object_clients_lock = threading.Lock()
_last_letter = None

def broadcast_letter(letter):
    global _last_letter
    if letter == _last_letter:
        return
    _last_letter = letter
    msg = f"LETTER:{letter}\n"
    with _object_clients_lock:
        dead = []
        for c in _object_clients:
            try:
                c.sendall(msg.encode('utf-8'))
            except:
                dead.append(c)
        for c in dead:
            try:
                c.close()
            except:
                pass
            _object_clients.remove(c)

def socket_server():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(('127.0.0.1', OBJECT_PORT))
    srv.listen(5)
    print(f"[SOCKET] Object detection server on port {OBJECT_PORT}")
    while True:
        try:
            client, addr = srv.accept()
            with _object_clients_lock:
                _object_clients.append(client)
            print(f"[SOCKET] Client connected: {addr}")
        except:
            break

DEFAULT_WEIGHTS = "runs/gestures_run/weights/best.pt"
DEFAULT_SOURCE  = 0
DEFAULT_CONF    = 0.35
DEFAULT_IOU     = 0.45
DEFAULT_IMGSZ   = 640
DEFAULT_DEVICE  = ""

CLASS_NAMES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'N', 'R', 'S', 'T', 'Y', 'Back', 'Confirm', 'Down', 'LeftSwipe', 'RightSwipe', 'Undo', 'Up']

CLASS_COLORS = [
    (220,  50,  50),
    ( 50, 220,  50),
    ( 50,  50, 220),
    (220, 180,  50),
    ( 50, 220, 220),
    (220,  50, 220),
    (150, 255, 100),
    (255, 150,  50),
    (100, 100, 255),
    (  0, 200, 200),
    (200,   0, 200),
    (255, 200,   0),
    (  0, 180, 255),
]

def parse_args():
    p = argparse.ArgumentParser(description="YOLOv11 live detection with OpenCV")
    p.add_argument("--weights", default=DEFAULT_WEIGHTS)
    p.add_argument("--source",  default=DEFAULT_SOURCE)
    p.add_argument("--conf",    default=DEFAULT_CONF,    type=float)
    p.add_argument("--iou",     default=DEFAULT_IOU,     type=float)
    p.add_argument("--imgsz",   default=DEFAULT_IMGSZ,   type=int)
    p.add_argument("--device",  default=DEFAULT_DEVICE)
    p.add_argument("--save",    action="store_true")
    p.add_argument("--no-labels", action="store_true")
    return p.parse_args()

def draw_detections(frame, results, show_labels=True):
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color = CLASS_COLORS[cls_id % len(CLASS_COLORS)]
        name  = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        if show_labels:
            label     = f"{name}  {conf:.2f}"
            font      = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.65
            thickness  = 2
            (lw, lh), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            label_y1 = max(y1 - lh - baseline - 4, 0)
            label_y2 = y1
            cv2.rectangle(frame, (x1, label_y1), (x1 + lw + 4, label_y2), color, -1)
            text_color = (255, 255, 255)
            cv2.putText(frame, label, (x1 + 2, y1 - baseline - 2),
                        font, font_scale, text_color, thickness, cv2.LINE_AA)
    return frame

def draw_hud(frame, fps, det_count):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (200, 55), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    cv2.putText(frame, f"FPS: {fps:.1f}",         (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Detections: {det_count}", (10, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2, cv2.LINE_AA)
    return frame

def main():
    args = parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: '{weights_path}'")

    print(f"[INFO] Loading model from: {weights_path}")
    model = YOLO(str(weights_path))
    print(f"[INFO] Model loaded. Classes: {CLASS_NAMES}")

    threading.Thread(target=socket_server, daemon=True).start()

    source = args.source
    try:
        source = int(source)
    except (ValueError, TypeError):
        pass

    print(f"[INFO] Opening source: {'webcam' if isinstance(source, int) else source}")
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"[INFO] Source resolution: {frame_w}x{frame_h}  FPS: {fps_src:.1f}")

    writer = None
    if args.save:
        out_path = "output.mp4"
        fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
        writer   = cv2.VideoWriter(out_path, fourcc, fps_src, (frame_w, frame_h))
        print(f"[INFO] Saving output to: {out_path}")

    print("\n[INFO] Detection running. Press Q to quit.\n")
    prev_time = time.time()
    frame_count = 0
    INFERENCE_EVERY_N_FRAMES = 3
    last_results = None
    last_det_count = 0
    last_top_letter = "NONE"

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of stream.")
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        if max(h, w) > args.imgsz:
            scale = args.imgsz / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

        frame_count += 1
        if frame_count % INFERENCE_EVERY_N_FRAMES == 0:
            results = model.predict(
                source  = frame,
                conf    = args.conf,
                iou     = args.iou,
                imgsz   = args.imgsz,
                device  = args.device,
                verbose = False,
            )
            last_results = results
            det_count = len(results[0].boxes)
            last_det_count = det_count

            if det_count > 0:
                last_top_letter = CLASS_NAMES[int(results[0].boxes.cls[0])]
            else:
                last_top_letter = "NONE"
            broadcast_letter(last_top_letter)

        if last_results is not None:
            frame = draw_detections(frame, last_results, show_labels=not args.no_labels)

        now      = time.time()
        fps_live = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        frame = draw_hud(frame, fps_live, last_det_count)

        cv2.imshow("YOLOv11 Webcam", frame)

        if writer is not None:
            writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Quit signal received.")
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    print("[INFO] Done.")

if __name__ == "__main__":
    main()
