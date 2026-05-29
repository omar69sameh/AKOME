import face_recognition
import cv2
import numpy as np
import os
import time
import dlib
import socket
import threading
import tempfile
from concurrent.futures import ThreadPoolExecutor

STATUS_FILE = os.path.join(tempfile.gettempdir(), "akome_face.txt")

clients = []
clients_lock = threading.Lock()
camera_event = threading.Event()
current_frame = None
current_frame_lock = threading.Lock()

def handle_client(client_socket, addr):
    with clients_lock:
        clients.append(client_socket)
    camera_event.set()
    print(f"[INFO] Client connected ({addr})")
    try:
        buffer = ""
        while True:
            data = client_socket.recv(1024)
            if not data:
                break
            buffer += data.decode('utf-8', errors='ignore')
            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                line = line.strip()
                if line.startswith("SAVE|"):
                    name = line.split("|", 1)[1]
                    saved = False
                    with current_frame_lock:
                        global current_frame
                        if current_frame is not None:
                            os.makedirs("people", exist_ok=True)
                            path = os.path.join("people", f"{name}.jpg")
                            cv2.imwrite(path, current_frame)
                            saved = add_person_from_file(path, name)
                    if saved:
                        try:
                            client_socket.sendall(f"FACE:SAVED|{name}\n".encode('utf-8'))
                        except: pass
                    else:
                        try:
                            client_socket.sendall(f"FACE:ERROR|Failed to encode face\n".encode('utf-8'))
                        except: pass
    except Exception as e:
        print(f"[ERROR] Client handling error: {e}")
    finally:
        client_socket.close()
        with clients_lock:
            if client_socket in clients:
                clients.remove(client_socket)
            if not clients:
                camera_event.clear()
        print(f"[INFO] Client disconnected ({addr})")

def socket_server_thread():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        server.bind(('127.0.0.1', 14000))
        server.listen(5)
        print("[INFO] Face Socket Server on port 14000")
        while True:
            client_socket, addr = server.accept()
            threading.Thread(target=handle_client, args=(client_socket, addr), daemon=True).start()
    except Exception as e:
        print("[ERROR] Socket server error:", e)

def broadcast_face(is_recognized: bool, name: str):
    msg = f"FACE:{'True' if is_recognized else 'False'}|{name}\n"
    try:
        with open(STATUS_FILE, 'w') as f:
            f.write(msg.strip())
    except:
        pass
    print(f"[BROADCAST] {msg.strip()}")
    with clients_lock:
        snapshot = list(clients)
    to_remove = []
    for c in snapshot:
        try:
            c.sendall(msg.encode('utf-8'))
        except:
            to_remove.append(c)
    if to_remove:
        with clients_lock:
            for c in to_remove:
                if c in clients:
                    clients.remove(c)
                    try:
                        c.close()
                    except:
                        pass
            if not clients:
                camera_event.clear()

known_face_encodings = []
known_face_names = []

CAMERA_WIDTH = 960
CAMERA_HEIGHT = 540
CAMERA_TARGET_FPS = 30
DETECTION_SCALE = 0.20
DETECTION_INTERVAL_SEC = 0.10
GPU_AVAILABLE = bool(getattr(dlib, "DLIB_USE_CUDA", False))


def add_person_from_file(image_path: str, name: str) -> bool:
    if not os.path.exists(image_path):
        return False
    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)
    if len(encodings) == 0:
        return False
    known_face_encodings.append(encodings[0])
    known_face_names.append(name)
    return True


def load_people_folder(folder: str = "people"):
    supported = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if not os.path.isdir(folder):
        return
    for filename in sorted(os.listdir(folder)):
        name, ext = os.path.splitext(filename)
        if ext.lower() in supported:
            try:
                add_person_from_file(os.path.join(folder, filename), name)
            except:
                pass


def identify_faces(rgb_small_frame: np.ndarray):
    model = "cnn" if GPU_AVAILABLE else "hog"
    face_locations = face_recognition.face_locations(rgb_small_frame, model=model)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    face_names = []
    for face_encoding in face_encodings:
        name = "Unknown"
        if known_face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
        face_names.append(name)
    return face_locations, face_names


def draw_results(frame: np.ndarray, face_locations: list, face_names: list, scale: float):
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top = int(top / scale)
        right = int(right / scale)
        bottom = int(bottom / scale)
        left = int(left / scale)
        color = (0, 200, 0) if name != "Unknown" else (0, 0, 220)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)


def draw_hud(frame: np.ndarray, face_count: int, fps: float):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 40), (0, 0, 0), cv2.FILLED)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    info = f"FPS: {fps:.1f}  |  Known: {len(known_face_names)}  |  Detected: {face_count}  |  [Q] Quit"
    cv2.putText(frame, info, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)


def run_camera_loop():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_TARGET_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    window_name = "Face Recognition  [Q=Quit]"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 540)
    face_locations = []
    face_names = []
    last_fps_tick = time.time()
    frame_counter = 0
    fps = 0.0
    last_detection_submit = 0.0
    last_face_broadcast = 0.0
    detect_executor = ThreadPoolExecutor(max_workers=1)
    detect_future = None
    while camera_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        
        global current_frame
        with current_frame_lock:
            current_frame = frame.copy()
            
        now = time.time()
        if detect_future is None and (now - last_detection_submit) >= DETECTION_INTERVAL_SEC:
            small_frame = cv2.resize(frame, (0, 0), fx=DETECTION_SCALE, fy=DETECTION_SCALE, interpolation=cv2.INTER_LINEAR)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            detect_future = detect_executor.submit(identify_faces, rgb_small_frame)
            last_detection_submit = now
        if detect_future is not None and detect_future.done():
            try:
                face_locations, face_names = detect_future.result()
            except:
                face_locations, face_names = [], []
            detect_future = None
        frame_counter += 1
        elapsed = now - last_fps_tick
        if elapsed >= 1.0:
            fps = frame_counter / elapsed
            frame_counter = 0
            last_fps_tick = now
        if now - last_face_broadcast >= 1.0:
            if len(face_names) == 0:
                broadcast_face(False, "NoFace")
            else:
                recognized_name = "Unknown"
                is_recognized = False
                for name in face_names:
                    if name != "Unknown":
                        recognized_name = name
                        is_recognized = True
                        break
                broadcast_face(is_recognized, recognized_name)
            last_face_broadcast = now
        draw_results(frame, face_locations, face_names, DETECTION_SCALE)
        draw_hud(frame, len(face_locations), fps)
        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            camera_event.clear()
            break
    detect_executor.shutdown(wait=False)
    cap.release()
    cv2.destroyAllWindows()


def main():
    load_people_folder("people")
    print(f"[INFO] {len(known_face_names)} people loaded.")
    detector_name = "CNN (GPU)" if GPU_AVAILABLE else "HOG (CPU)"
    print(f"[INFO] Detection model: {detector_name}")
    threading.Thread(target=socket_server_thread, daemon=True).start()
    print("[INFO] Waiting for client connection...")
    while True:
        camera_event.wait()
        run_camera_loop()
        time.sleep(0.5)

if __name__ == "__main__":
    main()
