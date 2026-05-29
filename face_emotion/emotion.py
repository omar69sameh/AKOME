import cv2, os, numpy as np
import mediapipe as mp
import time
import socket
import threading

clients = []
clients_lock = threading.Lock()

def socket_server_thread():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        server.bind(('127.0.0.1', 13000))
        server.listen(5)
        print("Emotion Socket Server listening on port 13000")
        while True:
            client_socket, addr = server.accept()
            with clients_lock:
                clients.append(client_socket)
    except Exception as e:
        print("Socket server error:", e)

def broadcast_emotion(emotion):
    msg = f"EMOTION:{emotion}\n"
    print(f"SENDING TO {len(clients)} CLIENTS: {emotion}")
    with clients_lock:
        to_remove = []
        for c in clients:
            try:
                c.sendall(msg.encode('utf-8'))
            except:
                to_remove.append(c)
        for c in to_remove:
            clients.remove(c)
            try:
                c.close()
            except:
                pass

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "face_landmarker.task")

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1, min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5, min_tracking_confidence=0.5,
    output_face_blendshapes=True)

EMOTIONS = ["HAPPY", "CONFUSED", "NEUTRAL"]
COLOURS  = {"HAPPY": (0, 210, 255), "CONFUSED": (0, 140, 255), "NEUTRAL": (160, 160, 160)}
SMOOTH = 0.75 # Lowered for faster, more lifelike responsiveness

def draw_bars(panel, bars, emotion):
    panel[:] = (20, 20, 25) # Dark minimal background
    ph = panel.shape[0]
    
    # Simple Title / Current Emotion
    col = COLOURS.get(emotion, (255,255,255))
    cv2.putText(panel, emotion, (20, ph // 2 - 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 2, cv2.LINE_AA)

    # Just the bars
    bar_w, bar_h, gap, y0 = 150, 20, 50, ph // 2 - 20
    for i, em in enumerate(EMOTIONS):
        y = y0 + i * gap
        pct = bars.get(em, 0.0)

        # Label
        cv2.putText(panel, em, (20, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOURS[em], 1, cv2.LINE_AA)
        
        # Background Bar
        cv2.rectangle(panel, (20, y), (20 + bar_w, y + bar_h), (45, 45, 50), -1)
        
        # Foreground Bar
        if pct > 0.01:
            cv2.rectangle(panel, (20, y), (20 + int(bar_w * pct), y + bar_h), COLOURS[em], -1)
            
        # Percentage text
        cv2.putText(panel, f"{int(pct*100)}%", (20 + bar_w + 10, y + 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1, cv2.LINE_AA)

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("ERROR: Cannot open camera"); return

    bars = {em: 0.0 for em in EMOTIONS}
    ts = 0
    cv2.namedWindow("Emotion Bars", cv2.WINDOW_AUTOSIZE)

    threading.Thread(target=socket_server_thread, daemon=True).start()
    
    current_stable_emotion = "NEUTRAL"
    stable_start_time = time.time()
    last_sent_time = 0.0

    with FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok: break
            
            frame = cv2.flip(frame, 1)
            
            # Resize frame to a smaller, more compact size (height 360)
            target_h = 360
            h, w = frame.shape[:2]
            scale = target_h / float(h)
            frame = cv2.resize(frame, (int(w * scale), target_h))
            
            # Yield CPU
            time.sleep(0.01)
            
            fh, fw = frame.shape[:2]
            
            # Create a simple panel just for the bars (smaller width: 260)
            panel = np.zeros((fh, 260, 3), dtype=np.uint8)

            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                              data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = landmarker.detect_for_video(mp_img, ts)
            ts += 33

            emotion = "NEUTRAL"
            target = {em: 0.0 for em in EMOTIONS}

            if result.face_blendshapes:
                blendshapes = result.face_blendshapes[0]
                bs = {b.category_name: b.score for b in blendshapes}
                
                # Happy: Glasses can obscure cheeks, so we rely heavily (95%) on the mouth.
                mouth_smile = (bs.get('mouthSmileLeft', 0) + bs.get('mouthSmileRight', 0)) / 2.0
                cheek_squint = (bs.get('cheekSquintLeft', 0) + bs.get('cheekSquintRight', 0)) / 2.0
                happy_score = (mouth_smile * 0.95) + (cheek_squint * 0.05)
                
                # Confused: Glasses obscure eye squints and outer brows. 
                # We boost sensitivity on the inner brows and center frowns which sit visibly above the nose bridge.
                brow_frown = (bs.get('browDownLeft', 0) + bs.get('browDownRight', 0)) / 2.0
                brow_up = bs.get('browInnerUp', 0)
                confused_score = max(brow_frown * 1.2, brow_up * 1.2)
                                  
                # Realistic non-linear mapping (exponents suppress noise and organically ramp up)
                target["HAPPY"] = min(1.0, max(0.0, (happy_score * 1.5) ** 1.5))
                target["CONFUSED"] = min(1.0, max(0.0, (confused_score * 1.5) ** 1.5))
                
                # Neutral organically fills the void left by other emotions
                target["NEUTRAL"] = max(0.0, 1.0 - (target["HAPPY"] + target["CONFUSED"]))

            # Smooth the bars
            for em in EMOTIONS:
                bars[em] = bars[em]*SMOOTH + target.get(em, 0.0)*(1-SMOOTH)

            if result.face_blendshapes:
                # Determine dominant emotion from smoothed bars
                if bars["HAPPY"] > 0.35 and bars["HAPPY"] > bars["CONFUSED"]:
                    emotion = "HAPPY"
                elif bars["CONFUSED"] > 0.35:
                    emotion = "CONFUSED"
                else:
                    emotion = "NEUTRAL"
            else:
                emotion = "NO FACE"

            # Send every 1 second
            if time.time() - last_sent_time >= 1.0:
                broadcast_emotion(emotion)
                last_sent_time = time.time()

            # Draw the simple design
            draw_bars(panel, bars, emotion)
            
            # Show camera beside the bars
            combined = cv2.hconcat([frame, panel])
            cv2.imshow("Emotion Bars", combined)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
