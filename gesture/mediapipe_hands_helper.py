import os
import cv2
import mediapipe as mp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "hand_landmarker.task")

def create_hand_landmarker():
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    return HandLandmarker.create_from_options(
        HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )

def detect_hands(landmarker, frame_bgr, timestamp_ms=0):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)
    if not result.hand_landmarks:
        return None
    return result.hand_landmarks[0]

def draw_hands_on_bgr(frame_bgr, hand_landmarks):
    if not hand_landmarks:
        return
    h, w, _ = frame_bgr.shape
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

    connections = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (5,9),(9,10),(10,11),(11,12),
        (9,13),(13,14),(14,15),(15,16),
        (13,17),(17,18),(18,19),(19,20),
        (0,17),
    ]

    for a, b in connections:
        if a < len(pts) and b < len(pts):
            cv2.line(frame_bgr, pts[a], pts[b], (0, 255, 0), 2)

    for i, (px, py) in enumerate(pts):
        color = (0, 0, 255) if i in (4, 8, 12, 16, 20) else (255, 0, 0)
        cv2.circle(frame_bgr, (px, py), 4, color, cv2.FILLED)
