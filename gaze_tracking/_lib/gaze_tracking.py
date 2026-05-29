from pathlib import Path

import cv2
import dlib

from .calibration import Calibration
from .eye import Eye


class GazeTracking:
    """
    This class tracks the user's gaze.
    It provides useful information like the position of the eyes
    and pupils and lets you know if the eyes are open or closed.
    """

    def __init__(self):
        self.frame = None
        self.eye_left = None
        self.eye_right = None
        self.calibration = Calibration()

        # _face_detector is used to detect faces
        self._face_detector = dlib.get_frontal_face_detector()

        # _predictor is used to get facial landmarks of a given face
        model_path = Path(__file__).parent / "trained_models" / "shape_predictor_68_face_landmarks.dat"
        self._predictor = dlib.shape_predictor(str(model_path))

    @property
    def pupils_located(self):
        """Check that the pupils have been located"""
        try:
            int(self.eye_left.pupil.x)
            int(self.eye_left.pupil.y)
            int(self.eye_right.pupil.x)
            int(self.eye_right.pupil.y)
            return True
        except (AttributeError, TypeError):
            return False

    def _analyze(self):
        """Detects the face and initializes Eye objects.

        Performance: a user-facing webcam face is large in the frame, so we
        try dlib's HOG detector at upsample=0 first (~10x faster than
        upsample=1) and only fall back to upsampling if no face is found —
        e.g. user is far from the camera. This single change typically
        takes per-frame detection from ~400 ms to ~40 ms on a laptop.

        We also cache the most recent detection and reuse it for a few
        frames between full detections — landmark prediction still runs
        every frame so the eye/pupil tracking stays responsive.
        """
        frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

        face = None
        # Reuse the cached face rect for a few frames (skip HOG entirely).
        # _det_skip counter resets to _DET_EVERY whenever we re-detect.
        if (getattr(self, "_cached_face", None) is not None
                and getattr(self, "_det_skip", 0) > 0):
            self._det_skip -= 1
            face = self._cached_face
        else:
            faces = self._face_detector(frame, 0)   # fast: no upsample
            if len(faces) == 0:
                # Fallback for far-away / small faces only.
                faces = self._face_detector(frame, 1)
            if len(faces) == 0:
                self._cached_face = None
                self.eye_left = None
                self.eye_right = None
                return
            face = max(faces, key=lambda r: r.width() * r.height())
            self._cached_face = face
            self._det_skip = 4   # reuse for the next 4 frames, then redetect

        landmarks = self._predictor(frame, face)
        self.eye_left = Eye(frame, landmarks, 0, self.calibration)
        self.eye_right = Eye(frame, landmarks, 1, self.calibration)

    def refresh(self, frame):
        """Refreshes the frame and analyzes it.

        Arguments:
            frame (numpy.ndarray): The frame to analyze
        """
        self.frame = frame
        self._analyze()

    def pupil_left_coords(self):
        """Returns the coordinates of the left pupil (integer pixels for drawing).
        The underlying pupil.x/y are now float (subpixel-refined); cast to int
        here so cv2 drawing primitives accept them."""
        if self.pupils_located:
            x = int(self.eye_left.origin[0] + self.eye_left.pupil.x)
            y = int(self.eye_left.origin[1] + self.eye_left.pupil.y)
            return (x, y)

    def pupil_right_coords(self):
        """Returns the coordinates of the right pupil (integer pixels for drawing).
        See pupil_left_coords for the float-to-int rationale."""
        if self.pupils_located:
            x = int(self.eye_right.origin[0] + self.eye_right.pupil.x)
            y = int(self.eye_right.origin[1] + self.eye_right.pupil.y)
            return (x, y)

    def horizontal_ratio(self):
        """Returns a number between 0.0 and 1.0 that indicates the
        horizontal direction of the gaze. The extreme right is 0.0,
        the center is 0.5 and the extreme left is 1.0
        """
        if self.pupils_located:
            pupil_left = self.eye_left.pupil.x / (self.eye_left.center[0] * 2 - 10)
            pupil_right = self.eye_right.pupil.x / (self.eye_right.center[0] * 2 - 10)
            return (pupil_left + pupil_right) / 2

    def vertical_ratio(self):
        """Returns a number between 0.0 and 1.0 that indicates the
        vertical direction of the gaze. The extreme top is 0.0,
        the center is 0.5 and the extreme bottom is 1.0
        """
        if self.pupils_located:
            pupil_left = self.eye_left.pupil.y / (self.eye_left.center[1] * 2 - 10)
            pupil_right = self.eye_right.pupil.y / (self.eye_right.center[1] * 2 - 10)
            return (pupil_left + pupil_right) / 2

    def is_right(self):
        """Returns true if the user is looking to the right"""
        if self.pupils_located:
            return self.horizontal_ratio() <= 0.35

    def is_left(self):
        """Returns true if the user is looking to the left"""
        if self.pupils_located:
            return self.horizontal_ratio() >= 0.65

    def is_center(self):
        """Returns true if the user is looking at the center"""
        if self.pupils_located:
            return self.is_right() is not True and self.is_left() is not True

    def is_blinking(self):
        """Returns true if the user closes his eyes"""
        if self.pupils_located:
            blinking_ratio = (self.eye_left.blinking + self.eye_right.blinking) / 2
            return blinking_ratio > 3.8

    def annotated_frame(self):
        """Returns the main frame with pupils highlighted"""
        frame = self.frame.copy()

        if self.pupils_located:
            color = (0, 255, 0)
            x_left, y_left = self.pupil_left_coords()
            x_right, y_right = self.pupil_right_coords()
            cv2.line(frame, (x_left - 5, y_left), (x_left + 5, y_left), color)
            cv2.line(frame, (x_left, y_left - 5), (x_left, y_left + 5), color)
            cv2.line(frame, (x_right - 5, y_right), (x_right + 5, y_right), color)
            cv2.line(frame, (x_right, y_right - 5), (x_right, y_right + 5), color)

        return frame
