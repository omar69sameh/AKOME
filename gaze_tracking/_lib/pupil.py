import cv2
import numpy as np


class Pupil:
    """
    This class detects the iris of an eye and estimates
    the position of the pupil
    """

    def __init__(self, eye_frame, threshold):
        self.iris_frame = None
        self.threshold = threshold
        self.x = None
        self.y = None

        self.detect_iris(eye_frame)

    @staticmethod
    def image_processing(eye_frame, threshold):
        """Performs operations on the eye frame to isolate the iris

        Arguments:
            eye_frame (numpy.ndarray): Frame containing an eye and nothing else
            threshold (int): Threshold value used to binarize the eye frame

        Returns:
            A frame with a single element representing the iris
        """
        kernel = np.ones((3, 3), np.uint8)
        new_frame = cv2.bilateralFilter(eye_frame, 10, 15, 15)
        new_frame = cv2.erode(new_frame, kernel, iterations=3)
        new_frame = cv2.threshold(new_frame, threshold, 255, cv2.THRESH_BINARY)[1]

        return new_frame

    @staticmethod
    def _refine_subpixel(eye_frame, x, y, radius=6):
        """Refine the integer-pixel pupil hit to subpixel precision using
        an intensity-weighted centroid over a small neighbourhood.

        The contour-moments centroid quantises to whole pixels — on a
        ~25 px eye crop that's ~4 % of the eye width per axis, which shows
        up as visible dot jitter and bleeds into the gaze ratio. Weighting
        by darkness (pupil = darkest region) gives ~0.1 px precision and
        is cheap (single 13x13 patch sum)."""
        h, w = eye_frame.shape[:2]
        x0 = max(0, x - radius); y0 = max(0, y - radius)
        x1 = min(w, x + radius + 1); y1 = min(h, y + radius + 1)
        patch = eye_frame[y0:y1, x0:x1]
        if patch.size == 0:
            return float(x), float(y)
        # Pupil is the DARK region. Invert so darkness becomes weight.
        # Subtract the local mean so only the *darker than surrounding*
        # pixels contribute — kills the bias from generally-dark skin.
        weight = patch.astype(np.float32)
        local_mean = float(weight.mean())
        weight = np.maximum(local_mean - weight, 0.0)
        total = float(weight.sum())
        if total < 1e-3:
            return float(x), float(y)
        yy, xx = np.indices(patch.shape, dtype=np.float32)
        cx = float((weight * xx).sum() / total) + x0
        cy = float((weight * yy).sum() / total) + y0
        return cx, cy

    def detect_iris(self, eye_frame):
        """Detects the iris and estimates the position of the pupil by
        calculating the centroid (with subpixel refinement).

        Arguments:
            eye_frame (numpy.ndarray): Frame containing an eye and nothing else
        """
        self.iris_frame = self.image_processing(eye_frame, self.threshold)

        contours, _ = cv2.findContours(self.iris_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]
        contours = sorted(contours, key=cv2.contourArea)

        try:
            moments = cv2.moments(contours[-2])
            cx_int = int(moments['m10'] / moments['m00'])
            cy_int = int(moments['m01'] / moments['m00'])
            # Subpixel refinement on the original (non-binarised) eye frame —
            # uses real intensities, not the post-threshold blob.
            self.x, self.y = self._refine_subpixel(eye_frame, cx_int, cy_int)
        except (IndexError, ZeroDivisionError):
            pass
