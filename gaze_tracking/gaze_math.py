"""Shared gaze estimation helpers for gaze_tracker and split_screen_gaze."""

import math
import time as _time
from collections import deque
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np


# ─── One-Euro filter ────────────────────────────────────────────────────────
# Casiez et al. CHI 2012, "1€ Filter: A Simple Speed-Based Low-Pass Filter".
# The principled replacement for a fixed-alpha EMA + saccade snap:
#   * When the eye is still → high smoothing (low cutoff) kills jitter.
#   * When the eye saccades → low smoothing (high cutoff) gives near-zero
#     latency, no snap-to-overshoot.
# Tuning for gaze (30 fps, ratio units):
#   mincutoff ≈ 0.8 Hz → ~250 ms half-life at rest (kills sub-pixel jitter)
#   beta      ≈ 0.5    → derivative ramp → ~30 ms half-life during a saccade
#   dcutoff   ≈ 1.0 Hz → smoothing on the derivative estimate itself

class OneEuroFilter:
    def __init__(self, mincutoff: float = 0.8, beta: float = 0.5,
                 dcutoff: float = 1.0) -> None:
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self._x_prev: Optional[float] = None
        self._dx_prev: float = 0.0
        self._t_prev: Optional[float] = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self, x: Optional[float] = None) -> None:
        self._x_prev = x
        self._dx_prev = 0.0
        self._t_prev = None

    def __call__(self, x: float, t: Optional[float] = None) -> float:
        if t is None:
            t = _time.perf_counter()
        if self._x_prev is None or self._t_prev is None:
            self._x_prev = x
            self._t_prev = t
            return x
        dt = t - self._t_prev
        if dt <= 1e-4:
            dt = 1e-3
        self._t_prev = t
        # Estimate derivative + smooth it
        dx = (x - self._x_prev) / dt
        a_d = self._alpha(self.dcutoff, dt)
        dx_smooth = a_d * dx + (1.0 - a_d) * self._dx_prev
        self._dx_prev = dx_smooth
        # Cutoff grows with speed → less smoothing when moving fast
        cutoff = self.mincutoff + self.beta * abs(dx_smooth)
        a = self._alpha(cutoff, dt)
        x_smooth = a * x + (1.0 - a) * self._x_prev
        self._x_prev = x_smooth
        return x_smooth

# Symmetric bands around anchor (learned per session, tightly capped).
# Polarity matches the GazeTracking convention documented in gaze_tracker.py:
#   pupil moves UP   -> raw_v decreases -> delta NEGATIVE -> question (top)
#   pupil moves DOWN -> raw_v increases -> delta POSITIVE -> answer  (bottom)
QUESTION_DELTA = 0.011
ANSWER_DELTA = -0.011
DEAD_BAND = 0.006

# Filter constants tuned for ~30 fps webcam input.
# Combined response time target: < 150 ms for a deliberate top<->bottom saccade.
EMA_ALPHA = 0.40              # was 0.12 — responsive but still smooths jitter
MEDIAN_WINDOW = 5             # was 9 — halves the median lag
EYE_AGREE_MAX = 0.22   # was 0.14 — too strict at extreme corners where one
                       # eye is more occluded than the other; samples were
                       # being silently dropped when looking at the screen
                       # edges, esp. when paired with a downward gaze.

# Large jump in the median (== majority of buffer agrees on the new
# position) indicates a real saccade. Snap the EMA to follow.
# Replaces the old MAX_FRAME_JUMP gate which REJECTED jumps > 0.10 and
# permanently froze the EMA on big eye movements — the root cause of
# "answer area never registered" dead zones.
SACCADE_SNAP_THRESHOLD = 0.05

FRAMES_TO_ENTER = 2           # was 3
FRAMES_TO_LEAVE = 2           # was 5 — much faster top<->bottom transitions


# Module-level CLAHE — was being recreated EVERY frame in the old code,
# which is a constant alloc + setup cost we pay 30 times per second.
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    enhanced = _CLAHE.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def pupil_vertical_raw(eye) -> float:
    pupil_abs_y = eye.origin[1] + eye.pupil.y
    pts = eye.landmark_points
    upper = (pts[1][1] + pts[2][1]) / 2.0
    lower = (pts[4][1] + pts[5][1]) / 2.0
    opening = lower - upper
    if opening <= 0:
        return 0.5
    return (pupil_abs_y - upper) / opening


def pupil_horizontal_ratio(eye) -> float:
    pupil_abs_x = eye.origin[0] + eye.pupil.x
    pts = eye.landmark_points
    left_x = pts[0][0]
    right_x = pts[3][0]
    width = right_x - left_x
    if width <= 0:
        return 0.5
    return (pupil_abs_x - left_x) / width


def median(values: List[float]) -> float:
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0


def _stdev(values: List[float]) -> float:
    if len(values) < 2:
        return 0.012
    m = sum(values) / len(values)
    var = sum((x - m) ** 2 for x in values) / len(values)
    return max(var ** 0.5, 0.006)


def _trimmed_samples(samples: List[float]) -> List[float]:
    if len(samples) < 10:
        return samples
    s = sorted(samples)
    lo = len(s) // 5
    hi = len(s) - lo
    return s[lo:hi]


class _LandmarkSmoother:
    """Per-eye low-pass on the 6 dlib landmark points.

    dlib's 68-point predictor produces points that jitter by 1-2 px per
    frame even on a perfectly still face, and that jitter feeds directly
    into the denominators of pupil_vertical_raw / pupil_horizontal_ratio.
    A light EMA (alpha 0.5) halves the jitter without introducing
    noticeable head-tracking lag. We reset the smoother whenever the
    face shifts by more than half its width so head movements don't
    smear."""
    def __init__(self, alpha: float = 0.5) -> None:
        self.alpha = alpha
        self._prev: Optional[np.ndarray] = None

    def smooth(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float32)
        if self._prev is None or self._prev.shape != pts.shape:
            self._prev = pts.copy()
            return self._prev
        # Reset on large head motion (anything > half the eye width).
        eye_w = max(1.0, float(pts[3, 0] - pts[0, 0]))
        if np.max(np.abs(pts - self._prev)) > eye_w * 0.5:
            self._prev = pts.copy()
            return self._prev
        self._prev = self.alpha * pts + (1.0 - self.alpha) * self._prev
        return self._prev


# One smoother per eye side. raw_gaze_from_tracker swaps in the smoothed
# landmark points before computing pupil ratios.
_LM_SMOOTH_L = _LandmarkSmoother()
_LM_SMOOTH_R = _LandmarkSmoother()


def _eye_opening_score(eye) -> float:
    """Eye-aspect-ratio-style score in [0, 1].
       ~0 = fully closed (blink), ~1 = wide open.
    Used both as a blink detector and as a quality weight when fusing
    the two eyes — a half-closed eye contributes a worse pupil estimate
    than a wide-open one."""
    try:
        pts = eye.landmark_points
        width = float(pts[3][0] - pts[0][0])
        if width <= 0:
            return 0.0
        upper = (pts[1][1] + pts[2][1]) / 2.0
        lower = (pts[4][1] + pts[5][1]) / 2.0
        opening = (lower - upper) / width
        # Typical wide-open ratio is ~0.30-0.40; squinting ~0.18; blink <0.12.
        return float(max(0.0, min(1.0, (opening - 0.10) / 0.25)))
    except (AttributeError, TypeError, IndexError):
        return 0.0


def raw_gaze_from_tracker(gaze) -> Optional[Tuple[float, float, bool]]:
    """Return (h, v, blinking) or None.

    Per-eye fusion uses opening-score weighting instead of flat 50/50 —
    when one eye is half-closed (occlusion, lid droop, asymmetric blink)
    it contributes proportionally less to the pooled estimate.
    """
    if not gaze.pupils_located or gaze.eye_left is None or gaze.eye_right is None:
        return None
    try:
        el, er = gaze.eye_left, gaze.eye_right
        # Smooth the landmarks IN-PLACE on the eye object so the helper
        # functions below see jitter-reduced coordinates.
        el.landmark_points = _LM_SMOOTH_L.smooth(el.landmark_points)
        er.landmark_points = _LM_SMOOTH_R.smooth(er.landmark_points)
        wl = _eye_opening_score(el)
        wr = _eye_opening_score(er)
        # Both eyes effectively closed → blink. Don't update the filter.
        blinking = (wl + wr) < 0.30 or (wl < 0.08 and wr < 0.08)
        if wl + wr < 1e-3:
            return None

        v_l = pupil_vertical_raw(el)
        v_r = pupil_vertical_raw(er)
        # If the two eyes wildly disagree, drop to the more open eye only
        # — averaging a bad sample with a good one just halves the bias.
        if abs(v_l - v_r) > EYE_AGREE_MAX:
            v = v_l if wl >= wr else v_r
        else:
            v = (wl * v_l + wr * v_r) / (wl + wr)

        h_l = pupil_horizontal_ratio(el)
        h_r = pupil_horizontal_ratio(er)
        h = (wl * h_l + wr * h_r) / (wl + wr)
    except (AttributeError, TypeError, IndexError):
        return None
    if not (0.0 < h < 1.0 and 0.0 < v < 1.0):
        return None
    return float(h), float(v), bool(blinking)


def raw_to_screen_v(delta: float) -> float:
    """Continuous map matching the corrected polarity:
      delta < 0 (looking up)   -> screen_v < 0.5 (top    / question)
      delta > 0 (looking down) -> screen_v > 0.5 (bottom / answer)"""
    screen_v = 0.50 + delta * 4.5
    return float(max(0.08, min(0.92, screen_v)))


def heatmap_screen_v(zone: str, delta: float) -> float:
    """Stable heatmap Y per zone — pushes toward the correct screen edge as
    |delta| grows (negative for question/top, positive for answer/bottom)."""
    if zone == "question":
        # delta is negative here; adding it pulls toward the top edge.
        return float(max(0.10, min(0.38, 0.22 + delta * 1.5)))
    if zone == "answer":
        # delta is positive here; adding it pushes toward the bottom edge.
        return float(max(0.62, min(0.90, 0.78 + delta * 1.5)))
    return raw_to_screen_v(delta)


class VerticalGazePipeline:
    def __init__(self) -> None:
        self.anchor: Optional[float] = None
        self.question_delta = QUESTION_DELTA
        self.answer_delta = ANSWER_DELTA
        self._buf: Deque[float] = deque(maxlen=MEDIAN_WINDOW)
        self._ema: float = 0.5
        self._screen_v_ema: float = 0.5
        # Adaptive filter — replaces the old fixed-alpha EMA + saccade-snap
        # combo. Tuned for ~30 fps gaze input in raw_v ratio units.
        self._one_euro = OneEuroFilter(mincutoff=0.8, beta=0.5, dcutoff=1.0)
        # Frames to keep the EMA frozen after a blink ends (eye reopening
        # produces 1-2 frames of garbage pupil hits while the iris settles).
        self._blink_recovery: int = 0

        # Adaptive screen-v range — expands as we observe the user moving
        # their eyes. Solves the "dot only goes down a bit" problem on
        # laptop screens where the user's eye-movement range is tiny
        # (typical delta swing ~±0.03 instead of the ~±0.1 the fixed
        # mapping was tuned for). We grow only — never shrink — so a
        # single deliberate look at the extremes locks the calibration.
        self._max_pos_delta: float = 0.020
        self._max_neg_delta: float = -0.020

        # Adaptive screen-h range — same problem on the horizontal axis.
        # raw pupil-h is naturally compressed to roughly [0.35, 0.65] on
        # a laptop because the user's eyes don't physically rotate much
        # to cover the screen width. Without this remap the dot only
        # travels through the middle ~30% of the heatmap and "misses
        # the left/right ends" of the answer area.
        self._max_h: float = 0.55
        self._min_h: float = 0.45

    def calibrate(self, samples: List[float]) -> None:
        trimmed = _trimmed_samples(samples)
        self.anchor = median(trimmed)
        spread = min(_stdev(trimmed), 0.015)
        # Adaptive threshold from calibration spread rather than a fixed cap.
        # Lower bound 0.006 prevents noise triggering false switches;
        # upper bound 0.012 ensures the user's eye movements can reach both zones.
        threshold = min(max(spread * 0.8 + 0.003, 0.006), 0.012)
        self.question_delta = threshold
        self.answer_delta = -threshold
        # Pre-set adaptive screen-v range from the calibration spread so the
        # heatmap dot reaches screen edges from the first classified frame.
        est_range = max(spread * 3.0, 0.025)
        self._max_pos_delta = est_range
        self._max_neg_delta = -est_range
        self._ema = self.anchor
        self._screen_v_ema = 0.5
        self._buf.clear()
        for _ in range(MEDIAN_WINDOW):
            self._buf.append(self.anchor)
        self._one_euro.reset(self.anchor)
        self._blink_recovery = 0

    def update(self, raw_v: float, *, blinking: bool = False) -> float:
        """Filter the raw vertical pupil ratio.

        Pipeline:
          1. Skip entirely while the user is blinking — the pupil detector
             returns garbage when the eyelid covers the iris.
          2. Discard the first 2 frames after a blink ends (iris hasn't
             settled).
          3. Robust median over a short window (rejects single-frame
             outliers from sudden occlusion / detection glitches).
          4. One-Euro adaptive low-pass — heavy smoothing at rest,
             near-zero lag during saccades.
        """
        if blinking:
            self._blink_recovery = 2
            return self._ema
        if self._blink_recovery > 0:
            self._blink_recovery -= 1
            return self._ema

        self._buf.append(raw_v)
        # Median over the window: a single bad sample (head jolt, false
        # pupil detection) is rejected without distorting the result.
        med = median(list(self._buf))
        # One-Euro filter does the velocity-adaptive smoothing.
        self._ema = self._one_euro(med)
        return self._ema

    def delta(self) -> float:
        if self.anchor is None:
            return 0.0
        return self._ema - self.anchor

    def smooth_screen_v(self, target: float) -> float:
        # was 0.18 / 0.82 — far too slow for the heatmap dot to follow saccades
        self._screen_v_ema = 0.45 * target + 0.55 * self._screen_v_ema
        return self._screen_v_ema

    def observe_range(self, delta: float) -> None:
        """Call this every classified frame to grow the observed range.
        Used by adaptive_screen_v() so the dot reaches the screen edges
        for users with naturally small eye movements (laptop / close screens)."""
        if delta > self._max_pos_delta:
            self._max_pos_delta = delta
        elif delta < self._max_neg_delta:
            self._max_neg_delta = delta

    def adaptive_screen_v(self, delta: float) -> float:
        """Map delta to [0.08, 0.92] using each user's observed range
        rather than a one-size-fits-all multiplier."""
        if delta >= 0:
            denom = max(self._max_pos_delta, 0.005)
            return float(min(0.92, 0.50 + min(delta / denom, 1.0) * 0.42))
        denom = max(abs(self._max_neg_delta), 0.005)
        return float(max(0.08, 0.50 + max(delta / denom, -1.0) * 0.42))

    def observe_h_range(self, h: float) -> None:
        """Grow the observed horizontal range (call once per classified frame)."""
        if h > self._max_h:
            self._max_h = h
        elif h < self._min_h:
            self._min_h = h

    def adaptive_screen_h(self, h: float) -> float:
        """Map raw pupil-h to [0.05, 0.95] using the user's observed range.
        Fixes the 'dot doesn't reach left/right edges' problem on laptop
        cameras where the natural h-range is compressed to ~[0.35, 0.65]."""
        span = self._max_h - self._min_h
        if span < 0.01:
            return 0.5
        norm = (h - self._min_h) / span
        return float(max(0.05, min(0.95, norm)))

    @property
    def filtered_v(self) -> float:
        return self._ema


class VerticalZoneClassifier:
    def __init__(self, pipeline: VerticalGazePipeline) -> None:
        self._pipe = pipeline
        self._zone = "distracted"
        self._toward: Optional[str] = None
        self._toward_count = 0

    def reset(self) -> None:
        self._zone = "distracted"
        self._toward = None
        self._toward_count = 0

    @property
    def zone(self) -> str:
        return self._zone

    def _target_zone(self, d: float) -> str:
        q = self._pipe.question_delta   # positive threshold
        a = self._pipe.answer_delta     # negative threshold
        # Polarity matches the documented contract in gaze_tracker.py:
        #   delta < 0  -> eyes UP   -> looking at TOP of screen    -> question
        #   delta > 0  -> eyes DOWN -> looking at BOTTOM of screen -> answer
        # The previous code had these swapped, which is why the bottom area
        # was almost never reported and time piled up in "question".
        if d <= a:
            return "question"
        if d >= q:
            return "answer"
        if abs(d) <= DEAD_BAND and self._zone in ("question", "answer"):
            return self._zone
        return "distracted"

    def _commit(self, target: str, frames_needed: int) -> str:
        if target == self._zone:
            self._toward = None
            self._toward_count = 0
            return self._zone
        if target == self._toward:
            self._toward_count += 1
        else:
            self._toward = target
            self._toward_count = 1
        needed = frames_needed
        if self._zone in ("question", "answer") and target in ("question", "answer"):
            needed = FRAMES_TO_LEAVE
        elif target in ("question", "answer"):
            needed = FRAMES_TO_ENTER
        if self._toward_count >= needed:
            self._zone = target
            self._toward = None
            self._toward_count = 0
        return self._zone

    def classify(self) -> str:
        target = self._target_zone(self._pipe.delta())
        if target == "distracted" and self._zone in ("question", "answer"):
            return self._zone
        if target in ("question", "answer"):
            return self._commit(target, FRAMES_TO_ENTER)
        return self._zone
