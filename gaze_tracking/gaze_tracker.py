"""Background gaze tracker — per-question sessions, implicit anchor calibration.

Runs headlessly, communicates with the C# app via TCP socket.
Generates per-question heatmaps and a comprehensive final JSON report.

Socket commands (one per TCP connection, newline-terminated):
  PING
  START_QUESTION <index> <screen_type> [question_text...]
      screen_type: build_a_sentence | whats_in_image | complete_sentence | spelling
  END_QUESTION [was_answered=0|1] [is_correct=0|1|-1]
      is_correct = -1 means "not applicable"
  GET_REPORT
  STOP

Legacy aliases (still accepted):
  START_GAME <name>   → same as START_QUESTION 0 <name>
  END_GAME            → same as END_QUESTION 0 -1
"""

import os
import sys
import argparse
import json
import socket
import signal
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

_this_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_this_dir))

from _lib import GazeTracking
from gaze_math import (
    VerticalGazePipeline,
    VerticalZoneClassifier,
    heatmap_screen_v,
    median,
    preprocess_frame,
    raw_gaze_from_tracker,
    raw_to_screen_v,
)
from heatmap import generate_heatmap


# ---------------------------------------------------------------------------
# Screen-type zone definitions
# split: "y" = top/bottom split (question top, answer bottom)
#        "x" = left/right split (question left = image, answer right = choices)
# threshold: fraction of screen where question zone ends (informational only,
#            anchor-relative displacement is used for live classification)
# ---------------------------------------------------------------------------

SCREEN_ZONES: Dict[str, Dict] = {
    "build_a_sentence":  {"split": "y", "threshold": 0.45},
    "whats_in_image":    {"split": "y", "threshold": 0.50},
    "complete_sentence": {"split": "y", "threshold": 0.40},
    "spelling":          {"split": "y", "threshold": 0.55},
    "draw_the_answer":   {"split": "y", "threshold": 0.45},
}

# Map legacy C# game names to SCREEN_ZONES keys
LEGACY_NAME_MAP: Dict[str, str] = {
    "buildsentence":     "build_a_sentence",
    "build_sentence":    "build_a_sentence",
    "build_a_sentence":  "build_a_sentence",
    "whatsinimage":      "whats_in_image",
    "whats_in_image":    "whats_in_image",
    "completesentence":  "complete_sentence",
    "complete_sentence": "complete_sentence",
    "spelling":          "spelling",
    "drawtheanswer":     "draw_the_answer",
    "draw_the_answer":   "draw_the_answer",
    "drawanswer":        "draw_the_answer",
    "draw":              "draw_the_answer",
}

# How long to collect gaze samples before fixing the anchor (ms)
ANCHOR_WINDOW_MS: float = 800.0

# Minimum relative displacement from anchor to classify a zone (dead zone)
# Vertical ratio range from GazeTracking is naturally compressed (pupil moves
# little within the eye socket for screen gaze), so keep this small.
DEAD_ZONE: float = 0.015

# Frames without a face before reporting no_face (reduces flicker)
NO_FACE_HYSTERESIS_FRAMES: int = 4
STABLE_FRAMES_BEFORE_LOG: int = 1
H_EMA_ALPHA: float = 0.40


# ---------------------------------------------------------------------------
# Per-question session data
# ---------------------------------------------------------------------------

@dataclass
class QuestionSession:
    """Accumulates gaze data for a single question screen."""

    question_index: int
    screen_type: str
    question_text: str
    split_axis: str          # "x" or "y" derived from SCREEN_ZONES
    start_time: float        # time.perf_counter()

    end_time: float = 0.0
    was_answered: bool = False
    answer_correct: Optional[bool] = None   # None = not applicable

    question_seconds: float = 0.0
    answer_seconds: float = 0.0
    distracted_seconds: float = 0.0
    no_face_seconds: float = 0.0

    # Raw (h, v) gaze ratio pairs for heatmap generation.
    # h: horizontal_ratio (0.0 = left corner, 1.0 = right corner)
    # v: vertical_ratio   (0.0 = looking up,    1.0 = looking down)
    gaze_points: List[Tuple[float, float]] = field(default_factory=list)

    # Vertical diagnostics for the session report
    v_samples: List[float] = field(default_factory=list, repr=False)
    frames_with_face: int = 0
    frames_total: int = 0

    # Zone switch counter — how many times user switched between question/answer
    zone_switches: int = 0
    _prev_zone: str = field(default="", repr=False)

    _smoothed_h: float = 0.5
    _pipeline: VerticalGazePipeline = field(default_factory=VerticalGazePipeline, repr=False)
    _zone_classifier: Optional[VerticalZoneClassifier] = field(default=None, repr=False)
    _stable_zone: str = field(default="distracted", repr=False)
    _stable_count: int = field(default=0, repr=False)
    _last_screen_v: float = 0.5
    _last_screen_h: float = 0.5

    # Implicit anchor — accumulated from first ANCHOR_WINDOW_MS of each question
    _anchor_samples: List[Tuple[float, float]] = field(default_factory=list, repr=False)
    anchor_h: Optional[float] = None
    anchor_v: Optional[float] = None

    # Per-question area bounds (set via SET_AREAS from C#)
    q_area_bottom: Optional[float] = None   # normalized Y where question area ends

    heatmap_path: str = ""

    # ------------------------------------------------------------------
    # Anchor (implicit calibration)
    # ------------------------------------------------------------------

    def try_set_anchor(self, h: float, v: float, now: float) -> None:
        """Feed a gaze sample during the anchor window.

        Once ANCHOR_WINDOW_MS has elapsed, averages all samples and locks in
        the anchor. Does nothing if the anchor is already set.
        """
        if self.anchor_h is not None:
            return
        elapsed_ms = (now - self.start_time) * 1000.0
        if elapsed_ms < ANCHOR_WINDOW_MS:
            self._anchor_samples.append((h, v))
        elif self._anchor_samples:
            self.anchor_h = median([s[0] for s in self._anchor_samples])
            vs = [s[1] for s in self._anchor_samples]
            self._pipeline.calibrate(vs)
            self.anchor_v = self._pipeline.anchor
            self._zone_classifier = VerticalZoneClassifier(self._pipeline)
            self._zone_classifier.reset()
            self._stable_zone = "distracted"
            self._stable_count = 0

    # ------------------------------------------------------------------
    # Zone classification (anchor-relative displacement)
    # ------------------------------------------------------------------

    def classify(self, h: float, v: float, blinking: bool = False) -> str:
        """Return 'question', 'answer', or 'distracted'.

        Uses the gaze DIRECTION relative to the anchor rather than absolute
        screen coordinates, so no user-facing calibration step is needed.

        horizontal_ratio semantics: 0.0 = looking right, 1.0 = looking left
        vertical_ratio   semantics: 0.0 = looking up,    1.0 = looking down

        For split_axis == "x" (whats_in_image):
            Question = image on LEFT of screen → user looks LEFT → h increases
            delta_h positive (h > anchor) → looking left  → question zone
            delta_h negative (h < anchor) → looking right → answer zone

        For split_axis == "y" (all other screens):
            Question = text at TOP of screen → user looks UP → v decreases
            delta_v negative (v < anchor) → looking up   → question zone
            delta_v positive (v > anchor) → looking down → answer zone
        """
        self._smoothed_h = H_EMA_ALPHA * h + (1.0 - H_EMA_ALPHA) * self._smoothed_h
        h = self._smoothed_h

        if self.anchor_h is None or self.anchor_v is None:
            return "distracted"

        if self.split_axis == "x":
            delta = h - self.anchor_h
            if delta > DEAD_ZONE:
                zone = "question"
            elif delta < -DEAD_ZONE:
                zone = "answer"
            else:
                zone = (
                    self._prev_zone
                    if self._prev_zone in ("question", "answer")
                    else "distracted"
                )
        else:
            self._pipeline.update(v, blinking=blinking)
            if self._zone_classifier is not None:
                delta_v = self._pipeline.delta()
                self._pipeline.observe_range(delta_v)
                if not blinking:
                    self._pipeline.observe_h_range(self._smoothed_h)
                screen_v = self._pipeline.smooth_screen_v(
                    self._pipeline.adaptive_screen_v(delta_v)
                )
                self._last_screen_v = screen_v
                self._last_screen_h = self._pipeline.adaptive_screen_h(self._smoothed_h)
                if self.q_area_bottom is not None:
                    # Use a fixed 50/50 split boundary for balanced classification.
                    boundary = 0.5
                    hyst = 0.04
                    if screen_v < boundary - hyst:
                        zone = "question"
                    elif screen_v > boundary + hyst:
                        zone = "answer"
                    else:
                        zone = self._prev_zone if self._prev_zone in ("question", "answer") else "distracted"
                else:
                    zone = self._zone_classifier.classify()
            else:
                zone = "distracted"

        # Track zone switches (only count question <-> answer transitions)
        if zone in ("question", "answer") and self._prev_zone in ("question", "answer"):
            if zone != self._prev_zone:
                self.zone_switches += 1
        if zone in ("question", "answer"):
            self._prev_zone = zone

        return zone

    # ------------------------------------------------------------------
    # Time accumulation
    # ------------------------------------------------------------------

    def add_time(self, zone: str, delta: float) -> None:
        if zone == "question":
            self.question_seconds += delta
        elif zone == "answer":
            self.answer_seconds += delta
        elif zone == "distracted":
            self.distracted_seconds += delta
        elif zone == "no_face":
            self.no_face_seconds += delta

    # ------------------------------------------------------------------
    # Behavioral analysis
    # ------------------------------------------------------------------

    def behavioral_flags(self) -> List[str]:
        flags: List[str] = []
        if self.zone_switches > 3:
            flags.append("Hesitation detected")
        if self.question_seconds < 0.5 and self.anchor_h is not None:
            flags.append("Question skipped - very little time on question zone")
        if not self.was_answered:
            flags.append("Not answered")
        if self.answer_correct is False:
            flags.append("Incorrect answer")
        return flags

    def vertical_diagnostics(self) -> dict:
        """Summarize vertical gaze spread and bias relative to anchor."""
        if not self.v_samples or self.anchor_v is None:
            return {
                "anchor_v": round(self.anchor_v, 4) if self.anchor_v is not None else None,
                "mean_v": None,
                "median_v": None,
                "std_v": None,
                "mean_delta_from_anchor": None,
                "bias": "insufficient_data",
            }
        mean_v = sum(self.v_samples) / len(self.v_samples)
        median_v = median(self.v_samples)
        variance = sum((x - mean_v) ** 2 for x in self.v_samples) / len(self.v_samples)
        std_v = variance ** 0.5
        screen_samples = [
            raw_to_screen_v(v - self.anchor_v) for v in self.v_samples
        ]
        mean_screen = sum(screen_samples) / len(screen_samples)
        if mean_screen < 0.42:
            bias = "toward_question"
        elif mean_screen > 0.58:
            bias = "toward_answer"
        else:
            bias = "neutral"
        return {
            "anchor_raw": round(self.anchor_v, 4),
            "mean_raw_v": round(mean_v, 4),
            "median_raw_v": round(median_v, 4),
            "std_raw_v": round(std_v, 4),
            "mean_screen_v": round(mean_screen, 4),
            "bias": bias,
        }

    def face_detection_rate(self) -> float:
        if self.frames_total <= 0:
            return 0.0
        return round(self.frames_with_face / self.frames_total, 4)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        total = (self.question_seconds + self.answer_seconds
                 + self.distracted_seconds + self.no_face_seconds)
        active = self.question_seconds + self.answer_seconds

        def pct(v: float, base: float) -> float:
            return round((v / base) * 100, 2) if base > 0 else 0.0

        return {
            "question_index":     self.question_index,
            "screen_type":        self.screen_type,
            "question_text":      self.question_text,
            "duration_seconds":   round(total, 3),
            "was_answered":       self.was_answered,
            "answer_correct":     self.answer_correct,
            "question_seconds":   round(self.question_seconds, 3),
            "answer_seconds":     round(self.answer_seconds, 3),
            "distracted_seconds": round(self.distracted_seconds, 3),
            "no_face_seconds":    round(self.no_face_seconds, 3),
            "zone_switches":      self.zone_switches,
            "anchor": {
                "h": round(self.anchor_h, 4) if self.anchor_h is not None else None,
                "v": round(self.anchor_v, 4) if self.anchor_v is not None else None,
            },
            "percentages": {
                "question_percent": pct(self.question_seconds, active),
                "answer_percent":   pct(self.answer_seconds, active),
            },
            "gaze_points_count":  len(self.gaze_points),
            "face_detection_rate": self.face_detection_rate(),
            "vertical_diagnostics": self.vertical_diagnostics(),
            "behavioral_flags":   self.behavioral_flags(),
            "heatmap_path":       self.heatmap_path,
        }


# ---------------------------------------------------------------------------
# Main tracker
# ---------------------------------------------------------------------------

class GazeSessionTracker:
    """Background gaze tracker with per-question session support."""

    def __init__(self, camera_index: int = 0, output_dir: Optional[str] = None):
        self.camera_index = camera_index
        self.gaze = GazeTracking()

        # Force the fast Windows path (see split_screen_gaze._open_camera for
        # rationale): DirectShow + MJPG + 640x480 @ 30 fps + 1-frame buffer.
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(camera_index)
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera index {camera_index}")

        self.active = True
        self.start_time_utc = datetime.now(timezone.utc)
        self.end_time_utc: Optional[datetime] = None

        self._lock = threading.Lock()

        self.output_dir = output_dir or str(_this_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "heatmaps"), exist_ok=True)

        # Current signed-in user (set via SET_USER socket command).
        # When set, per-question heatmaps + the per-user report go to
        # <output_dir>/users/<sanitised_mac>/  instead of the shared
        # <output_dir>/heatmaps/ folder.
        self.user_mac: Optional[str] = None
        self.user_name: Optional[str] = None
        self._user_dir: Optional[str] = None

        self.completed_sessions: List[QuestionSession] = []
        self.current_session: Optional[QuestionSession] = None
        self.last_tick: float = 0.0
        self.frame_count: int = 0
        self._no_face_streak: int = 0
        self._session_frames_with_face: int = 0
        self._session_frames_total: int = 0

        # Live zone state — exposed via GET_ZONE for the C# dwell-trigger UI.
        # Updated on every _tick() so dwell time reflects the latest classifier
        # output without polling-rate aliasing.
        self._live_zone: str = "no_face"
        self._live_zone_start: float = time.perf_counter()

    # -- lifecycle ----------------------------------------------------------

    def stop(self, *_args) -> None:
        self.active = False

    # -- gaze helpers -------------------------------------------------------

    def _raw_gaze(self) -> Optional[Tuple[float, float, bool]]:
        """Return landmark-normalized (h, v, blinking) if pupils are located."""
        return raw_gaze_from_tracker(self.gaze)

    def _has_face(self) -> bool:
        return self.gaze.eye_left is not None and self.gaze.eye_right is not None

    def _classify_zone(self) -> str:
        """Classify current gaze into question / answer / distracted / no_face."""
        has_face = self._has_face()
        if has_face:
            self._no_face_streak = 0
        else:
            self._no_face_streak += 1
            if self._no_face_streak < NO_FACE_HYSTERESIS_FRAMES:
                return "distracted"
            return "no_face"

        with self._lock:
            session = self.current_session
            if session is not None:
                session.frames_total += 1
                session.frames_with_face += 1
                self._session_frames_total += 1
                self._session_frames_with_face += 1

        raw = self._raw_gaze()
        if raw is None:
            return "distracted"

        h, v, blinking = raw
        now = time.perf_counter()

        with self._lock:
            session = self.current_session
            if session is None:
                return "distracted"
            session.v_samples.append(v)
            session.try_set_anchor(h, v, now)
            zone = session.classify(h, v, blinking=blinking)
            if session.anchor_h is not None and session.anchor_v is not None:
                if zone == session._stable_zone:
                    session._stable_count += 1
                else:
                    session._stable_zone = zone
                    session._stable_count = 1
                if (session._stable_count >= STABLE_FRAMES_BEFORE_LOG
                        and zone in ("question", "answer")):
                    if session.q_area_bottom is not None:
                        screen_v = session._last_screen_v
                        screen_h = session._pipeline.adaptive_screen_h(session._smoothed_h)
                    else:
                        delta_v = session._pipeline.delta()
                        screen_v = session._pipeline.smooth_screen_v(
                            session._pipeline.adaptive_screen_v(delta_v)
                        )
                        screen_h = session._pipeline.adaptive_screen_h(session._smoothed_h)
                    session.gaze_points.append((screen_h, screen_v))

        return zone

    def _tick(self, zone: str) -> None:
        now = time.perf_counter()
        delta = now - self.last_tick
        self.last_tick = now
        # Track live zone + dwell start (consumed by GET_ZONE).
        # Reset the dwell clock whenever the zone changes.
        if zone != self._live_zone:
            self._live_zone = zone
            self._live_zone_start = now
        if delta <= 0 or delta > 1.0:
            # Skip outlier deltas (e.g. first tick or long pauses)
            return
        with self._lock:
            if self.current_session is not None:
                self.current_session.add_time(zone, delta)

    def current_zone_info(self) -> Tuple[str, float]:
        """Return (zone, dwell_seconds) — used by the C# UI to drive
        dwell-threshold triggers (e.g. font scaling after 10 s)."""
        return self._live_zone, max(0.0, time.perf_counter() - self._live_zone_start)

    def set_user(self, mac: str, name: str = "") -> str:
        """Bind the tracker to a signed-in user.

        Per-question heatmaps and the user's report file are routed to
        <output_dir>/users/<sanitised_mac>/ so multiple players on the
        same machine each have their own dataset.

        Returns the resolved user directory."""
        # Sanitise MAC for a safe folder name (replace colons, spaces).
        safe = "".join(c for c in (mac or "") if c.isalnum() or c in ("-", "_")).strip()
        if not safe:
            safe = "anonymous"
        user_dir = os.path.join(self.output_dir, "users", safe, "heatmaps")
        os.makedirs(user_dir, exist_ok=True)
        with self._lock:
            self.user_mac = mac
            self.user_name = name
            self._user_dir = user_dir
        # Also write/refresh a small profile file so reports are traceable.
        profile = os.path.join(self.output_dir, "users", safe, "profile.json")
        try:
            with open(profile, "w", encoding="utf-8") as f:
                json.dump({"mac": mac, "name": name,
                           "updated_utc": datetime.now(timezone.utc).isoformat()},
                          f, indent=2)
        except OSError:
            pass
        return user_dir

    def _current_heatmap_dir(self) -> str:
        """Per-user folder if SET_USER has been received, else the shared one."""
        if self._user_dir:
            return self._user_dir
        return os.path.join(self.output_dir, "heatmaps")

    # -- per-question session control ---------------------------------------

    def start_question(
        self,
        question_index: int,
        screen_type: str,
        question_text: str = "",
    ) -> None:
        """Begin tracking a new question screen.

        The first ANCHOR_WINDOW_MS of gaze data are used as the implicit
        anchor — no user-facing calibration is required.
        """
        zone_info = SCREEN_ZONES.get(screen_type)
        if zone_info is None:
            print(f"[GAZE] WARNING: unknown screen_type '{screen_type}', defaulting to Y-split")
            split_axis = "y"
        else:
            split_axis = zone_info["split"]

        with self._lock:
            if self.current_session is not None:
                self._finalise_session_locked()
            self.current_session = QuestionSession(
                question_index=question_index,
                screen_type=screen_type,
                question_text=question_text,
                split_axis=split_axis,
                start_time=time.perf_counter(),
            )
        self._no_face_streak = 0
        self.last_tick = time.perf_counter()
        print(f"[GAZE] Question {question_index} started: {screen_type!r}")

    def end_question(
        self,
        was_answered: bool = False,
        answer_correct: Optional[bool] = None,
    ) -> None:
        """Stop tracking the current question and generate its heatmap."""
        with self._lock:
            if self.current_session is None:
                return
            self.current_session.was_answered = was_answered
            self.current_session.answer_correct = answer_correct
            self._finalise_session_locked()

    def _finalise_session_locked(self) -> None:
        """Save heatmap and move session to completed list. Must hold _lock."""
        session = self.current_session
        if session is None:
            return
        session.end_time = time.perf_counter()

        seq = len(self.completed_sessions)
        # Filename also embeds user name for at-a-glance ID in the folder.
        user_tag = ""
        if self.user_name:
            safe_name = "".join(c for c in self.user_name if c.isalnum() or c in ("-", "_"))
            if safe_name:
                user_tag = f"_{safe_name[:24]}"
        fname = (f"heatmap_{seq:02d}_q{session.question_index:02d}"
                 f"_{session.screen_type}{user_tag}.png")
        path = os.path.join(self._current_heatmap_dir(), fname)

        generate_heatmap(
            gaze_points=session.gaze_points,
            split_axis=session.split_axis,
            anchor_h=session.anchor_h,
            anchor_v=session.anchor_v,
            output_path=path,
            title=f"Q{session.question_index} - {session.screen_type}",
            screen_type=session.screen_type,
            behavioral_flags=session.behavioral_flags(),
            question_seconds=session.question_seconds,
            answer_seconds=session.answer_seconds,
        )
        session.heatmap_path = path
        self.completed_sessions.append(session)
        self.current_session = None
        print(f"[GAZE] Question {session.question_index} ended -> {path}")

    # -- main camera loop ---------------------------------------------------

    def run(self) -> None:
        self.last_tick = time.perf_counter()
        while self.active:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                if self.current_session is not None:
                    self._tick("no_face")
                continue

            self.frame_count += 1
            self.gaze.refresh(preprocess_frame(frame))

            if self.current_session is not None:
                with self._lock:
                    if self.current_session is not None and not self._has_face():
                        self.current_session.frames_total += 1
                        self._session_frames_total += 1
                zone = self._classify_zone()
                self._tick(zone)

        self.end_time_utc = datetime.now(timezone.utc)
        self.end_question()   # finalise any active session
        self.cap.release()

    # -- report -------------------------------------------------------------

    def report(self) -> dict:
        with self._lock:
            sessions = [s.to_dict() for s in self.completed_sessions]
            if self.current_session is not None:
                sessions.append(self.current_session.to_dict())

        total_q  = sum(s["question_seconds"]   for s in sessions)
        total_a  = sum(s["answer_seconds"]      for s in sessions)
        total_d  = sum(s["distracted_seconds"]  for s in sessions)
        total_nf = sum(s["no_face_seconds"]     for s in sessions)
        active   = total_q + total_a + total_d

        def pct(v: float, base: float) -> float:
            return round((v / base) * 100, 2) if base > 0 else 0.0

        # Which screen type causes the most zone switches (difficulty indicator)
        by_type: Dict[str, int] = {}
        for s in sessions:
            st = s["screen_type"]
            by_type[st] = by_type.get(st, 0) + s["zone_switches"]
        hardest_type = max(by_type, key=by_type.get) if by_type else None

        face_rate = (
            round(self._session_frames_with_face / self._session_frames_total, 4)
            if self._session_frames_total > 0 else 0.0
        )
        downward_questions = [
            s["question_index"] for s in sessions
            if s.get("vertical_diagnostics", {}).get("bias") == "downward"
        ]

        return {
            "session_started_utc": self.start_time_utc.isoformat(),
            "session_ended_utc": (
                self.end_time_utc.isoformat() if self.end_time_utc else None
            ),
            "camera_index":    self.camera_index,
            "frames_processed": self.frame_count,
            "totals": {
                "question_seconds":   round(total_q, 3),
                "answer_seconds":     round(total_a, 3),
                "distracted_seconds": round(total_d, 3),
                "no_face_seconds":    round(total_nf, 3),
                "active_gaze_seconds": round(active, 3),
            },
            "percentages": {
                "question_percent":   pct(total_q, active),
                "answer_percent":     pct(total_a, active),
                "distracted_percent": pct(total_d, active),
            },
            "difficulty_by_screen_type": by_type,
            "most_zone_switches_screen": hardest_type,
            "diagnostics": {
                "face_detection_rate": face_rate,
                "stable_frames_before_log": STABLE_FRAMES_BEFORE_LOG,
                "anchor_window_ms": ANCHOR_WINDOW_MS,
                "questions_with_downward_bias": downward_questions,
                "notes": (
                    "vertical_ratio uses pupil position within eyelid landmarks "
                    "(0=up, 1=down). Anchor is median of first "
                    f"{ANCHOR_WINDOW_MS:.0f}ms per question."
                ),
            },
            "questions": sessions,
        }


# ---------------------------------------------------------------------------
# Socket control
# ---------------------------------------------------------------------------

def _socket_worker(tracker: GazeSessionTracker, host: str, port: int) -> None:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        server.bind((host, port))
    except OSError as exc:
        # Fail loudly — silent loss of this socket is what caused C# to
        # send commands to whatever OTHER server happened to own the port
        # (e.g. the laser server's "bad_json" response).
        print(f"[GAZE] FATAL: could not bind {host}:{port} — {exc}", flush=True)
        print(f"[GAZE]   Likely another process owns the port. Check that "
              f"draw_answer_server.py (5002) and gaze_tracker.py (5004) "
              f"are using different ports.", flush=True)
        tracker.stop()
        return
    server.listen(5)
    server.settimeout(0.5)
    print(f"[GAZE] Control socket on {host}:{port}")

    try:
        while tracker.active:
            try:
                conn, _ = server.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            with conn:
                conn.settimeout(1.0)
                try:
                    raw = conn.recv(4096)
                except OSError:
                    continue
                if not raw:
                    continue
                line = raw.decode("utf-8", errors="ignore").strip()
                parts = line.split(maxsplit=3)   # max 4 tokens, last absorbs spaces
                if not parts:
                    conn.sendall(b"ERROR empty command\n")
                    continue
                cmd = parts[0].upper()

                if cmd == "PING":
                    conn.sendall(b"PONG\n")

                elif cmd == "STOP":
                    tracker.stop()
                    conn.sendall(b"OK stopping\n")

                elif cmd == "GET_REPORT":
                    payload = json.dumps(tracker.report())
                    conn.sendall((payload + "\n").encode("utf-8"))

                elif cmd == "GET_ZONE":
                    # Cheap, high-rate poll endpoint for the C# dwell UI.
                    # Response format: "OK <zone> <dwell_seconds>"
                    zone, dwell = tracker.current_zone_info()
                    conn.sendall(f"OK {zone} {dwell:.3f}\n".encode("utf-8"))

                elif cmd == "SET_USER" and len(parts) >= 2:
                    # SET_USER <mac> [name...]   — routes per-question
                    # heatmaps + the final report into a per-user folder.
                    mac = parts[1]
                    name = parts[2] if len(parts) > 2 else ""
                    if len(parts) > 3:
                        name = name + " " + parts[3]
                    user_dir = tracker.set_user(mac, name)
                    conn.sendall(f"OK user_dir {user_dir}\n".encode("utf-8"))

                elif cmd == "SET_AREAS" and len(line.split()) == 9:
                    # SET_AREAS qx qy qw qh ax ay aw ah (all normalized 0-1)
                    try:
                        vals = [float(x) for x in line.split()[1:]]
                        qx, qy, qw, qh = vals[0], vals[1], vals[2], vals[3]
                        if tracker.current_session is not None:
                            # Store question area bottom Y as zone boundary
                            tracker.current_session.q_area_bottom = qy + qh
                        conn.sendall(b"OK areas set\n")
                    except (ValueError, IndexError) as exc:
                        conn.sendall(f"ERROR {exc}\n".encode())

                # ── New per-question commands ──────────────────────────────

                elif cmd == "START_QUESTION" and len(parts) >= 3:
                    # START_QUESTION <index> <screen_type> [question text...]
                    try:
                        q_index = int(parts[1])
                        screen_type = parts[2]
                        q_text = parts[3] if len(parts) > 3 else ""
                        tracker.start_question(q_index, screen_type, q_text)
                        conn.sendall(b"OK question started\n")
                    except (ValueError, IndexError) as exc:
                        conn.sendall(f"ERROR {exc}\n".encode())

                elif cmd == "END_QUESTION":
                    # END_QUESTION [was_answered=0|1] [is_correct=0|1|-1]
                    try:
                        was_answered = bool(int(parts[1])) if len(parts) > 1 else False
                        ic_raw = int(parts[2]) if len(parts) > 2 else -1
                        answer_correct = None if ic_raw == -1 else bool(ic_raw)
                        tracker.end_question(was_answered, answer_correct)
                        conn.sendall(b"OK question ended\n")
                    except (ValueError, IndexError) as exc:
                        conn.sendall(f"ERROR {exc}\n".encode())

                # ── Legacy aliases ─────────────────────────────────────────

                elif cmd == "START_GAME" and len(parts) >= 2:
                    # Map legacy C# game name to screen_type
                    name = parts[1]
                    key = name.lower().replace(" ", "_")
                    st = LEGACY_NAME_MAP.get(key, key)
                    if st not in SCREEN_ZONES:
                        st = "complete_sentence"
                    q_idx = len(tracker.completed_sessions)
                    tracker.start_question(q_idx, st, name)
                    conn.sendall(b"OK game started (legacy)\n")

                elif cmd == "END_GAME":
                    tracker.end_question(was_answered=False)
                    conn.sendall(b"OK game ended (legacy)\n")

                else:
                    conn.sendall(b"ERROR unknown command\n")

    finally:
        try:
            server.close()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Background gaze tracker with per-question reports and heatmaps."
    )
    p.add_argument("--camera-index", type=int, default=0)
    p.add_argument(
        "--output",
        default=str(_this_dir / "gaze_report.json"),
        help="Path to write final JSON report.",
    )
    p.add_argument("--socket", action="store_true", help="Enable TCP control socket.")
    p.add_argument("--socket-host", default="127.0.0.1")
    # 5004 — was 5002, but the Laser Server (draw_answer_server.py) also
    # binds 5002, so the second-launched process silently lost its socket
    # and every START_GAME from C# ended up at the laser server, which
    # answered "bad_json". Heatmaps and per-question tracking both died.
    p.add_argument("--socket-port", type=int, default=5004)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    output_path = Path(args.output).resolve()

    tracker = GazeSessionTracker(
        camera_index=args.camera_index,
        output_dir=str(output_path.parent),
    )

    signal.signal(signal.SIGINT, tracker.stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, tracker.stop)

    if args.socket:
        t = threading.Thread(
            target=_socket_worker,
            args=(tracker, args.socket_host, args.socket_port),
            daemon=True,
        )
        t.start()

    tracker.run()
    report = tracker.report()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Also drop a copy in the per-user folder so each player's reports
    # are co-located with their heatmaps.
    if tracker._user_dir:
        user_report = Path(tracker._user_dir).parent / "report.json"
        user_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[GAZE] Per-user report: {user_report}")

    print(f"[GAZE] Report saved: {output_path}")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())