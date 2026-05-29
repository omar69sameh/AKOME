"""Heatmap generation utility for gaze tracking data.

Renders a per-question heatmap image with:
- JET colormap density overlay from accumulated gaze points
- Anchor-relative zone divider line (where anchor = implicit calibration center)
- Zone labels (question / answer) on correct sides for the screen type
- Behavioral flag text overlay
- Time-in-zone summary bar at the bottom
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_heatmap(
    gaze_points: List[Tuple[float, float]],
    split_axis: str,                    # "x" or "y"
    anchor_h: Optional[float],          # horizontal anchor (0-1), None if no data
    anchor_v: Optional[float],          # vertical anchor   (0-1), None if no data
    output_path: str,
    title: str = "",
    screen_type: str = "",
    behavioral_flags: Optional[List[str]] = None,
    question_seconds: float = 0.0,
    answer_seconds: float = 0.0,
    width: int = 1024,
    height: int = 640,
) -> None:
    """Generate and save a per-question heatmap image.

    Coordinate conventions
    ----------------------
    Raw gaze points are (horizontal_ratio, vertical_ratio) pairs from the
    GazeTracking library:
        horizontal_ratio: 0.0 = looking right, 1.0 = looking left
        vertical_ratio:   0.0 = looking up,    1.0 = looking down

    We FLIP the x-axis before rendering so the heatmap matches the physical
    screen layout (left of heatmap = user was looking left = left of screen).

    Zone divider
    ------------
    The divider is drawn at the anchor position, which is the implicit
    calibration center recorded during the first 800 ms of the question.
    A DEAD_ZONE band (±5% of the axis) is shown around the divider.
    """
    behavioral_flags = behavioral_flags or []

    # Reserve a bottom strip for the time summary bar
    bar_h = 60
    img_h = height - bar_h   # usable area for the heatmap

    # ------------------------------------------------------------------
    # Build the heatmap density canvas
    # ------------------------------------------------------------------

    density = np.zeros((img_h, width), dtype=np.float32)

    for h_ratio, v_ratio in gaze_points:
        # Flip x so left-of-heatmap = user looking left = left side of screen
        xi = int(max(0, min(width - 1,  (1.0 - h_ratio) * width)))
        yi = int(max(0, min(img_h - 1, v_ratio * img_h)))
        density[yi, xi] += 1.0

    # Gaussian smoothing
    sigma_px = int(width * 0.05)          # 5% of width ≈ natural gaze spread
    sigma_px = max(sigma_px, 21) | 1      # ensure odd, at least 21
    density = cv2.GaussianBlur(density, (sigma_px, sigma_px), 0)

    max_val = np.max(density)
    if max_val > 0:
        density = (density / max_val * 255).astype(np.uint8)
    else:
        density = density.astype(np.uint8)

    # ------------------------------------------------------------------
    # Base canvas
    # ------------------------------------------------------------------

    # Dark grey background so the heatmap reads well in both cases
    canvas = np.full((img_h, width, 3), (30, 30, 30), dtype=np.uint8)

    if max_val > 0:
        colored = cv2.applyColorMap(density, cv2.COLORMAP_JET)
        # Blend at 70% opacity so the background context is still visible
        cv2.addWeighted(colored, 0.70, canvas, 0.30, 0, dst=canvas)

    # ------------------------------------------------------------------
    # Zone divider line + dead-zone band
    # ------------------------------------------------------------------

    DEAD_ZONE = 0.015   # matches gaze_tracker.py

    if split_axis == "x":
        # Vertical divider — anchor is on the horizontal axis
        # We flipped x, so the anchor position maps to: (1 - anchor_h) * width
        if anchor_h is not None:
            center_x = int((1.0 - anchor_h) * width)
            dz_px = int(DEAD_ZONE * width)

            # Dead-zone band (semi-transparent dark strip)
            overlay = canvas.copy()
            cv2.rectangle(overlay,
                          (center_x - dz_px, 0),
                          (center_x + dz_px, img_h),
                          (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0, dst=canvas)

            # Divider line
            cv2.line(canvas, (center_x, 0), (center_x, img_h), (255, 255, 255), 1)

            # Zone corner labels
            _put_zone_label(canvas, "QUESTION (image)", 12, 28)
            _put_zone_label(canvas, "ANSWER (choices)", width - 12, 28, anchor="right")
        else:
            _no_data_text(canvas, width, img_h)

    else:   # "y" — horizontal divider
        if anchor_v is not None:
            center_y = int(anchor_v * img_h)
            dz_px = int(DEAD_ZONE * img_h)

            overlay = canvas.copy()
            cv2.rectangle(overlay,
                          (0, center_y - dz_px),
                          (width, center_y + dz_px),
                          (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0, dst=canvas)

            cv2.line(canvas, (0, center_y), (width, center_y), (255, 255, 255), 1)

            _put_zone_label(canvas, "QUESTION", 12, 28)
            _put_zone_label(canvas, "ANSWER", 12, img_h - 12)
        else:
            _no_data_text(canvas, width, img_h)

    # ------------------------------------------------------------------
    # Title
    # ------------------------------------------------------------------

    if title:
        cv2.putText(canvas, title,
                    (width // 2, 20),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, (230, 230, 230), 1,
                    cv2.LINE_AA)

    # ------------------------------------------------------------------
    # Behavioral flags overlay (bottom-left of the heatmap area)
    # ------------------------------------------------------------------

    y_flag = img_h - 12 - (len(behavioral_flags) * 22)
    for flag in behavioral_flags:
        y_flag += 22
        # Shadow for readability on any background
        cv2.putText(canvas, f"! {flag}",
                    (12, y_flag + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"! {flag}",
                    (12, y_flag),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 240), 1, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # Time summary bar (bottom strip)
    # ------------------------------------------------------------------

    bar_canvas = _draw_time_bar(width, bar_h, question_seconds, answer_seconds, screen_type)
    full_image = np.vstack([canvas, bar_canvas])

    cv2.imwrite(output_path, full_image)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _put_zone_label(
    img: np.ndarray,
    text: str,
    x: int,
    y: int,
    anchor: str = "left",
    color: Tuple[int, int, int] = (200, 200, 200),
) -> None:
    scale, thickness = 0.52, 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    if anchor == "right":
        x = x - tw
    # Shadow
    cv2.putText(img, text, (x + 1, y + 1), font, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def _no_data_text(img: np.ndarray, width: int, height: int) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    msg = "No gaze data - anchor not established"
    (tw, _), _ = cv2.getTextSize(msg, font, 0.65, 1)
    cv2.putText(img, msg, ((width - tw) // 2, height // 2),
                font, 0.65, (80, 80, 80), 1, cv2.LINE_AA)


def _draw_time_bar(
    width: int,
    bar_h: int,
    question_seconds: float,
    answer_seconds: float,
    screen_type: str,
) -> np.ndarray:
    """Render a horizontal proportional bar showing question vs answer time."""
    bar = np.full((bar_h, width, 3), (20, 20, 20), dtype=np.uint8)

    total = question_seconds + answer_seconds
    if total <= 0:
        cv2.putText(bar, "No zone time recorded",
                    (12, bar_h // 2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1, cv2.LINE_AA)
        return bar

    # Proportional widths
    q_w = int((question_seconds / total) * (width - 24))
    a_w = (width - 24) - q_w

    # Question block (teal)
    cv2.rectangle(bar, (12, 12), (12 + q_w, bar_h - 12), (0, 160, 140), -1)
    # Answer block (orange)
    cv2.rectangle(bar, (12 + q_w, 12), (12 + q_w + a_w, bar_h - 12), (0, 140, 210), -1)

    # Labels
    q_label = f"Question  {question_seconds:.1f}s"
    a_label = f"Answer  {answer_seconds:.1f}s"
    font = cv2.FONT_HERSHEY_SIMPLEX

    (q_tw, _), _ = cv2.getTextSize(q_label, font, 0.42, 1)
    q_center_x = 12 + q_w // 2
    if q_w > q_tw + 10:
        cv2.putText(bar, q_label, (q_center_x - q_tw // 2, bar_h // 2 + 5),
                    font, 0.42, (20, 20, 20), 1, cv2.LINE_AA)

    (a_tw, _), _ = cv2.getTextSize(a_label, font, 0.42, 1)
    a_center_x = 12 + q_w + a_w // 2
    if a_w > a_tw + 10:
        cv2.putText(bar, a_label, (a_center_x - a_tw // 2, bar_h // 2 + 5),
                    font, 0.42, (20, 20, 20), 1, cv2.LINE_AA)

    # Screen type label (bottom-right of the bar)
    if screen_type:
        st_label = screen_type.replace("_", " ")
        (st_tw, _), _ = cv2.getTextSize(st_label, font, 0.38, 1)
        cv2.putText(bar, st_label,
                    (width - st_tw - 12, bar_h - 4),
                    font, 0.38, (80, 80, 80), 1, cv2.LINE_AA)

    return bar