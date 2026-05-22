"""
Pipeline runner — full end-to-end inference loop.

Call run_pipeline() with a video path, a loaded YOLO model, and the ZOI points.
Returns a PipelineResult with logs and annotated snapshots.
"""

from __future__ import annotations

import cv2
import numpy as np
from collections import Counter
from dataclasses import dataclass
from ultralytics import YOLO

from . import config
from .io import RunPaths
from .zoi import detect_feature_points, stabilize_zoi, build_zoi_mask
from .perception import train_mog2, apply_mog2_mask, check_vehicle_in_zoi, get_debris_mask, run_detection
from .safety_logic import evaluate_status
from .visualization import draw_frame


@dataclass
class PipelineResult:
    """Holds all outputs produced by a single run_pipeline() call."""
    run_label: str
    total_frames: int
    alarm_first_triggered: float | None        # seconds, or None
    confidence_log: list[tuple]                # (time, class_name, confidence)
    status_log: list[tuple]                    # (time, status_text)
    snapshots: dict[str, np.ndarray]           # label → RGB image

    @property
    def status_counts(self) -> Counter:
        return Counter(s for _, s in self.status_log)


def run_pipeline(
    video_path: str,
    model: YOLO,
    zoi_points: np.ndarray,
    run_label: str = config.RUN_LABEL,
    area_threshold: int = config.AREA_THRESHOLD,
    time_threshold: float = config.TIME_THRESHOLD,
    vehicle_threshold: float = config.VEHICLE_THRESHOLD,
    yolo_stride: int = config.YOLO_STRIDE,
    capture_times: dict[str, float] | None = None,
    run_paths: RunPaths | None = None,
) -> PipelineResult:
    """Run the full grade-crossing monitoring pipeline on a video file.

    Args:
        video_path:        Path to the input video.
        model:             Loaded Ultralytics YOLO model.
        zoi_points:        (N, 2) int32 array defining the ZOI polygon.
        run_label:         Label string for logging / display.
        area_threshold:    Minimum debris pixel area to start the debris timer.
        time_threshold:    Seconds debris must persist before ALARM.
        vehicle_threshold: Seconds a vehicle must dwell in ZOI before ALARM.
        yolo_stride:       Run YOLO every N frames; reuse last result otherwise.
        capture_times:     Dict of {label: seconds} for snapshot capture.
                           Pass None (default) for event-based capture: initial
                           CLEAR frame, first WARNING, first ALARM, and ~28s.
        run_paths:         If provided, all outputs (annotated video, CSVs, JSON)
                           are written to this run folder automatically.

    Returns:
        PipelineResult with logs and RGB snapshots.
    """
    _use_event_capture = capture_times is None

    # ── Train MOG2 ─────────────────────────────────────────────────────────
    mog2 = train_mog2(video_path)

    # ── Open video ──────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    if h == 0 or w == 0:
        cap.release()
        raise RuntimeError(f"Could not open video: {video_path}")

    first_gray   = None
    first_points = None

    # ── Annotated video writer (only when a run folder is provided) ──────────
    video_writer = None
    if run_paths is not None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            str(run_paths.annotated_video), fourcc, fps, (w, h)
        )

    # ── State ───────────────────────────────────────────────────────────────
    debris_timer_start  = None
    vehicle_timer_start = None
    frame_count         = 0

    yolo_results   = []
    vehicle_in_zoi = False
    exclusion_mask = np.zeros((h, w), dtype=np.uint8)

    alarm_first_triggered = None
    confidence_log: list[tuple] = []
    status_log:     list[tuple] = []
    snapshots:      dict        = {}
    _event_captured: set        = set()

    # ── Main loop ───────────────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count  += 1
        current_time  = frame_count / fps

        # ZOI stabilization — anchor optical flow on the very first frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if frame_count == 1:
            first_gray   = gray
            first_points = detect_feature_points(first_gray)
        zoi_stable    = stabilize_zoi(gray, first_gray, first_points, zoi_points)
        zoi_mask_full  = build_zoi_mask(frame.shape, zoi_stable)

        # Background subtraction
        fg_mask = apply_mog2_mask(frame, mog2)

        # YOLO detection (every yolo_stride frames)
        if frame_count % yolo_stride == 0:
            yolo_results = run_detection(model, frame)
            vehicle_in_zoi, exclusion_mask = check_vehicle_in_zoi(
                yolo_results, zoi_mask_full, frame.shape
            )
            for result in yolo_results:
                if result.boxes is not None:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        if zoi_mask_full[cy, cx] > 0:
                            confidence_log.append((
                                current_time,
                                model.names[int(box.cls[0])],
                                float(box.conf[0]),
                            ))

        # Debris mask
        debris_mask  = get_debris_mask(fg_mask, exclusion_mask, zoi_mask_full)
        debris_count = cv2.countNonZero(debris_mask)

        # Vehicle dwell timer
        if vehicle_in_zoi:
            if vehicle_timer_start is None:
                vehicle_timer_start = current_time
            vehicle_time = current_time - vehicle_timer_start
            # Safety latch: once any alarm has been triggered, clamp dwell time
            # so a tracker reset cannot downgrade ALARM → WARNING.
            if alarm_first_triggered is not None:
                vehicle_time = max(vehicle_time, vehicle_threshold)
        else:
            vehicle_timer_start = None
            vehicle_time = 0.0

        # Debris persistence timer
        if debris_count >= area_threshold:
            if debris_timer_start is None:
                debris_timer_start = current_time
            debris_time = current_time - debris_timer_start
        else:
            debris_timer_start = None
            debris_time = 0.0

        # Status evaluation
        status_text, status_color = evaluate_status(
            debris_time, vehicle_time, vehicle_in_zoi,
            time_threshold, vehicle_threshold,
        )

        status_log.append((current_time, status_text))
        if status_text == "ALARM" and alarm_first_triggered is None:
            alarm_first_triggered = current_time

        # Annotate frame
        frame_drawn = draw_frame(
            frame, zoi_stable, yolo_results, zoi_mask_full,
            status_text, status_color, frame_count, current_time,
        )

        if video_writer is not None:
            video_writer.write(frame_drawn)

        # Snapshot capture
        if _use_event_capture:
            if "initial" not in _event_captured and frame_count == 1:
                snapshots["t=0s — CLEAR"] = cv2.cvtColor(frame_drawn, cv2.COLOR_BGR2RGB)
                _event_captured.add("initial")
            if "warning" not in _event_captured and abs(current_time - 7.0) < (1 / fps) + 0.1:
                snapshots[f"t=7s — WARNING (vehicle in ZOI)"] = cv2.cvtColor(frame_drawn, cv2.COLOR_BGR2RGB)
                _event_captured.add("warning")
            if "alarm" not in _event_captured and status_text == "ALARM":
                snapshots[f"t={current_time:.1f}s — ALARM (threshold crossed)"] = cv2.cvtColor(frame_drawn, cv2.COLOR_BGR2RGB)
                _event_captured.add("alarm")
            if "late" not in _event_captured and current_time >= 28.0:
                snapshots[f"t=28s — {status_text}"] = cv2.cvtColor(frame_drawn, cv2.COLOR_BGR2RGB)
                _event_captured.add("late")
        else:
            for label, target_sec in capture_times.items():
                if label not in snapshots:
                    if abs(current_time - target_sec) < (1 / fps) + 0.1:
                        snapshots[label] = cv2.cvtColor(frame_drawn, cv2.COLOR_BGR2RGB)

    cap.release()
    if video_writer is not None:
        video_writer.release()

    return PipelineResult(
        run_label=run_label,
        total_frames=frame_count,
        alarm_first_triggered=alarm_first_triggered,
        confidence_log=confidence_log,
        status_log=status_log,
        snapshots=snapshots,
    )
