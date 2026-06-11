"""
Visualization utilities — frame annotation for display and snapshots.
"""

import cv2
import numpy as np


def draw_frame(
    frame: np.ndarray,
    zoi_stabilized: np.ndarray,
    yolo_results: list,
    zoi_mask: np.ndarray,
    status_text: str,
    status_color: tuple,
    frame_count: int,
    current_time: float,
) -> np.ndarray:
    """Annotate a frame with the ZOI polygon, YOLO boxes, and status overlay.

    Returns a new annotated copy; the original frame is not modified.
    """
    h, w = frame.shape[:2]
    # Scale factor relative to 1920px reference width — all sizes stay
    # proportional regardless of input resolution.
    s = w / 1920.0

    frame_drawn = frame.copy()

    # ZOI polygon
    cv2.polylines(frame_drawn, [zoi_stabilized], isClosed=True,
                  color=(0, 255, 0), thickness=max(1, round(5 * s)))

    # YOLO detections that overlap the ZOI
    for result in yolo_results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = result.names[int(box.cls[0])]
            if cv2.countNonZero(zoi_mask[y1:y2, x1:x2]) > 0:
                track_id = int(box.id[0]) if box.id is not None else None
                id_suffix = f" #{track_id}" if track_id is not None else ""
                cv2.rectangle(frame_drawn, (x1, y1), (x2, y2),
                              (0, 165, 255), max(1, round(3 * s)))
                cv2.putText(frame_drawn, f"{label}{id_suffix} IN ZOI",
                            (x1, y1 - max(4, round(10 * s))),
                            cv2.FONT_HERSHEY_SIMPLEX, max(0.3, 0.8 * s),
                            (0, 165, 255), max(1, round(2 * s)))

    # Status overlay — all dimensions proportional to frame width
    pad      = max(4, round(30 * s))
    box_w    = round(590 * s)
    box_h    = round(155 * s)
    text_x   = pad + max(2, round(20 * s))
    status_y = pad + round(70 * s)
    info_y   = pad + round(135 * s)

    cv2.rectangle(frame_drawn,
                  (pad, pad), (pad + box_w, pad + box_h),
                  (255, 255, 255), -1)
    cv2.putText(frame_drawn, f"STATUS: {status_text}",
                (text_x, status_y), cv2.FONT_HERSHEY_SIMPLEX,
                max(0.4, 2.0 * s), status_color, max(1, round(4 * s)))
    cv2.putText(frame_drawn,
                f"Frame: {frame_count}  |  Time: {current_time:.1f}s",
                (text_x, info_y), cv2.FONT_HERSHEY_SIMPLEX,
                max(0.3, 0.9 * s), (50, 50, 50), max(1, round(2 * s)))

    return frame_drawn
