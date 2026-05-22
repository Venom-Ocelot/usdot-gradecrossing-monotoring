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
    frame_drawn = frame.copy()

    # ZOI polygon
    cv2.polylines(frame_drawn, [zoi_stabilized], isClosed=True, color=(0, 255, 0), thickness=5)

    # YOLO detections that overlap the ZOI
    for result in yolo_results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = result.names[int(box.cls[0])]
            if cv2.countNonZero(zoi_mask[y1:y2, x1:x2]) > 0:
                cv2.rectangle(frame_drawn, (x1, y1), (x2, y2), (0, 165, 255), 3)
                cv2.putText(frame_drawn, f"{label} IN ZOI", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)

    # Status overlay 
    cv2.rectangle(frame_drawn, (30, 20), (620, 175), (255, 255, 255), -1)

    cv2.putText(frame_drawn, f"STATUS: {status_text}",
                (50, 90),  cv2.FONT_HERSHEY_SIMPLEX, 2, status_color, 4)
    cv2.putText(frame_drawn,
                f"Frame: {frame_count}  |  Time: {current_time:.1f}s",
                (50, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 50, 50), 2)

    return frame_drawn
