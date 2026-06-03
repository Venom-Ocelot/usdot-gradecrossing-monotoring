"""
YOLO detection utilities — model inference and ZOI overlap checking.
"""

import cv2
import numpy as np

from .. import config


def run_detection(model, frame: np.ndarray) -> list:
    """Run YOLO tracking on a single frame.

    Returns the raw Ultralytics results list.
    """
    return model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False,
        conf=config.YOLO_CONF,
    )


def check_vehicle_in_zoi(
    yolo_results: list,
    zoi_mask: np.ndarray,
    frame_shape: tuple,
) -> tuple[bool, np.ndarray]:
    """Check whether any YOLO detection overlaps with the ZOI.

    Returns:
        vehicle_in_zoi  — True if at least one detection overlaps the ZOI
        exclusion_mask  — binary mask covering all YOLO bounding boxes
    """
    vehicle_in_zoi = False
    exclusion_mask = np.zeros(frame_shape[:2], dtype=np.uint8)

    for result in yolo_results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(exclusion_mask, (x1, y1), (x2, y2), 255, -1)
            if cv2.countNonZero(zoi_mask[y1:y2, x1:x2]) > 0:
                vehicle_in_zoi = True

    return vehicle_in_zoi, exclusion_mask
