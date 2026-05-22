"""
MOG2 background subtraction — background training and debris detection.
"""

import cv2
import numpy as np

from .. import config

_MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def train_mog2(video_path: str) -> cv2.BackgroundSubtractorMOG2:
    """Train a MOG2 subtractor on the first MOG2_HISTORY frames of the video.

    Returns the trained subtractor ready for inference.
    """
    mog2 = cv2.createBackgroundSubtractorMOG2(
        history=config.MOG2_HISTORY,
        varThreshold=config.MOG2_VAR_THRESHOLD,
        detectShadows=config.MOG2_DETECT_SHADOWS,
    )
    cap = cv2.VideoCapture(video_path)
    trained = 0
    while trained < config.MOG2_HISTORY:
        ret, frame = cap.read()
        if not ret:
            break
        mog2.apply(frame)
        trained += 1
    cap.release()
    return mog2


def apply_mog2_mask(frame: np.ndarray, mog2: cv2.BackgroundSubtractorMOG2) -> np.ndarray:
    """Apply MOG2 to a frame and return a cleaned binary foreground mask.

    Shadows (value 127) are suppressed, then morphological open/close
    remove noise and fill small holes.
    """
    fg_mask = mog2.apply(frame)
    fg_mask[fg_mask == 127] = 0
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,  _MORPH_KERNEL)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, _MORPH_KERNEL)
    return fg_mask


def get_debris_mask(
    fg_mask: np.ndarray,
    exclusion_mask: np.ndarray,
    zoi_mask: np.ndarray,
) -> np.ndarray:
    """Return the debris mask: MOG2 foreground minus YOLO detections, inside ZOI only."""
    clean_mask  = cv2.bitwise_and(fg_mask, cv2.bitwise_not(exclusion_mask))
    debris_mask = cv2.bitwise_and(clean_mask, zoi_mask)
    return debris_mask
