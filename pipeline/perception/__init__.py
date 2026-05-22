"""
Perception package — YOLO detection and MOG2 anomaly detection.
"""

from .mog2_anomaly import train_mog2, apply_mog2_mask, get_debris_mask
from .yolo_detector import run_detection, check_vehicle_in_zoi

__all__ = [
    "train_mog2",
    "apply_mog2_mask",
    "get_debris_mask",
    "run_detection",
    "check_vehicle_in_zoi",
]
