"""
Safety logic — alarm status evaluation.
"""

from . import config


def evaluate_status(
    debris_time: float,
    vehicle_time: float,
    vehicle_in_zoi: bool,
    time_threshold: float = None,
    vehicle_threshold: float = None,
) -> tuple[str, tuple[int, int, int]]:
    """Return (status_text, bgr_color) based on current timer state.

    Priority order:
      1. Vehicle inside ZOI AND dwell time >= vehicle_threshold → ALARM
      2. Debris persisting >= time_threshold → ALARM
      3. Vehicle inside ZOI (dwell below threshold) → WARNING
      4. Otherwise → CLEAR

    threshold arguments default to config values when not supplied.
    """
    t_thresh = time_threshold    if time_threshold    is not None else config.TIME_THRESHOLD
    v_thresh = vehicle_threshold if vehicle_threshold is not None else config.VEHICLE_THRESHOLD

    if vehicle_in_zoi and vehicle_time >= v_thresh:
        return "ALARM",   (0, 0, 255)
    elif debris_time >= t_thresh:
        return "ALARM",   (0, 0, 255)
    elif vehicle_in_zoi:
        return "WARNING", (0, 165, 255)
    else:
        return "CLEAR",   (0, 255, 0)
