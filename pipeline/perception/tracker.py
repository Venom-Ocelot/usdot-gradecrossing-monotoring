"""
Vehicle dwell tracker — per-track-ID dwell time management inside the ZOI.

Owns the mapping of ByteTrack IDs → dwell start times. Runner calls
update() on each YOLO stride frame and reads max_dwell() every frame.
Safety decisions remain in safety_logic.py; this module only manages time.
"""

from __future__ import annotations


class VehicleTracker:
    """Tracks per-vehicle dwell time inside the ZOI using ByteTrack IDs.

    Only real ByteTrack IDs accumulate dwell time. Detections where
    box.id is None count as vehicle presence (WARNING) but do not
    create a persistent dwell timer and cannot trigger an ALARM.
    """

    def __init__(self) -> None:
        self._timers: dict[int, float] = {}  # track_id → dwell start time (seconds)

    def update(self, active_ids: set[int], current_time: float) -> None:
        """Sync timer state with the current set of in-ZOI track IDs.

        Call this once per YOLO stride frame after check_vehicle_in_zoi().
        - IDs no longer present are removed.
        - New IDs start a fresh timer at current_time.
        - Existing IDs are left untouched so their timers keep accumulating.
        """
        for tid in set(self._timers) - active_ids:
            del self._timers[tid]
        for tid in active_ids:
            if tid not in self._timers:
                self._timers[tid] = current_time

    def max_dwell(self, current_time: float) -> float:
        """Return the longest dwell time (seconds) of any tracked vehicle.

        Call this every frame. Returns 0.0 when no vehicles are tracked.
        """
        if not self._timers:
            return 0.0
        return max(current_time - t for t in self._timers.values())

    @property
    def active(self) -> bool:
        """True if at least one ByteTrack-ID'd vehicle has an active dwell timer."""
        return bool(self._timers)
