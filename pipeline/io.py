"""
pipeline/io.py — Input/output management for the grade-crossing monitoring pipeline.

Responsibilities:
  - Load video manifests (data/manifests/videos.yaml)
  - Resolve video paths for local, Google Drive, and OneDrive source types
  - Create per-run output directories under outputs/runs/
  - Save run results (CSV, JSON, YAML, annotated video snapshots) to the run folder
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class VideoEntry:
    id: str
    source_type: str                        # "local" | "google_drive" | "onedrive"
    source: str                             # local path or remote URL
    local_cache_path: str                   # where the cached copy lives under data/cache/
    vehicle_threshold: float  # seconds vehicle must dwell in ZOI before ALARM
    time_threshold: float     # seconds debris must persist before ALARM


@dataclass
class RunPaths:
    """All file/directory paths for a single pipeline run. Pass to run_pipeline()
    so every output is written to the correct isolated location."""
    run_id: str
    run_dir: Path
    annotated_video: Path
    detections_csv: Path
    alerts_csv: Path
    metrics_json: Path
    run_metadata_json: Path
    config_used_yaml: Path
    debug_frames_dir: Path


# ── Manifest ──────────────────────────────────────────────────────────────────

def load_manifest(manifest_path: str | Path = "data/manifests/videos.yaml") -> list[VideoEntry]:
    """Load and parse the video manifest YAML file into a list of VideoEntry objects."""
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    with open(path) as f:
        raw = yaml.safe_load(f)
    return [VideoEntry(**entry) for entry in raw.get("videos", [])]


def find_video(video_id: str, manifest_path: str | Path = "data/manifests/videos.yaml") -> VideoEntry:
    """Return the VideoEntry matching *video_id*. Raises KeyError if not found."""
    for entry in load_manifest(manifest_path):
        if entry.id == video_id:
            return entry
    raise KeyError(f"Video id '{video_id}' not found in manifest: {manifest_path}")


# ── Video path resolution ─────────────────────────────────────────────────────

def _download_google_drive(entry: VideoEntry) -> str:
    """Download a Google Drive video to the local cache path using gdown."""
    import gdown

    cache = Path(entry.local_cache_path)
    cache.parent.mkdir(parents=True, exist_ok=True)
    gdown.download(entry.source, str(cache), quiet=False, fuzzy=True)
    if not cache.exists() or cache.stat().st_size < 1024:
        raise RuntimeError(
            f"gdown download for '{entry.id}' appears incomplete — "
            f"file missing or smaller than 1 KB at '{cache}'. "
            f"Delete the file (if it exists) and retry."
        )
    return str(cache)


def _download_onedrive(entry: VideoEntry) -> str:
    """Placeholder: download a OneDrive video to the local cache path."""
    raise NotImplementedError(
        f"OneDrive download is not yet implemented for '{entry.id}'. "
        f"Manually download from '{entry.source}' and place the file at "
        f"'{entry.local_cache_path}'."
    )


def resolve_video_path(entry: VideoEntry) -> str:
    """
    Resolve the local file path for a VideoEntry.

    Resolution order:
      1. If a cached copy already exists at *local_cache_path*, reuse it.
      2. For source_type == "local", use *source* directly.
      3. For source_type == "google_drive" or "onedrive", call the (placeholder)
         download helper which will fetch and store the file at *local_cache_path*.
    """
    cache = Path(entry.local_cache_path)
    if cache.exists():
        return str(cache)

    if entry.source_type == "local":
        src = Path(entry.source)
        if not src.exists():
            raise FileNotFoundError(
                f"Local video '{src}' not found for entry '{entry.id}'."
            )
        return str(src)

    if entry.source_type == "google_drive":
        return _download_google_drive(entry)

    if entry.source_type == "onedrive":
        return _download_onedrive(entry)

    raise ValueError(f"Unknown source_type '{entry.source_type}' for entry '{entry.id}'.")


# ── Run folder creation ───────────────────────────────────────────────────────

def create_run(video_id: str, base_dir: str | Path = "outputs/runs") -> RunPaths:
    """
    Create a unique, isolated run folder for one video processing run.

    Folder name: <video_id>_<YYYY-MM-DD>_<HHMM>
    All subdirectories are created immediately so callers can write to them
    without any extra setup.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_id = f"{video_id}_{timestamp}"
    run_dir = Path(base_dir) / run_id

    debug_frames_dir = run_dir / "debug_frames"

    debug_frames_dir.mkdir(parents=True, exist_ok=True)

    return RunPaths(
        run_id=run_id,
        run_dir=run_dir,
        annotated_video=run_dir / "annotated_video.mp4",
        detections_csv=run_dir / "detections.csv",
        alerts_csv=run_dir / "alerts.csv",
        metrics_json=run_dir / "metrics.json",
        run_metadata_json=run_dir / "run_metadata.json",
        config_used_yaml=run_dir / "config_used.yaml",
        debug_frames_dir=debug_frames_dir,
    )


# ── Result saving ─────────────────────────────────────────────────────────────

def save_run_results(
    result: Any,
    run_paths: RunPaths,
    run_metadata: dict[str, Any] | None = None,
    config_snapshot: dict[str, Any] | None = None,
) -> None:
    """
    Write all run outputs to the run folder.

    Args:
        result:          PipelineResult returned by run_pipeline().
        run_paths:       RunPaths returned by create_run().
        run_metadata:    Optional dict merged into run_metadata.json
                         (e.g. video_id, video_path, model_path).
        config_snapshot: Optional dict saved as config_used.yaml
                         (e.g. all threshold / stride values used).
    """
    # detections.csv — every in-ZOI detection with time, class, confidence
    with open(run_paths.detections_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "class_name", "confidence"])
        writer.writerows(result.confidence_log)

    # alerts.csv — frames where status == ALARM
    with open(run_paths.alerts_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "status"])
        writer.writerows((t, s) for t, s in result.status_log if s == "ALARM")

    # metrics.json — summary statistics
    metrics: dict[str, Any] = {
        "run_id": run_paths.run_id,
        "total_frames": result.total_frames,
        "alarm_first_triggered_s": result.alarm_first_triggered,
        "status_counts": dict(result.status_counts),
    }
    if result.confidence_log:
        confs = [c for _, _, c in result.confidence_log]
        metrics["detection_count"]  = len(confs)
        metrics["avg_confidence"]   = round(sum(confs) / len(confs), 4)
        metrics["min_confidence"]   = round(min(confs), 4)
        metrics["max_confidence"]   = round(max(confs), 4)
    else:
        metrics["detection_count"] = 0

    with open(run_paths.metrics_json, "w") as f:
        json.dump(metrics, f, indent=2)

    # run_metadata.json
    meta: dict[str, Any] = {
        "run_id": run_paths.run_id,
        "run_label": result.run_label,
        "created_at": datetime.now().isoformat(),
    }
    if run_metadata:
        meta.update(run_metadata)

    with open(run_paths.run_metadata_json, "w") as f:
        json.dump(meta, f, indent=2)

    # config_used.yaml — exact config values used in this run
    if config_snapshot:
        with open(run_paths.config_used_yaml, "w") as f:
            yaml.dump(config_snapshot, f, default_flow_style=False, sort_keys=False)

    # debug_frames/ — save event snapshots (in-memory RGB arrays from runner)
    if result.snapshots:
        try:
            import cv2
            for label, frame_rgb in result.snapshots.items():
                safe_label = label.replace(" ", "_").replace("/", "-").replace(":", "")
                out_path = run_paths.debug_frames_dir / f"{safe_label}.png"
                cv2.imwrite(str(out_path), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        except Exception:
            pass  # debug frames are best-effort; never block a run

    # debug_frames/ — save debris mask at each event moment for diagnostics
    if result.debris_snapshots:
        try:
            import cv2
            for label, mask in result.debris_snapshots.items():
                safe_label = label.replace(" ", "_").replace("/", "-").replace(":", "")
                out_path = run_paths.debug_frames_dir / f"{safe_label}_debris.png"
                cv2.imwrite(str(out_path), mask)
        except Exception:
            pass  # best-effort; never block a run
