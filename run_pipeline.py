#!/usr/bin/env python3
"""
run_pipeline.py — Command-line entry point for the grade-crossing monitoring pipeline.

Usage:
  python run_pipeline.py --video-id crossing_001
  python run_pipeline.py --video-id crossing_001 --vehicle-threshold 5.0 --time-threshold 2.0
"""

import argparse
from pathlib import Path

import numpy as np
from ultralytics import YOLO

import pipeline.config as config
from pipeline.io import create_run, find_video, resolve_video_path, save_run_results
from pipeline.runner import run_pipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the USDOT grade-crossing monitoring pipeline on a video file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--video-id", required=True,
                   help="Video id from data/manifests/videos.yaml.")
    p.add_argument("--model",    default=config.MODEL_PATH,
                   help="Path to YOLO .pt weights.")
    p.add_argument("--manifest", default="data/manifests/videos.yaml",
                   help="Path to the video manifest YAML.")
    p.add_argument("--area-threshold",    type=int,   default=config.AREA_THRESHOLD)
    p.add_argument("--time-threshold",    type=float, default=config.TIME_THRESHOLD)
    p.add_argument("--vehicle-threshold", type=float, default=config.VEHICLE_THRESHOLD)
    p.add_argument("--yolo-stride",       type=int,   default=config.YOLO_STRIDE)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    entry      = find_video(args.video_id, args.manifest)
    video_path = resolve_video_path(entry)
    run_paths  = create_run(entry.id)

    print(f"Video  : {video_path}")
    print(f"Model  : {args.model}")
    print(f"Run ID : {run_paths.run_id}")

    model = YOLO(args.model)

    result = run_pipeline(
        video_path=video_path,
        model=model,
        zoi_points=np.array(config.ZOI_POINTS, dtype=np.int32),
        run_label=entry.name,
        area_threshold=args.area_threshold,
        time_threshold=args.time_threshold,
        vehicle_threshold=args.vehicle_threshold,
        yolo_stride=args.yolo_stride,
        run_paths=run_paths,
    )

    save_run_results(
        result,
        run_paths,
        run_metadata={
            "video_id":   entry.id,
            "video_path": video_path,
            "model_path": args.model,
        },
        config_snapshot={
            "video_path":        video_path,
            "model_path":        args.model,
            "area_threshold":    args.area_threshold,
            "time_threshold":    args.time_threshold,
            "vehicle_threshold": args.vehicle_threshold,
            "yolo_stride":       args.yolo_stride,
        },
    )

    print(f"Done — outputs saved to: {run_paths.run_dir}")


if __name__ == "__main__":
    main()

