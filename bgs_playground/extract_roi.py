#!/usr/bin/env python3
"""Extract the rail track-corridor ROI (region of interest) polygon.

ROI-only slice of the counting pipeline: this derives the track-corridor
polygon from a trained rail segmentation model (or loads a manually supplied
polygon JSON), then saves the polygon and a preview overlay. It does NOT do any
object detection, tracking, or enter/exit counting.

Usage:
    # Derive ROI from the rail segmentation model by scanning a video:
    python training/extract_roi.py \
        --video /path/to/video \
        --rail-model /output/rail_seg_all_rails \
        --output-dir /output/roi

    # Or use a manual polygon and just render a preview over the first frame:
    python extract_roi.py \
        --video /path/to/video \
        --zone-json my_zone.json \
        --output-dir /output/roi_still_img

    # Add --annotate-video to render the ROI over every frame.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

import cv2
import numpy as np
import torch

from infer_seg_lines import extract_zone_from_heatmap
from train_rail_seg import RailSegModel


def load_zone_json(path: Path) -> np.ndarray:
    """Load a manual ROI polygon from Nx2 points or {'vertices'|'polygon'|'points': ...}."""
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        data = data.get("vertices") or data.get("polygon") or data.get("points")
    zone = np.asarray(data, dtype=np.float32)
    if zone.ndim != 2 or zone.shape[1] != 2 or len(zone) < 3:
        raise ValueError(f"{path} must contain an Nx2 polygon or a dict with vertices/polygon/points")
    return zone


def infer_rail_heatmap(model_dir: Path, frame: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Run the rail segmentation model on one frame → (heatmap, scale_x, scale_y).

    scale_x/scale_y map heatmap pixel coords back to the original frame size.
    """
    config = json.loads((model_dir / "config.json").read_text())
    img_w, img_h = config["img_size"]
    num_classes = int(config.get("num_classes", 1))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RailSegModel(
        freeze_backbone=0,
        num_classes=num_classes,
        backbone_name=config.get("backbone", "resnet34"),
    )
    try:
        state = torch.load(model_dir / "best_model.pth", weights_only=True, map_location=device)
    except TypeError:
        state = torch.load(model_dir / "best_model.pth", map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()

    orig_h, orig_w = frame.shape[:2]
    resized = cv2.resize(frame, (img_w, img_h), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = torch.sigmoid(model(tensor))[0].cpu().numpy()
    heatmap = pred.max(axis=0)
    return heatmap, orig_w / img_w, orig_h / img_h


def detect_zone_from_rail_model(
    model_dir: Path,
    video_path: Path,
    threshold: float,
    scan_step: int,
    scan_limit: int,
) -> tuple[np.ndarray, dict]:
    """Scan video frames until the rail model yields a corridor ROI polygon."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    max_frame = min(total_frames, scan_limit) if scan_limit > 0 and total_frames > 0 else scan_limit
    if max_frame <= 0:
        max_frame = total_frames if total_frames > 0 else scan_step

    frame_index = 0
    while frame_index < max_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            break
        heatmap, sx, sy = infer_rail_heatmap(model_dir, frame)
        polygon = extract_zone_from_heatmap(heatmap, threshold=threshold)
        if polygon is not None:
            polygon = polygon.astype(np.float32)
            polygon[:, 0] *= sx
            polygon[:, 1] *= sy
            mask_values = heatmap[heatmap >= threshold]
            cap.release()
            return polygon, {
                "source": "rail_model",
                "scan_frame": frame_index,
                "threshold": threshold,
                "confidence": float(mean(mask_values.tolist())) if mask_values.size else 0.0,
            }
        frame_index += scan_step

    cap.release()
    raise RuntimeError(f"No rail corridor found in first {max_frame} frames of {video_path}")


def draw_zone_overlay(frame: np.ndarray, zone: np.ndarray) -> np.ndarray:
    """Draw the ROI polygon (translucent fill + outline) on a copy of the frame."""
    out = frame.copy()
    overlay = out.copy()
    cv2.fillPoly(overlay, [zone.astype(np.int32)], (0, 180, 255))
    cv2.addWeighted(overlay, 0.22, out, 0.78, 0, out)
    cv2.polylines(out, [zone.astype(np.int32)], True, (0, 220, 255), 2, cv2.LINE_AA)
    return out


def read_first_frame(video_path: Path) -> tuple[np.ndarray, float, int, int]:
    """Grab the first frame plus (fps, width, height) for previews/annotation."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read a frame from {video_path}")
    return frame, fps, width, height


def annotate_full_video(video_path: Path, zone: np.ndarray, out_path: Path, fps: float,
                        width: int, height: int, progress_every: int = 100) -> None:
    """Render the ROI polygon over every frame of the video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create output video: {out_path}")

    processed = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(draw_zone_overlay(frame, zone))
        processed += 1
        if progress_every and processed % progress_every == 0:
            print(f"Annotated {processed} frames")
    cap.release()
    writer.release()
    print(f"Wrote {out_path}")


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.zone_json:
        zone = load_zone_json(args.zone_json)
        zone_metadata = {"source": "zone_json", "path": str(args.zone_json)}
    elif args.rail_model:
        zone, zone_metadata = detect_zone_from_rail_model(
            args.rail_model,
            args.video,
            args.rail_threshold,
            args.scan_step,
            args.scan_limit,
        )
    else:
        raise SystemExit("Provide either --rail-model or --zone-json.")

    frame, fps, width, height = read_first_frame(args.video)

    # Save the ROI polygon.
    zone_path = args.output_dir / "track_corridor.json"
    zone_path.write_text(json.dumps({"vertices": zone.tolist(), **zone_metadata}, indent=2) + "\n")

    # Save a preview of the ROI over the first frame.
    preview_path = args.output_dir / "roi_preview.png"
    cv2.imwrite(str(preview_path), draw_zone_overlay(frame, zone))

    summary = {
        "video": str(args.video),
        "rail_model": str(args.rail_model) if args.rail_model else None,
        "frame_size": {"width": width, "height": height},
        "zone": {
            **zone_metadata,
            "vertices": zone.tolist(),
            "area_pct": float(cv2.contourArea(zone.astype(np.float32)) / max(1, width * height) * 100.0),
        },
    }
    summary_path = args.output_dir / "roi_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(json.dumps(summary["zone"], indent=2))
    print(f"Wrote {zone_path}")
    print(f"Wrote {preview_path}")
    print(f"Wrote {summary_path}")

    if args.annotate_video:
        annotate_full_video(
            args.video, zone, args.output_dir / "roi_annotated.mp4", fps, width, height
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path, help="Input video path.")
    parser.add_argument("--rail-model", type=Path, help="Rail model dir with best_model.pth and config.json.")
    parser.add_argument("--zone-json", type=Path, help="Manual ROI polygon JSON as Nx2 points or {'vertices': ...}.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--rail-threshold", type=float, default=0.35)
    parser.add_argument("--scan-step", type=int, default=30)
    parser.add_argument("--scan-limit", type=int, default=900)
    parser.add_argument("--annotate-video", action="store_true",
                        help="Also render the ROI over every frame of the video.")
    args = parser.parse_args()
    if args.rail_model and args.zone_json:
        parser.error("Use only one of --rail-model or --zone-json.")
    if not args.rail_model and not args.zone_json:
        parser.error("Provide either --rail-model or --zone-json.")
    return args


if __name__ == "__main__":
    run(parse_args())
