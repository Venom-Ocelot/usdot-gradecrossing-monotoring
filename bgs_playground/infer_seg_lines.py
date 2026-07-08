#!/usr/bin/env python3
"""Extract polylines from segmentation model heatmap.

Two-stage approach:
  1. Segmentation model → pixel-level rail probability heatmap
  2. Post-processing → individual polylines with vertices

The segmentation model works well (dice=0.72+) but outputs blobs.
This script converts those blobs into clean, ordered polylines.

Algorithm per connected component:
  1. Threshold heatmap → binary mask
  2. Morphological cleanup (close small gaps, remove noise)
  3. Connected components → individual rail regions
  4. For each region: skeletonize → ordered point sequence
  5. Subsample skeleton to N vertices
  6. Output as polyline

Usage:
    python infer_seg_lines.py \
        --model-dir output/rail_seg_v2 \
        --labels-dir ~/.private/comfyui/chat-outputs/_labels \
        --mask-set rails
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent))
from train_rail_seg import RailSegModel

SLOT_COLORS = [
    (0, 255, 0),     # green
    (0, 0, 255),     # red
    (0, 165, 255),   # orange
    (0, 255, 255),   # yellow
    (255, 0, 255),   # magenta
    (255, 255, 0),   # cyan
    (128, 0, 128),   # purple
    (0, 128, 255),   # dark orange
]


def skeleton_to_polyline(skeleton_mask: np.ndarray, num_verts: int = 8) -> list:
    """Convert a binary skeleton mask to an ordered polyline.

    Extracts nonzero points, orders them by tracing connectivity,
    and subsamples to num_verts evenly-spaced points.
    """
    ys, xs = np.nonzero(skeleton_mask)
    if len(xs) < 2:
        return []

    points = np.column_stack([xs, ys])

    # Order points by tracing: start from the point with min y (topmost)
    ordered = [points[np.argmin(points[:, 1])]]
    remaining = set(range(len(points)))
    remaining.discard(np.argmin(points[:, 1]))

    while remaining:
        last = ordered[-1]
        # Find nearest remaining point
        min_dist = float('inf')
        min_idx = -1
        for idx in remaining:
            d = np.sqrt((points[idx][0] - last[0]) ** 2 +
                        (points[idx][1] - last[1]) ** 2)
            if d < min_dist:
                min_dist = d
                min_idx = idx

        # If nearest is too far (gap in skeleton), stop this branch
        if min_dist > 15:
            break
        ordered.append(points[min_idx])
        remaining.discard(min_idx)

    if len(ordered) < 2:
        return []

    ordered = np.array(ordered)

    # Subsample to num_verts evenly-spaced points by arc length
    diffs = np.diff(ordered, axis=0)
    seg_lens = np.sqrt((diffs ** 2).sum(axis=1))
    cum_len = np.concatenate([[0], np.cumsum(seg_lens)])
    total_len = cum_len[-1]

    if total_len < 5:
        return []

    targets = np.linspace(0, total_len, num_verts)
    result = []
    for t in targets:
        idx = np.searchsorted(cum_len, t, side='right') - 1
        idx = min(idx, len(ordered) - 2)
        seg_start = cum_len[idx]
        seg_end = cum_len[idx + 1]
        if seg_end - seg_start < 1e-8:
            result.append(ordered[idx].tolist())
        else:
            frac = (t - seg_start) / (seg_end - seg_start)
            pt = ordered[idx] * (1 - frac) + ordered[idx + 1] * frac
            result.append(pt.tolist())

    return result


def thin_skeleton(mask: np.ndarray) -> np.ndarray:
    """Morphological thinning to get 1-pixel-wide skeleton.

    Uses Zhang-Suen thinning via iterative erosion.
    """
    skeleton = np.zeros_like(mask)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    temp = mask.copy()

    while True:
        eroded = cv2.erode(temp, element)
        opened = cv2.dilate(eroded, element)
        diff = cv2.subtract(temp, opened)
        skeleton = cv2.bitwise_or(skeleton, diff)
        temp = eroded.copy()
        if cv2.countNonZero(temp) == 0:
            break

    return skeleton


def extract_zone_from_heatmap(
    heatmap: np.ndarray,
    threshold: float = 0.4,
    dilate_frac: float = 0.045,
    min_area_frac: float = 0.004,
    simplify_frac: float = 0.012,
    convex: bool = True,
) -> np.ndarray | None:
    """Collapse a rail heatmap into a single track-corridor zone polygon.

    Rather than decoding each rail into a fragile vector polyline, this merges
    all rails into one filled corridor and returns its outline as an ordered
    polygon. The polygon is intended to be used directly as a counting *zone*
    (enter/exit), which is far more robust for "did a vehicle cross the tracks"
    than thin per-rail trip-wires.

    Steps:
      1. Threshold the heatmap into a rail mask.
      2. Dilate/close with a kernel sized relative to image width so the
         separate rails merge into one connected track-bed band.
      3. Take the external contour(s), keep significant ones, and build either
         their convex hull (default, clean trapezoid) or the merged outline.
      4. Simplify with approxPolyDP to a handful of vertices.

    Args:
        heatmap: (H, W) probability map in [0, 1].
        threshold: binarization threshold.
        dilate_frac: kernel size as a fraction of image width (rail-merge gap).
        min_area_frac: drop contours smaller than this fraction of the frame.
        simplify_frac: approxPolyDP epsilon as a fraction of contour perimeter.
        convex: if True, return the convex hull of the corridor; otherwise the
            simplified concave outline of the largest band.

    Returns:
        Nx2 float32 array of polygon vertices in heatmap pixel coords, or None
        if no rail mass is present.
    """
    h, w = heatmap.shape
    mask = (heatmap >= threshold).astype(np.uint8)
    if int(mask.sum()) == 0:
        return None

    k = max(3, int(dilate_frac * w))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    band = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    band = cv2.dilate(band, kernel, iterations=1)

    contours, _ = cv2.findContours(band, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    min_area = min_area_frac * h * w
    big = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not big:
        big = [max(contours, key=cv2.contourArea)]

    if convex:
        outline = cv2.convexHull(np.vstack(big))
    else:
        outline = max(big, key=cv2.contourArea)

    peri = cv2.arcLength(outline, True)
    poly = cv2.approxPolyDP(outline, max(1.0, simplify_frac * peri), True)
    poly = poly.reshape(-1, 2).astype(np.float32)
    if len(poly) < 3:
        return None
    return poly


def extract_lines_from_heatmap(heatmap: np.ndarray, threshold: float = 0.4,
                                min_length: int = 20, num_verts: int = 6,
                                max_displacement: float = 30,
                                max_slope_change: float = 1.0) -> list:
    """Extract polylines from a probability heatmap using strict row-wise tracking.

    Scans each row for peaks and tracks them across rows, then straightens
    the result with a linear regression fit:
      - Reduced max displacement per row-step (25px vs original 40px)
      - Slope-change limit prevents wild track jumps
      - Post-tracking linear regression fit enforces near-straight output
      - Points deviating >8px from the best-fit line are snapped to it

    Args:
        heatmap: (H, W) probability map [0, 1]
        threshold: binarization threshold
        min_length: minimum track length (in rows) to keep
        num_verts: vertices per output polyline
        max_displacement: max px a peak can be from prediction to match
        max_slope_change: max allowed change in slope (dx/dy) per step

    Returns:
        list of polylines, each a list of [x, y] vertices
    """
    H, W = heatmap.shape

    # Active tracks: each is a list of (x, y) points
    tracks = []
    active = []  # indices into tracks that are still being extended

    # Scan rows top to bottom
    sample_rows = range(0, H, 2)  # sample every 2 rows for speed
    for y in sample_rows:
        row = heatmap[y]
        binary = (row > threshold).astype(np.uint8)

        # Find contiguous regions
        peaks = []
        in_region = False
        x_start = 0
        for x in range(W):
            if binary[x] and not in_region:
                x_start = x
                in_region = True
            elif not binary[x] and in_region:
                region = row[x_start:x]
                cx = x_start + np.average(np.arange(len(region)), weights=region)
                peaks.append(cx)
                in_region = False
        if in_region:
            region = row[x_start:W]
            if len(region) > 0:
                cx = x_start + np.average(np.arange(len(region)), weights=region)
                peaks.append(cx)

        # Match peaks to active tracks (strict: displacement + slope-change)
        used_peaks = set()
        for ti in list(active):
            track = tracks[ti]
            last_x = track[-1][0]

            # Compute current slope (dx per pixel of y) from recent points
            if len(track) >= 3:
                recent = track[-3:]
                slope = (recent[-1][0] - recent[0][0]) / max(1, recent[-1][1] - recent[0][1])
                predicted_x = last_x + slope * 2
            elif len(track) >= 2:
                slope = (track[-1][0] - track[-2][0]) / max(1, track[-1][1] - track[-2][1])
                predicted_x = last_x + slope * 2
            else:
                slope = None
                predicted_x = last_x

            best_dist = float('inf')
            best_pi = -1
            for pi, px in enumerate(peaks):
                if pi in used_peaks:
                    continue
                dist = abs(px - predicted_x)
                if dist >= best_dist or dist >= max_displacement:
                    continue
                # Slope-change gate (only after we have a baseline slope)
                if slope is not None and len(track) >= 4:
                    new_slope = (px - last_x) / 2.0  # dx/dy for this 2-row step
                    if abs(new_slope - slope) > max_slope_change:
                        continue
                best_dist = dist
                best_pi = pi

            if best_pi >= 0:
                track.append((peaks[best_pi], y))
                used_peaks.add(best_pi)
            else:
                active.remove(ti)

        # Start new tracks for unmatched peaks
        for pi, px in enumerate(peaks):
            if pi not in used_peaks:
                tracks.append([(px, y)])
                active.append(len(tracks) - 1)

    # Filter tracks by length
    good_tracks = [t for t in tracks if len(t) >= min_length]

    # Strict straightening: fit a linear regression (x = m*y + b) to each
    # track and snap outlier points to the fit.
    straightened_tracks = []
    for track in good_tracks:
        pts = np.array(track, dtype=np.float64)
        xs, ys = pts[:, 0], pts[:, 1]

        y_mean = ys.mean()
        x_mean = xs.mean()
        denom = ((ys - y_mean) ** 2).sum()
        if denom < 1e-8:
            straightened_tracks.append(pts.tolist())
            continue
        m = ((ys - y_mean) * (xs - x_mean)).sum() / denom
        b = x_mean - m * y_mean
        fitted_x = m * ys + b

        max_dev = 8  # pixels — strict but allows slight curvature
        deviation = np.abs(xs - fitted_x)
        pts[:, 0] = np.where(deviation > max_dev, fitted_x, xs)

        straightened_tracks.append(pts.tolist())

    good_tracks = [[(p[0], p[1]) for p in t] for t in straightened_tracks]

    # Convert to evenly-subsampled polylines
    polylines = []
    for track in good_tracks:
        pts = np.array(track)

        diffs = np.diff(pts, axis=0)
        seg_lens = np.sqrt((diffs ** 2).sum(axis=1))
        cum_len = np.concatenate([[0], np.cumsum(seg_lens)])
        total_len = cum_len[-1]

        if total_len < 10:
            continue

        targets = np.linspace(0, total_len, num_verts)
        resampled = []
        for t in targets:
            idx = np.searchsorted(cum_len, t, side='right') - 1
            idx = min(idx, len(pts) - 2)
            seg_start = cum_len[idx]
            seg_end = cum_len[idx + 1]
            if seg_end - seg_start < 1e-8:
                resampled.append(pts[idx].tolist())
            else:
                frac = (t - seg_start) / (seg_end - seg_start)
                pt = pts[idx] * (1 - frac) + pts[idx + 1] * frac
                resampled.append(pt.tolist())

        polylines.append(resampled)

    polylines.sort(key=lambda p: np.mean([pt[0] for pt in p]))

    return polylines


def _draw_dotted_line(img, pt1, pt2, color, thickness=2, gap=10):
    """Draw a dotted/dashed line between two points."""
    x1, y1 = pt1
    x2, y2 = pt2
    dx = x2 - x1
    dy = y2 - y1
    length = max(1, int(np.sqrt(dx * dx + dy * dy)))
    dash_len = gap
    num_segments = max(1, length // (dash_len * 2))
    for i in range(num_segments):
        t0 = (2 * i * dash_len) / length
        t1 = min(((2 * i + 1) * dash_len) / length, 1.0)
        p0 = (int(x1 + dx * t0), int(y1 + dy * t0))
        p1 = (int(x1 + dx * t1), int(y1 + dy * t1))
        cv2.line(img, p0, p1, color, thickness, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description="Segmentation → polyline extraction")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--labels-dir", required=True)
    parser.add_argument("--predictions-dir", default=None,
                        help="Destination root for predictions "
                             "(default: sibling 'label_predictions' next to --labels-dir)")
    parser.add_argument("--mask-set", default="rails")
    parser.add_argument("--output-set", default=None,
                        help="Write predictions to this mask set (default: same as --mask-set)")
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--num-verts", type=int, default=6,
                        help="Vertices per extracted polyline")
    parser.add_argument("--min-length", type=int, default=15,
                        help="Minimum track length (rows) to keep")
    parser.add_argument("--max-lines-per-channel", type=int, default=1,
                        help="Max lines to keep per class channel")
    parser.add_argument("--label", default=None,
                        help="Override class name in output (e.g. 'track')")
    parser.add_argument("--clean", action="store_true",
                        help="Minimalist rendering: no per-segment distance labels, "
                             "smaller vertex dots. Keeps confidence label.")
    parser.add_argument("--images-dir", default=None,
                        help="If set, iterate over every image in this dir; "
                             "write predictions to <predictions_dir>/<image_stem>/<output_set>/. "
                             "Used to produce overlays for the full label-studio image pool.")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    labels_dir = Path(args.labels_dir)
    predictions_dir = Path(args.predictions_dir) if args.predictions_dir else (labels_dir.parent / "label_predictions")
    predictions_dir.mkdir(parents=True, exist_ok=True)

    config = json.loads((model_dir / "config.json").read_text())
    img_w, img_h = config["img_size"]
    num_classes = config.get("num_classes", 1)
    classes = config.get("classes", None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, img={img_w}x{img_h}, classes={classes or ['all']}")
    print(f"Predictions root: {predictions_dir}")

    model = RailSegModel(freeze_backbone=config.get("freeze_backbone", 4),
                         num_classes=num_classes)
    model.load_state_dict(
        torch.load(model_dir / "best_model.pth", weights_only=True, map_location=device)
    )
    model.to(device).eval()

    # Per-class colors (BGR + alpha)
    CLASS_COLORS = {
        "left_track": (0, 0, 255),    # red
        "right_track": (0, 255, 0),    # green
        "ego_track": (255, 180, 0),    # cyan-blue
    }

    count = 0
    total_lines = 0
    image_iter = []
    seen_ids = set()
    if args.images_dir:
        imgs_dir = Path(args.images_dir)
        EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        for p in sorted(imgs_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in EXT:
                image_iter.append((p.stem, p))
                seen_ids.add(p.stem)
        print(f"Pool images from {imgs_dir}: {len(image_iter)}")

    # Also include any label-studio item dirs that have their own image.png
    # (items not present in the shared pool — e.g. earlier ds-dataset uploads).
    extra = 0
    for d in sorted(labels_dir.iterdir()):
        if not d.is_dir() or d.name.startswith("_") or d.name in seen_ids:
            continue
        src = d / "image.png"
        if not src.exists():
            for ext in [".jpg", ".jpeg", ".webp"]:
                p = d / f"image{ext}"
                if p.exists():
                    src = p
                    break
        if src.exists():
            image_iter.append((d.name, src))
            extra += 1
    if extra:
        print(f"Plus {extra} item dirs with their own image.png")
    total = len(image_iter)
    print(f"Running inference on {total} images total...")

    for item_id, src in image_iter:
        img = cv2.imread(str(src))
        if img is None:
            continue
        orig_h, orig_w = img.shape[:2]

        # Run model
        resized = cv2.resize(img, (img_w, img_h), interpolation=cv2.INTER_AREA)
        inp = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        inp = torch.from_numpy(inp.transpose(2, 0, 1)).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = torch.sigmoid(model(inp))[0].cpu().numpy()  # (C, H, W)

        # Extract polylines per channel, with confidence scores
        overlay = np.zeros((orig_h, orig_w, 4), dtype=np.uint8)
        sx = orig_w / img_w
        sy = orig_h / img_h
        all_lines = {}
        scored_lines = []  # (confidence, cls_name, scaled_pts, color)

        for ch in range(num_classes):
            cls_name = classes[ch] if classes else f"class_{ch}"
            heatmap = pred[ch] if num_classes > 1 else pred[0]

            polylines = extract_lines_from_heatmap(
                heatmap, threshold=args.threshold,
                min_length=args.min_length, num_verts=args.num_verts
            )

            # Score each line by average heatmap value along its track
            scored = []
            for poly in polylines:
                vals = [heatmap[min(int(p[1]), img_h-1), min(int(p[0]), img_w-1)]
                        for p in poly]
                scored.append((float(np.mean(vals)), poly))
            scored.sort(key=lambda x: -x[0])

            # Keep top N per channel
            scored = scored[:args.max_lines_per_channel]

            color = CLASS_COLORS.get(cls_name, SLOT_COLORS[ch % len(SLOT_COLORS)])

            scaled_polys = []
            for conf, poly in scored:
                pts = [(int(p[0] * sx), int(p[1] * sy)) for p in poly]
                scaled_polys.append(pts)
                scored_lines.append((conf, cls_name, pts, color))

            all_lines[cls_name] = [[[p[0], p[1]] for p in poly] for poly in scaled_polys]

        # Find the highest confidence line
        if scored_lines:
            best_conf = max(sl[0] for sl in scored_lines)
        else:
            best_conf = 0.0

        # Draw: solid for highest confidence, dotted for the rest
        for conf, cls_name, pts, color in scored_lines:
            is_best = (conf >= best_conf - 1e-6)
            alpha = 220 if is_best else 180
            if is_best:
                for i in range(len(pts) - 1):
                    cv2.line(overlay, pts[i], pts[i + 1],
                             color + (alpha,), 3, cv2.LINE_AA)
            else:
                for i in range(len(pts) - 1):
                    _draw_dotted_line(overlay, pts[i], pts[i + 1],
                                      color + (alpha,), thickness=3, gap=12)

            # Vertices + (optional) node-to-node distance labels
            dist_font = cv2.FONT_HERSHEY_SIMPLEX
            dist_scale = max(0.35, orig_w / 2400)
            dist_thick = max(1, int(orig_w / 1400))
            vradius_best = 3 if args.clean else 6
            vradius_alt  = 2 if args.clean else 5
            for i, pt in enumerate(pts):
                if is_best:
                    cv2.circle(overlay, pt, vradius_best, color + (alpha,), -1, cv2.LINE_AA)
                else:
                    cv2.circle(overlay, pt, vradius_alt, color + (alpha,), 2, cv2.LINE_AA)

                if args.clean:
                    continue  # skip distance labels

                # Distance label on each segment
                if i < len(pts) - 1:
                    dx = pts[i + 1][0] - pt[0]
                    dy = pts[i + 1][1] - pt[1]
                    seg_dist = int(round((dx * dx + dy * dy) ** 0.5))
                    mx = (pt[0] + pts[i + 1][0]) // 2
                    my = (pt[1] + pts[i + 1][1]) // 2
                    dlabel = f"{seg_dist}px"
                    (tw, th), _ = cv2.getTextSize(dlabel, dist_font, dist_scale, dist_thick)
                    dpad = int(3 * dist_scale + 1)
                    seg_len = max(1, seg_dist)
                    nx, ny = -dy / seg_len, dx / seg_len
                    off = 12
                    lx = int(mx + nx * off) - tw // 2
                    ly = int(my + ny * off) + th // 2
                    lx = max(0, min(lx, orig_w - tw - dpad * 2))
                    ly = max(th + dpad, min(ly, orig_h - dpad))
                    cv2.rectangle(overlay,
                                  (lx - dpad, ly - th - dpad),
                                  (lx + tw + dpad, ly + dpad),
                                  (0, 0, 0, 160), -1)
                    cv2.putText(overlay, dlabel, (lx, ly),
                                dist_font, dist_scale, (255, 255, 255, 220),
                                dist_thick, cv2.LINE_AA)

            # Confidence label near the midpoint of the line
            if len(pts) >= 2:
                mid_idx = len(pts) // 2
                mx, my = pts[mid_idx]
                label = f"{conf:.0%}"
                if not args.label:
                    label = f"{cls_name.replace('_track','').replace('_rail','')} {label}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = max(0.6, orig_w / 900)
                thickness_txt = max(1, int(orig_w / 700))
                (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness_txt)
                pad = int(6 * font_scale)
                tx, ty = mx + 14, my - 10
                tx = min(tx, orig_w - tw - pad * 2)
                ty = max(ty, th + pad * 2)
                cv2.rectangle(overlay,
                              (tx - pad, ty - th - pad),
                              (tx + tw + pad, ty + pad + baseline),
                              (0, 0, 0, 200), -1)
                cv2.putText(overlay, label, (tx, ty),
                            font, font_scale, color + (255,), thickness_txt, cv2.LINE_AA)

        img_line_count = len(scored_lines)

        total_lines += img_line_count

        # Save prediction artifacts (separate from labels store)
        output_set = args.output_set or args.mask_set
        set_dir = predictions_dir / item_id / output_set
        set_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(set_dir / "prediction.png"), overlay)

        # Save predicted polylines with confidence
        json_out = {}
        for i, (conf, cls_name, pts, color) in enumerate(scored_lines):
            out_name = f"{args.label}_{i}" if args.label else cls_name
            json_out[out_name] = {
                "confidence": round(conf, 3),
                "vertices": [[p[0], p[1]] for p in pts],
            }
        (set_dir / "predicted_lines.json").write_text(json.dumps(json_out, indent=2))

        count += 1
        if args.images_dir and count % 500 == 0:
            print(f"  {count}/{total}")

    if count == 0:
        print("No predictions generated.")
    else:
        print(f"Generated {count} predictions, {total_lines} total lines "
              f"({total_lines/count:.1f} lines/image)")


if __name__ == "__main__":
    main()
