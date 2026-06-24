"""Blob analysis for rail grade-crossing monitoring.

Takes a binary foreground mask (e.g. the output of background subtraction in
``playground.py``) and turns the white pixels into discrete *blobs* — connected
regions with an area, centroid, and bounding box — then filters them by size.

Why ``connectedComponentsWithStats`` instead of ``cv2.SimpleBlobDetector``
(used in ``playground.py``):
  - It returns area, centroid and bounding box for every region in one pass,
    which is exactly what downstream tracking logic needs.
  - SimpleBlobDetector is tuned for dot-like blobs and only hands back
    keypoints (center + diameter), so a long vehicle on the tracks gets a
    poor circular approximation.

The core functions take a mask and stay independent of *which* background
subtractor produced it; the demo below wires them to the SuBSENSE subtractor
from ``playground.py``.
"""

import os
from dataclasses import dataclass

import cv2
import pybgs as bgs


@dataclass
class Blob:
    """A single connected foreground region.

    Attributes:
        label:    connected-component id (1..N; 0 is always the background).
        area:     foreground pixel count of the region.
        centroid: (x, y) center of mass, in full-resolution pixel coords.
        bbox:     (x, y, w, h) axis-aligned bounding box.
    """

    label: int
    area: int
    centroid: tuple
    bbox: tuple


def clean_mask(mask, open_ksize=3, close_ksize=7):
    """Morphologically denoise a binary foreground mask.

    Opening removes salt-and-pepper speckle (a few stray foreground pixels);
    closing then fills small holes so one object stays a single blob instead of
    fragmenting into several. Returns a uint8 mask with values in {0, 255}.
    """
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # Force a clean binary image; some subtractors emit shadow pixels as 127.
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    if open_ksize:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    if close_ksize:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return binary


def detect_blobs(mask, min_area=None, max_area=None, clean=True):
    """Extract blobs from a binary foreground mask.

    Args:
        mask:     binary (or BGR) foreground mask.
        min_area: drop regions smaller than this (px). Filters camera noise.
        max_area: drop regions larger than this (px), or None for no cap.
                  Useful for rejecting lighting/whole-frame flicker.
        clean:    apply ``clean_mask`` first (morphological denoise).

    Returns:
        (blobs, cleaned_mask) where blobs is a list of ``Blob`` sorted by
        descending area, and cleaned_mask is the mask the stats were computed
        on (handy for visualization/debugging).
    """
    binary = clean_mask(mask) if clean else mask
    if binary.ndim == 3:
        binary = cv2.cvtColor(binary, cv2.COLOR_BGR2GRAY)

    # One pass gives labels + per-component area/bbox/centroid.
    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    blobs = []
    for label in range(1, num_labels):  # 0 is the background component
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        if max_area is not None and area > max_area:
            continue

        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        cx, cy = centroids[label]

        blobs.append(Blob(
            label=label,
            area=area,
            centroid=(float(cx), float(cy)),
            bbox=(x, y, w, h),
        ))

    blobs.sort(key=lambda b: b.area, reverse=True)
    return blobs, binary


def draw_blobs(frame, blobs):
    """Annotate a frame with blob boxes, centroids, and areas.

    Returns the same frame (annotated in place) for convenience.
    """
    for blob in blobs:
        x, y, w, h = blob.bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cx, cy = blob.centroid
        cv2.circle(frame, (int(cx), int(cy)), 3, (0, 255, 0), -1)
        cv2.putText(frame, f"{blob.area}", (x, max(0, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    return frame


def analyze_video(video_path, min_area=None, max_area=None, display=True,
                  output_path=None):
    """Run BGS + blob analysis over a video.

    Uses the SuBSENSE subtractor from ``playground.py``. The core blob
    functions are subtractor-agnostic, so swapping ``algorithm`` for another
    ``pybgs`` method is a one-line change.

    Args:
        output_path: if given, write the annotated frames to this path (any
                     directory you choose — parent folders are created if
                     missing). Blobs are drawn whether or not ``display`` is
                     True, so the saved video always carries the annotations.

    Press Esc to quit when ``display`` is True.
    """
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    #uncomment either or to try algorithms
    #algorithm = bgs.SuBSENSE()
    #algorithm = bgs.ViBe()
    writer = None
    if output_path:
        out_dir = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(out_dir, exist_ok=True)
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            raise ValueError(f"Could not open video writer for: {output_path}")

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        fg_mask = algorithm.apply(frame)
        blobs, cleaned = detect_blobs(fg_mask, min_area=min_area, max_area=max_area)

        draw_blobs(frame, blobs)

        if writer is not None:
            writer.write(frame)

        if display:
            cv2.imshow("Blob Analysis", frame)
            cv2.imshow("Foreground (cleaned)", cleaned)
            if cv2.waitKey(10) & 0xFF == 27:
                print("Exiting...")
                break

    capture.release()
    if writer is not None:
        writer.release()
        print(f"Saved annotated video to: {output_path}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    analyze_video(
        "/home/gaelmarquez/usdot-gradecrossing-monotoring/bgs_playground/myData/clip_08.mp4",
        min_area=3900,
        max_area=7800,
        display=True,
        output_path="/home/gaelmarquez/usdot-gradecrossing-monotoring/bgs_playground/SuBSENSEBGS_output/clip_08_annotated.mp4",
    )
