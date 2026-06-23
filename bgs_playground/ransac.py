import cv2
import numpy as np
from skimage.measure import LineModelND, ransac

def calculate_ransac_roi(frame, min_samples=10, residual_threshold=5.0, max_trials=1000,
                         scale=0.5, max_points=5000):
    """
    Core math function: Runs the double-pass RANSAC on a single frame.
    Returns the ROI polygon points (in full-resolution coordinates).

    Efficiency:
      - `scale` < 1.0 runs edge detection + RANSAC on a downscaled frame
        (~1/scale^2 fewer edge points) and rescales the result back up.
      - `max_points` caps how many edge points RANSAC sees; RANSAC computes
        residuals across every point on every trial, so this bounds the cost.
    """
    # Work on a downscaled copy: rails are long straight lines, so half-res
    # edge detection finds them just as well for a fraction of the points.
    small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    y_coords, x_coords = np.where(edges > 0)
    data = np.column_stack([x_coords, y_coords])

    if len(data) < min_samples * 2:
        return None

    # Subsample edge points to a fixed budget so RANSAC cost stays bounded
    # regardless of how cluttered the scene is.
    if len(data) > max_points:
        idx = np.random.choice(len(data), max_points, replace=False)
        data = data[idx]

    # First Pass
    model_1, inliers_1 = ransac(data, LineModelND, min_samples=min_samples,
                                residual_threshold=residual_threshold, max_trials=max_trials)

    outliers_1 = data[~inliers_1]
    if len(outliers_1) < min_samples:
        return None

    # Second Pass
    model_2, inliers_2 = ransac(outliers_1, LineModelND, min_samples=min_samples,
                                residual_threshold=residual_threshold, max_trials=max_trials)

    # Generate Polygon boundaries (in downscaled coordinates)
    height, width = small.shape[:2]
    y_bottom = height
    y_top = height // 3
    
    def get_x_for_y(model, target_y):
        origin, direction = model.origin, model.direction
        if direction[1] == 0: 
            return int(origin[0])
        t = (target_y - origin[1]) / direction[1]
        return int(origin[0] + t * direction[0])

    x1_bottom = get_x_for_y(model_1, y_bottom)
    x1_top = get_x_for_y(model_1, y_top)
    x2_bottom = get_x_for_y(model_2, y_bottom)
    x2_top = get_x_for_y(model_2, y_top)
    
    if x1_bottom < x2_bottom:
        roi_pts = np.array([[x1_bottom, y_bottom], [x1_top, y_top],
                            [x2_top, y_top], [x2_bottom, y_bottom]], np.float32)
    else:
        roi_pts = np.array([[x2_bottom, y_bottom], [x2_top, y_top],
                            [x1_top, y_top], [x1_bottom, y_bottom]], np.float32)

    # Rescale polygon from downscaled coordinates back to full resolution.
    roi_pts = (roi_pts / scale).astype(np.int32)
    return roi_pts


def process_crossing_video(input_path, output_path):
    """
    Video pipeline: Loads video, calculates ROI once, and exports an annotated video.
    """
    # 1. Initialize Video Capture
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {input_path}")
    
    # 2. Get Video Properties for the Writer
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 3. Initialize Video Writer (mp4v codec is standard and reliable)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Processing video: {width}x{height} at {fps} FPS. Total frames: {total_frames}")
    
    frame_count = 0
    saved_roi_pts = None
    roi_mask = None  # precomputed boolean mask of the ROI (the polygon never changes)
    tint = np.array([0, 0, 255], dtype=np.float32)  # red, BGR

    # 4. The Processing Loop
    while cap.isOpened():
        ret, frame = cap.read()

        # 'ret' is False when the video ends
        if not ret:
            break

        # Calculate ROI on the very first frame only
        if frame_count == 0:
            print("Calculating RANSAC ROI on frame 0...")
            saved_roi_pts = calculate_ransac_roi(frame, residual_threshold=8.0)

            if saved_roi_pts is None:
                print("WARNING: Could not find tracks in the first frame. ROI will be empty.")
            else:
                # Precompute the fill mask once; reused for every frame below.
                fill = np.zeros((height, width), dtype=np.uint8)
                cv2.fillPoly(fill, [saved_roi_pts], 255)
                roi_mask = fill.astype(bool)

        # 5. Draw the ROI on every frame — blend only the ROI pixels (no full
        #    frame copy / whole-frame addWeighted needed since the ROI is fixed).
        if roi_mask is not None:
            frame[roi_mask] = (frame[roi_mask] * 0.7 + tint * 0.3).astype(np.uint8)
            cv2.polylines(frame, [saved_roi_pts], isClosed=True, color=(0, 255, 255), thickness=3)

        # Write the annotated frame to the new video file
        writer.write(frame)

        frame_count += 1
        
        # Simple progress indicator
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} / {total_frames} frames...")

    # 6. Clean up resources
    cap.release()
    writer.release()
    print(f"Done! Saved annotated video to: {output_path}")

# --- Execution ---
# Replace these strings with your actual file names. NOTE: the input must be a
# single video FILE, not a directory — cv2.VideoCapture cannot open a folder.
if __name__ == "__main__":
    process_crossing_video(
        "/home/gaelmarquez/usdot-gradecrossing-monotoring/bgs_playground/myData/clip_08.mp4",
        "Output_roi_video_08.mp4",
    )