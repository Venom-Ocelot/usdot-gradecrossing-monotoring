# Pipeline Helper Functions
import cv2
import numpy as np

def stabilize_zoi(gray, first_gray, first_points, lk_params, zoi_points):
    """Track ZOI from frame 1 to current frame using optical flow + RANSAC."""
    curr_points, status, _ = cv2.calcOpticalFlowPyrLK(
        first_gray, gray, first_points, None, **lk_params
    )
    good_first = first_points[status == 1]
    good_curr  = curr_points[status == 1]
    if len(good_first) >= 4:
        H, _ = cv2.findHomography(good_first, good_curr, cv2.RANSAC, 3.0)
        if H is not None:
            zoi_reshaped = zoi_points.astype(np.float32).reshape(-1, 1, 2)
            return cv2.perspectiveTransform(zoi_reshaped, H).reshape(-1, 2).astype(np.int32)
    return zoi_points


def build_zoi_mask(frame_shape, zoi_stabilized):
    """Create a filled polygon mask for the ZOI."""
    zoi_mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    cv2.fillPoly(zoi_mask, [zoi_stabilized], 255)
    return zoi_mask


def apply_mog2_mask(frame, mog2):
    """Apply MOG2 and return a cleaned foreground mask."""
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = mog2.apply(frame)
    fg_mask[fg_mask == 127] = 0
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,  kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    return fg_mask


def check_vehicle_in_zoi(yolo_results, zoi_mask, frame_shape):
    """Check if any YOLO detection overlaps with the ZOI.
    Returns: vehicle_in_zoi (bool), vehicle_labels (list), exclusion_mask (array)
    """
    vehicle_in_zoi = False
    vehicle_labels = []
    exclusion_mask = np.zeros(frame_shape[:2], dtype=np.uint8)

    for result in yolo_results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = result.names[int(box.cls[0])]
            cv2.rectangle(exclusion_mask, (x1, y1), (x2, y2), 255, -1)
            bbox_mask = np.zeros(frame_shape[:2], dtype=np.uint8)
            cv2.rectangle(bbox_mask, (x1, y1), (x2, y2), 255, -1)
            if cv2.countNonZero(cv2.bitwise_and(bbox_mask, zoi_mask)) > 0:
                vehicle_in_zoi = True
                vehicle_labels.append(label)

    return vehicle_in_zoi, vehicle_labels, exclusion_mask


def get_debris_mask(fg_mask, exclusion_mask, zoi_mask):
    """Subtract YOLO exclusion mask from MOG2 mask, isolate inside ZOI."""
    clean_mask  = cv2.bitwise_and(fg_mask, cv2.bitwise_not(exclusion_mask))
    debris_mask = cv2.bitwise_and(clean_mask, zoi_mask)
    return debris_mask


def evaluate_status(debris_time, vehicle_time, vehicle_in_zoi,
                    TIME_THRESHOLD, VEHICLE_THRESHOLD):
    """Return status text and color based on timers and vehicle presence."""
    if debris_time >= TIME_THRESHOLD:
        return "ALARM",   (0, 0, 255)
    elif vehicle_time >= VEHICLE_THRESHOLD:
        return "ALARM",   (0, 0, 255)
    elif vehicle_in_zoi:
        return "WARNING", (0, 165, 255)
    else:
        return "CLEAR",   (0, 255, 0)


def draw_frame(frame, zoi_stabilized, yolo_results, zoi_mask,
               status_text, status_color, frame_count, current_time,
               debris_count, vehicle_in_zoi, vehicle_time):
    """Draw ZOI, YOLO boxes, and status overlay on the frame."""
    frame_drawn = frame.copy()
    cv2.polylines(frame_drawn, [zoi_stabilized], isClosed=True, color=(0, 255, 0), thickness=5)

    for result in yolo_results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = result.names[int(box.cls[0])]
            bbox_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.rectangle(bbox_mask, (x1, y1), (x2, y2), 255, -1)
            if cv2.countNonZero(cv2.bitwise_and(bbox_mask, zoi_mask)) > 0:
                cv2.rectangle(frame_drawn, (x1, y1), (x2, y2), (0, 165, 255), 3)
                cv2.putText(frame_drawn, f"{label} IN ZOI", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)

    cv2.putText(frame_drawn, f"STATUS: {status_text}",
                (50, 80),  cv2.FONT_HERSHEY_SIMPLEX, 2, status_color, 4)
    cv2.putText(frame_drawn, f"Frame: {frame_count} | Time: {current_time:.1f}s | Debris px: {debris_count}",
                (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame_drawn, f"Vehicle in ZOI: {vehicle_in_zoi} | Vehicle time: {vehicle_time:.1f}s",
                (50, 210), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
    return frame_drawn