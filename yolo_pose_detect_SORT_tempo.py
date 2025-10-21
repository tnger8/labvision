import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import time
import csv
from sort import Sort  # make sure sort.py is in the same folder
import os

# ===============================
# CONFIGURATION
# ===============================
POSE_MODEL_PATH = "yolov8n-pose.pt"
OBJECT_MODEL_PATH = "yolov8n.pt"
LOG_FILE = "activity_log.csv"

HAND_NEAR_MOUTH_THRESHOLD = 0.55
VIOLATION_FRAMES = 10
WINDOW_SIZE = 20
ELBOW_ANGLE_MIN = 40
ELBOW_ANGLE_MAX = 160

OBJECTS_OF_INTEREST = ['apple','banana','cup','bottle','orange']

# Long-eating snapshot config
SNAPSHOT_FOLDER = "snapshots"
os.makedirs(SNAPSHOT_FOLDER, exist_ok=True)
LONG_EATING_FRAMES = 50  # number of frames to consider as long eating
ENABLE_LONG_EATING_SNAPSHOT = True  # <-- turn on/off

# ===============================
# TEMPORAL CLASSIFIER
# ===============================
class TemporalGestureClassifier:
    def __init__(self, window_size=WINDOW_SIZE, approach_threshold=0.4, motion_var_thresh=0.0008):
        self.window_size = window_size
        self.distances = deque(maxlen=window_size)
        self.approach_threshold = approach_threshold
        self.motion_var_thresh = motion_var_thresh

    def update(self, dist_norm):
        self.distances.append(dist_norm)
        if len(self.distances) < self.window_size:
            return False
        arr = np.array(self.distances)
        diff = np.diff(arr)
        approach = np.sum(diff < -0.02)
        hold = np.mean(arr[-5:]) < self.approach_threshold
        motion_var = np.var(diff)
        return (approach > 2) and hold and (motion_var > self.motion_var_thresh)

    def reset(self):
        self.distances.clear()

# ===============================
# HELPER FUNCTIONS
# ===============================
def compute_angle(a, b, c):
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    return angle

def handle_long_eating_snapshot(pid, frame, bbox, frame_count, long_eating_flags):
    """
    Capture snapshot of a person eating long, only once per session.
    """
    if not ENABLE_LONG_EATING_SNAPSHOT:
        return
    
    x1, y1, x2, y2 = bbox
    if pid not in long_eating_flags or not long_eating_flags[pid]:
        snapshot_filename = os.path.join(
            SNAPSHOT_FOLDER, f"long_eating_pid{pid}_frame{frame_count}.jpg"
        )
        frame_copy = frame.copy()
        cv2.rectangle(frame_copy, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)
        cv2.putText(frame_copy, f"ID:{pid} LONG EATING", (int(x1), int(y1)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        cv2.imwrite(snapshot_filename, frame_copy)
        long_eating_flags[pid] = True

# ===============================
# INITIALIZATION
# ===============================
pose_model = YOLO(POSE_MODEL_PATH)
object_model = YOLO(OBJECT_MODEL_PATH)
tracker = Sort(max_age=10, min_hits=1)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise Exception("Camera not accessible!")

# Logging setup
with open(LOG_FILE, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["timestamp", "frame", "person_id", "hand_near_mouth",
                     "temporal_detect", "violation", "objects", "person_count"])

person_temporal = {}
violation_counters = {}
long_eating_flags = {}
frame_count = 0

print("✅ Running YOLO + SORT + Temporal Eating Tracker with elbow & object boost...")

# ===============================
# MAIN LOOP
# ===============================
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    h, w, _ = frame.shape

    # --- YOLO Object Detection ---
    results_obj = object_model(frame, verbose=False)
    detected_objects = []
    for box in results_obj[0].boxes:
        cls = int(box.cls[0])
        label = object_model.names[cls]
        detected_objects.append(label)
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 2)
        cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

    # --- YOLO Pose Detection ---
    results_pose = pose_model(frame, verbose=False)
    keypoints = results_pose[0].keypoints

    # Prepare detections for SORT
    detections_for_sort = []
    if keypoints is not None and len(keypoints.xy) > 0:
        for person in keypoints.xy:
            person_np = person.cpu().numpy()
            if person_np.shape[0] < 15:
                continue
            xs, ys = person_np[:,0], person_np[:,1]
            x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
            detections_for_sort.append([x1, y1, x2, y2, 1.0])

    dets_np = np.array(detections_for_sort) if len(detections_for_sort) > 0 else np.empty((0,5))
    tracked_objects = tracker.update(dets_np)

    # --- Count total people in this frame ---
    frame_person_ids = [int(trk[4]) for trk in tracked_objects]
    person_count = len(frame_person_ids)

    # --- Process each tracked person ---
    for trk in tracked_objects:
        x1, y1, x2, y2, pid = trk
        pid = int(pid)

        # Match closest keypoints
        min_dist = float('inf')
        matched_person = None
        if keypoints is not None and len(keypoints.xy) > 0:
            for person in keypoints.xy:
                person_np = person.cpu().numpy()
                cx = (person_np[:,0].min() + person_np[:,0].max())/2
                cy = (person_np[:,1].min() + person_np[:,1].max())/2
                dist = np.sqrt((cx-(x1+x2)/2)**2 + (cy-(y1+y2)/2)**2)
                if dist < min_dist:
                    min_dist = dist
                    matched_person = person_np
        if matched_person is None:
            continue

        # --- Keypoints ---
        nose = matched_person[0]
        left_wrist = matched_person[9]
        right_wrist = matched_person[10]
        left_elbow = matched_person[6]
        right_elbow = matched_person[7]

        nose_px = (int(nose[0]), int(nose[1]))
        lw_px = (int(left_wrist[0]), int(left_wrist[1]))
        rw_px = (int(right_wrist[0]), int(right_wrist[1]))
        le_px = (int(left_elbow[0]), int(left_elbow[1]))
        re_px = (int(right_elbow[0]), int(right_elbow[1]))

        # Draw keypoints
        cv2.circle(frame, nose_px, 5, (0,0,255), -1)
        cv2.circle(frame, lw_px, 5, (0,255,0), -1)
        cv2.circle(frame, rw_px, 5, (0,255,0), -1)
        cv2.circle(frame, le_px, 5, (255,0,255), -1)
        cv2.circle(frame, re_px, 5, (255,0,255), -1)

        # --- Hand distances ---
        lw_dist = np.linalg.norm(np.array(lw_px)-np.array(nose_px))/w
        rw_dist = np.linalg.norm(np.array(rw_px)-np.array(nose_px))/w
        dist_norm = np.clip(min(lw_dist, rw_dist), 0.01, 1.0)
        hand_near_mouth = dist_norm < HAND_NEAR_MOUTH_THRESHOLD

        # Temporal classifier
        if pid not in person_temporal:
            person_temporal[pid] = TemporalGestureClassifier()
        temporal_detect = person_temporal[pid].update(dist_norm)

        if pid not in violation_counters:
            violation_counters[pid] = 0

        # Elbow angle check
        left_elbow_angle = compute_angle(np.array(left_wrist), np.array(left_elbow), np.array(nose))
        right_elbow_angle = compute_angle(np.array(right_wrist), np.array(right_elbow), np.array(nose))
        elbow_ok = (ELBOW_ANGLE_MIN < left_elbow_angle < ELBOW_ANGLE_MAX) or \
                   (ELBOW_ANGLE_MIN < right_elbow_angle < ELBOW_ANGLE_MAX)

        # Object boost
        object_boost = any(obj in OBJECTS_OF_INTEREST for obj in detected_objects)

        # Final violation
        violation_flag = hand_near_mouth and (temporal_detect or object_boost) and elbow_ok
        if violation_flag:
            violation_counters[pid] += 1
        else:
            violation_counters[pid] = max(0, violation_counters[pid]-1)

        violation = violation_counters[pid] >= VIOLATION_FRAMES

        # Handle long-eating snapshot
        if violation_counters[pid] >= LONG_EATING_FRAMES:
            handle_long_eating_snapshot(pid, frame, (x1, y1, x2, y2), frame_count, long_eating_flags)
        else:
            long_eating_flags[pid] = False  # reset if they stop eating

        # Draw bounding box and status
        color = (0,0,255) if violation else (0,255,0)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        status_text = f"ID:{pid} {'EATING' if violation else ''}"
        cv2.putText(frame, status_text, (int(x1), int(y1)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Logging
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(LOG_FILE, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, frame_count, pid, hand_near_mouth,
                             temporal_detect, violation, detected_objects, person_count])

    cv2.imshow("YOLO + SORT + Temporal Eating Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Done.")
