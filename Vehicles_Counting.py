import cv2
import time
import urllib.request
import numpy as np
from datetime import datetime
from collections import defaultdict
from ultralytics import YOLO

# ================= CONFIG =================
ESP32_URL = "http://10.224.54.178:81/stream"
LINE_Y = 320
FRAME_SKIP = 2

VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

# ================= TRACKER =================
class VehicleTracker:
    def __init__(self):
        self.track_memory = {}
        self.counted_ids = set()
        self.counts = defaultdict(int)

        self.id_counter = 0
        self.object_ids = {}

    def update(self, cls, cx, cy):
        label = VEHICLE_CLASSES[cls]
        key = f"{cx//30}_{cy//30}"

        if key not in self.object_ids:
            self.id_counter += 1
            self.object_ids[key] = self.id_counter

        obj_id = f"{label}_{self.object_ids[key]}"

        if obj_id not in self.track_memory:
            self.track_memory[obj_id] = cy
            return False, self.object_ids[key]

        prev_y = self.track_memory[obj_id]
        self.track_memory[obj_id] = cy

        crossed = (
            (prev_y < LINE_Y and cy >= LINE_Y) or
            (prev_y > LINE_Y and cy <= LINE_Y)
        )

        if crossed and obj_id not in self.counted_ids:
            self.counted_ids.add(obj_id)
            self.counts[label] += 1
            return True, self.object_ids[key]

        return False, self.object_ids[key]

# ================= DRAW =================
def draw_overlay(frame, tracker, fps):
    cv2.line(frame, (0, LINE_Y), (800, LINE_Y), (0, 0, 255), 3)

    y = 40
    cv2.putText(frame, f"Total: {sum(tracker.counts.values())}",
                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    for vtype in ["car", "truck", "bus", "motorcycle"]:
        y += 30
        cv2.putText(frame,
                    f"{vtype.capitalize()}: {tracker.counts.get(vtype, 0)}",
                    (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2)

    cv2.putText(frame, f"FPS: {fps:.2f}",
                (600, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2)

# ================= MAIN =================
def main():
    print("Loading YOLO...")
    model = YOLO("yolov8n.pt")

    tracker = VehicleTracker()
    frame_id = 0
    prev_time = 0

    while True:
        try:
            print("Connecting to ESP32...")
            stream = urllib.request.urlopen(ESP32_URL, timeout=5)
            bytes_data = b''

            print("Connected! Starting...\n")

            while True:
                try:
                    bytes_data += stream.read(1024)

                    a = bytes_data.find(b'\xff\xd8')
                    b = bytes_data.find(b'\xff\xd9')

                    if a == -1 or b == -1:
                        continue

                    jpg = bytes_data[a:b+2]
                    bytes_data = bytes_data[b+2:]

                    # ===== VALIDASI =====
                    if len(jpg) == 0:
                        continue

                    frame = cv2.imdecode(
                        np.frombuffer(jpg, dtype=np.uint8),
                        cv2.IMREAD_COLOR
                    )

                    if frame is None:
                        continue

                    # ===== RESIZE =====
                    frame = cv2.resize(frame, (800, 600))
                    frame_id += 1

                    # ===== FPS =====
                    current_time = time.time()
                    fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
                    prev_time = current_time

                    # ===== SKIP FRAME =====
                    if frame_id % FRAME_SKIP != 0:
                        continue

                    # ===== YOLO DETECTION =====
                    results = model(frame, verbose=False)[0]

                    for box in results.boxes:
                        cls = int(box.cls[0])
                        if cls not in VEHICLE_CLASSES:
                            continue

                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                        counted, obj_id = tracker.update(cls, cx, cy)
                        label = VEHICLE_CLASSES[cls]

                        color = (0, 165, 255)

                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f"{label} {obj_id}",
                                    (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    color,
                                    2)

                        if counted:
                            print(f"{datetime.now().strftime('%H:%M:%S')} {label} lewat | Total: {sum(tracker.counts.values())}")

                    # ===== OVERLAY =====
                    draw_overlay(frame, tracker, fps)

                    cv2.imshow("Vehicle Counting", frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        raise KeyboardInterrupt

                except Exception as e:
                    print("⚠️ Frame error:", e)
                    continue

        except Exception as e:
            print("❌ Koneksi putus:", e)
            print("🔁 Reconnecting dalam 3 detik...\n")
            time.sleep(3)
            continue

    cv2.destroyAllWindows()

# ================= RUN =================
if __name__ == "__main__":
    main()