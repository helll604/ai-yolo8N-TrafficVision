import cv2
import time
import urllib.request
import numpy as np

from datetime import datetime
from collections import defaultdict
from ultralytics import YOLO


# ================= CONFIG =================

ESP32_URL = "http://10.224.54.178:81/stream"

WINDOW_NAME = "AI Traffic Vision"

FRAME_WIDTH = 800
FRAME_HEIGHT = 600

LINE_Y = 320

FRAME_SKIP = 2

CONFIDENCE_THRESHOLD = 0.4

MAX_BUFFER_SIZE = 50000

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

        self.last_seen = {}

    def cleanup_memory(self):

        current_time = time.time()

        remove_keys = []

        for key, t in self.last_seen.items():

            if current_time - t > 5:
                remove_keys.append(key)

        for key in remove_keys:

            self.last_seen.pop(key, None)

            self.track_memory.pop(key, None)

    def update(self, cls, cx, cy):

        label = VEHICLE_CLASSES[cls]

        grid_key = f"{label}_{cx//30}_{cy//30}"

        if grid_key not in self.object_ids:

            self.id_counter += 1

            self.object_ids[grid_key] = self.id_counter

        obj_id = self.object_ids[grid_key]

        unique_id = f"{label}_{obj_id}"

        self.last_seen[unique_id] = time.time()

        if unique_id not in self.track_memory:

            self.track_memory[unique_id] = cy

            return False, obj_id

        prev_y = self.track_memory[unique_id]

        self.track_memory[unique_id] = cy

        crossed = (
            (prev_y < LINE_Y and cy >= LINE_Y) or
            (prev_y > LINE_Y and cy <= LINE_Y)
        )

        if crossed and unique_id not in self.counted_ids:

            self.counted_ids.add(unique_id)

            self.counts[label] += 1

            return True, obj_id

        return False, obj_id


# ================= OVERLAY =================

def draw_overlay(frame, tracker, fps, latency):

    cv2.rectangle(
        frame,
        (0, 0),
        (340, 260),
        (0, 0, 0),
        -1
    )

    info = [
        f"FPS : {fps:.1f}",
        f"Latency : {latency:.1f} ms",
        "",
        f"Total : {sum(tracker.counts.values())}",
        f"Car : {tracker.counts.get('car', 0)}",
        f"Motorcycle : {tracker.counts.get('motorcycle', 0)}",
        f"Bus : {tracker.counts.get('bus', 0)}",
        f"Truck : {tracker.counts.get('truck', 0)}"
    ]

    y = 30

    for text in info:

        cv2.putText(
            frame,
            text,
            (15, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

        y += 28

    cv2.line(
        frame,
        (0, LINE_Y),
        (FRAME_WIDTH, LINE_Y),
        (0, 0, 255),
        3
    )

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    cv2.putText(
        frame,
        timestamp,
        (500, 580),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


# ================= MAIN =================

def main():

    print("Loading YOLOv8 Model...")

    model = YOLO("yolov8n.pt")

    tracker = VehicleTracker()

    frame_id = 0

    prev_time = time.time()

    fps = 0

    while True:

        try:

            print("Connecting to ESP32 Camera...")

            stream = urllib.request.urlopen(
                ESP32_URL,
                timeout=5
            )

            bytes_data = b''

            print("Connected Successfully!\n")

            while True:

                bytes_data += stream.read(4096)

                if len(bytes_data) > MAX_BUFFER_SIZE:
                    bytes_data = bytes_data[-MAX_BUFFER_SIZE:]

                a = bytes_data.find(b'\xff\xd8')
                b = bytes_data.find(b'\xff\xd9')

                if a == -1 or b == -1:
                    continue

                jpg = bytes_data[a:b+2]

                bytes_data = bytes_data[b+2:]

                frame = cv2.imdecode(
                    np.frombuffer(jpg, dtype=np.uint8),
                    cv2.IMREAD_COLOR
                )

                if frame is None:
                    continue

                frame = cv2.resize(
                    frame,
                    (FRAME_WIDTH, FRAME_HEIGHT)
                )

                frame_id += 1

                if frame_id % FRAME_SKIP != 0:
                    continue

                current_time = time.time()

                fps = 1 / (current_time - prev_time)

                prev_time = current_time

                start_time = time.time()

                results = model(
                    frame,
                    verbose=False
                )[0]

                latency = (
                    time.time() - start_time
                ) * 1000

                for box in results.boxes:

                    cls = int(box.cls[0])

                    if cls not in VEHICLE_CLASSES:
                        continue

                    confidence = float(box.conf[0])

                    if confidence < CONFIDENCE_THRESHOLD:
                        continue

                    x1, y1, x2, y2 = map(
                        int,
                        box.xyxy[0]
                    )

                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    counted, obj_id = tracker.update(
                        cls,
                        cx,
                        cy
                    )

                    label = VEHICLE_CLASSES[cls]

                    color = (0, 165, 255)

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        color,
                        2
                    )

                    cv2.putText(
                        frame,
                        f"ID:{obj_id} {label}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2
                    )

                    cv2.circle(
                        frame,
                        (cx, cy),
                        4,
                        (0, 255, 0),
                        -1
                    )

                    if counted:

                        print(
                            f"[{datetime.now().strftime('%H:%M:%S')}] "
                            f"{label.upper()} Counted"
                        )

                tracker.cleanup_memory()

                draw_overlay(
                    frame,
                    tracker,
                    fps,
                    latency
                )

                cv2.imshow(
                    WINDOW_NAME,
                    frame
                )

                if cv2.waitKey(1) & 0xFF == ord('q'):

                    raise KeyboardInterrupt

        except KeyboardInterrupt:

            print("\nSystem Stopped.")

            break

        except Exception as e:

            print("Connection Error:", e)

            print("Reconnecting...\n")

            time.sleep(2)

    cv2.destroyAllWindows()


# ================= RUN =================

if __name__ == "__main__":
    main()