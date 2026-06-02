import cv2
import time
import urllib.request
import numpy as np

from ultralytics import YOLO
from collections import defaultdict, deque

# ================= CONFIG =================

ESP32_URL = "http://10.211.187.178:81/stream"
CAMERA_NAME = "ESP32CAM 1"

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

LINE_Y = 260
FRAME_SKIP = 2
CONFIDENCE = 0.35

# HANYA KENDARAAN
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

# ================= TRACKER =================

class Tracker:
    def __init__(self):
        self.memory = {}
        self.counted = set()
        self.counts = defaultdict(int)

    def update(self, label, cx, cy):
        key = f"{label}_{cx//30}_{cy//30}"

        if key not in self.memory:
            self.memory[key] = cy
            return

        prev = self.memory[key]
        self.memory[key] = cy

        if prev < LINE_Y <= cy and key not in self.counted:
            self.counted.add(key)
            self.counts[label] += 1

# ================= STREAM =================

def connect_stream():
    return urllib.request.urlopen(ESP32_URL, timeout=5)

def get_frame(stream, buffer):
    try:
        buffer += stream.read(1024)

        a = buffer.find(b'\xff\xd8')
        b = buffer.find(b'\xff\xd9')

        if a != -1 and b != -1:
            jpg = buffer[a:b+2]
            buffer = buffer[b+2:]

            frame = cv2.imdecode(
                np.frombuffer(jpg, dtype=np.uint8),
                cv2.IMREAD_COLOR
            )

            return frame, buffer

        return None, buffer

    except:
        return None, buffer

# ================= MAIN =================

def main():

    print("Loading YOLOv8...")
    model = YOLO("yolov8n.pt")

    tracker = Tracker()

    fps_history = deque(maxlen=10)
    frame_id = 0

    while True:
        try:
            print(f"Connecting to {CAMERA_NAME}...")
            stream = connect_stream()
            print("Connected OK!")

            buffer = b''

            while True:

                frame, buffer = get_frame(stream, buffer)

                if frame is None:
                    continue

                frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

                # 🔥 SHARPEN
                frame = cv2.GaussianBlur(frame, (3,3), 0)
                frame = cv2.addWeighted(frame, 1.5, frame, -0.5, 0)

                frame_id += 1
                if frame_id % FRAME_SKIP != 0:
                    continue

                start = time.time()

                results = model(
                    frame,
                    imgsz=640,
                    conf=CONFIDENCE,
                    iou=0.5,
                    verbose=False
                )[0]

                fps = 1 / (time.time() - start + 0.0001)
                fps_history.append(fps)
                smooth_fps = sum(fps_history) / len(fps_history)

                # ================= LINE =================
                cv2.line(frame, (0, LINE_Y), (FRAME_WIDTH, LINE_Y), (0, 0, 255), 2)

                # ================= DETECTION =================
                for box in results.boxes:

                    conf = float(box.conf[0])
                    if conf < CONFIDENCE:
                        continue

                    cls = int(box.cls[0])

                    # 🔥 FILTER: hanya kendaraan
                    if cls not in VEHICLE_CLASSES:
                        continue

                    label_name = VEHICLE_CLASSES[cls]

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    tracker.update(label_name, cx, cy)

                    # ================= BOX =================
                    color = (0, 255, 0)

                    cv2.rectangle(frame, (x1, y1), (x2, y2),
                                  color, 2, cv2.LINE_AA)

                    label = f"{label_name} {conf:.2f}"

                    (w, h), _ = cv2.getTextSize(label,
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                0.6, 2)

                    cv2.rectangle(frame,
                                  (x1, y1 - h - 10),
                                  (x1 + w, y1),
                                  color, -1)

                    cv2.putText(frame,
                                label,
                                (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 0, 0),
                                2)

                # ================= UI CLEAN =================

                total = sum(tracker.counts.values())

                ui = [
                    f"FPS : {smooth_fps:.1f}",
                    f"Total : {total}",
                    f"Car : {tracker.counts['car']}",
                    f"Motor : {tracker.counts['motorcycle']}",
                    f"Bus : {tracker.counts['bus']}",
                    f"Truck : {tracker.counts['truck']}"
                ]

                y = 30
                for text in ui:
                    cv2.putText(frame, text, (10, y),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0,255,255), 2)
                    y += 30

                cv2.imshow("AI Traffic Vision", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    raise KeyboardInterrupt

        except KeyboardInterrupt:
            print("SYSTEM STOPPED")
            break

        except Exception as e:
            print("ERROR:", e)
            print("RECONNECTING...")
            time.sleep(2)

    cv2.destroyAllWindows()

# ================= RUN =================

if __name__ == "__main__":
    main()