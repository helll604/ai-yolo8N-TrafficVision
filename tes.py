import cv2
import os
import time
import threading
import urllib.request
import urllib.parse
import json
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import paho.mqtt.client as mqtt
import requests
from config.config import *

# ============================================================
# DEBUG FLAG
# ============================================================
DEBUG_TRACKING = False

# ============================================================
# OVERRIDE LOKAL
# ============================================================
COUNT_CONFIRM_FRAMES = 1
LOCAL_CONFIDENCE = 0.25
LOCAL_IMGSZ = None


# ============================================================
# 2. ESP32-CAM STREAM READER
# ============================================================

class ESP32StreamReader:
    MAX_BUF = 512 * 1024

    def __init__(self, url: str):
        self.url = url
        self._frame = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._connected = False
        self._stream = None

    def connect(self) -> bool:
        self._stop.clear()
        try:
            self._stream = urllib.request.urlopen(self.url, timeout=ESP32_TIMEOUT)
            self._connected = True
            threading.Thread(target=self._read_loop, daemon=True).start()
            print(f"[ESP32] Terhubung ke {self.url}")
            return True
        except Exception as e:
            print(f"[ESP32] Gagal connect: {e}")
            return False

    def _read_loop(self):
        buf = b""
        while not self._stop.is_set():
            try:
                chunk = self._stream.read(8192)
                if not chunk:
                    break
                buf += chunk
                if len(buf) > self.MAX_BUF:
                    buf = buf[-self.MAX_BUF:]
                while True:
                    start = buf.find(b"\xff\xd8")
                    end = buf.find(b"\xff\xd9", start + 2)
                    if start == -1 or end == -1:
                        break
                    jpg = buf[start:end + 2]
                    buf = buf[end + 2:]
                    frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if frame is not None:
                        with self._lock:
                            self._frame = frame
            except Exception as e:
                print(f"[ESP32] Stream error: {e}")
                break
        self._connected = False

    def read(self):
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def is_connected(self) -> bool:
        return self._connected

    def release(self):
        self._stop.set()
        if self._stream:
            try:
                self._stream.close()
            except Exception:
                pass


# ============================================================
# 3. VIDEO SOURCE
# ============================================================

class VideoSource:
    def __init__(self, video_path: str, loop: bool = True):
        self.video_path = video_path
        self.loop = loop
        self._cap = None
        self._video_done = False
        self._frame_width = 640
        self._frame_height = 480
        self._fps = 0
        self._total_frames = 0

    def connect(self) -> bool:
        self._cap = cv2.VideoCapture(self.video_path)
        if not self._cap.isOpened():
            print(f"[VIDEO] Gagal buka: {self.video_path}")
            return False
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[VIDEO] {self._total_frames} frame | {self._fps:.1f} FPS | resolusi asli: {w}x{h}")
        return True

    def set_target_size(self, width: int, height: int):
        self._frame_width = width
        self._frame_height = height

    def get_frame_size(self):
        return self._frame_width, self._frame_height

    def get_frame(self):
        ret, frame = self._cap.read()
        if not ret:
            if self.loop:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self._cap.read()
                if not ret:
                    self._video_done = True
                    return None
            else:
                self._video_done = True
                return None
        frame = cv2.resize(frame, (self._frame_width, self._frame_height))
        return frame

    def is_done(self) -> bool:
        return self._video_done

    def release(self):
        if self._cap:
            self._cap.release()


class ESP32Source:
    def __init__(self, url: str):
        self.url = url
        self._esp32 = None
        self._frame_width = 640
        self._frame_height = 480

    def connect(self) -> bool:
        self._esp32 = ESP32StreamReader(self.url)
        if self._esp32.connect():
            print(f"[ESP32] Resolusi: {self._frame_width}x{self._frame_height}")
            return True
        return False

    def get_frame_size(self):
        return self._frame_width, self._frame_height

    def get_frame(self):
        if self._esp32 and self._esp32.is_connected():
            frame = self._esp32.read()
            if frame is not None:
                if frame.shape[1] != self._frame_width or frame.shape[0] != self._frame_height:
                    frame = cv2.resize(frame, (self._frame_width, self._frame_height))
                return frame
        return None

    def is_done(self) -> bool:
        return False

    def reconnect(self):
        if self._esp32:
            self._esp32.release()
            time.sleep(ESP32_RECONNECT_S)
            self._esp32 = ESP32StreamReader(self.url)
            self._esp32.connect()

    def is_connected(self) -> bool:
        return self._esp32 is not None and self._esp32.is_connected()

    def release(self):
        if self._esp32:
            self._esp32.release()


# ============================================================
# 4. ROI CONFIG LOADER
# ============================================================

class ROIConfig:
    def __init__(self, frame_w: int, frame_h: int):
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.x1 = 0
        self.y1 = int(frame_h * 0.65)
        self.x2 = frame_w
        self.y2 = int(frame_h * 0.65)
        self._last_fetch = 0
        self._lock = threading.Lock()
        self._db_width = frame_w
        self._db_height = frame_h

    def _apply_data(self, data: dict):
        db_w = data.get("width", self._db_width)
        db_h = data.get("height", self._db_height)
        self.frame_w = db_w
        self.frame_h = db_h
        scale_x = self.frame_w / db_w if db_w else 1.0
        scale_y = self.frame_h / db_h if db_h else 1.0
        with self._lock:
            self.x1 = int(data["x1"] * scale_x)
            self.y1 = int(data["y1"] * scale_y)
            self.x2 = int(data["x2"] * scale_x)
            self.y2 = int(data["y2"] * scale_y)
            self._db_width = db_w
            self._db_height = db_h
        self._last_fetch = time.time()
        print(f"[ROI] Update → ({self.x1},{self.y1}) - ({self.x2},{self.y2})")

    def apply_from_mqtt(self, payload: str):
        try:
            data = json.loads(payload)
            if "camera_id" in data and data["camera_id"] != LARAVEL_CAMERA_ID:
                return
            self._apply_data(data)
            print("[ROI] Config dari MQTT (real-time)")
        except Exception as e:
            print(f"[ROI] Gagal parse MQTT: {e}")

    def fetch(self) -> bool:
        try:
            headers = {"Accept": "application/json"}
            if LARAVEL_API_TOKEN:
                headers["Authorization"] = f"Bearer {LARAVEL_API_TOKEN}"
            url = f"{LARAVEL_API_URL}?camera_id={LARAVEL_CAMERA_ID}"
            resp = requests.get(url, headers=headers, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            self._apply_data(data)
            print("[ROI] Config dari Laravel API")
            return True
        except Exception as e:
            print(f"[ROI] Gagal fetch API: {e}")
            return False

    def fetch_if_needed(self):
        if time.time() - self._last_fetch >= CONFIG_REFRESH_INTERVAL:
            threading.Thread(target=self.fetch, daemon=True).start()

    def get_line(self):
        with self._lock:
            return (self.x1, self.y1), (self.x2, self.y2)

    def get_line_y_at(self, cx: int) -> float:
        with self._lock:
            if self.x2 == self.x1:
                return float(self.y1)
            t = (cx - self.x1) / (self.x2 - self.x1)
            t = max(0.0, min(1.0, t))
            return self.y1 + t * (self.y2 - self.y1)

    def is_below_line(self, cx: int, cy: int) -> bool:
        return cy >= self.get_line_y_at(cx)

    def was_above_line(self, boxes: deque) -> bool:
        for (bx1, by1, bx2, by2) in boxes:
            cx = (bx1 + bx2) // 2
            cy = by2
            if not self.is_below_line(cx, cy):
                return True
        return False


# ============================================================
# 5. MJPEG RESTREAM SERVER
# ============================================================

class MJPEGStreamHandler(BaseHTTPRequestHandler):
    _frame_lock = threading.Lock()
    _latest_jpeg = None

    @classmethod
    def push_frame(cls, bgr_frame):
        ok, buf = cv2.imencode(".jpg", bgr_frame, [cv2.IMWRITE_JPEG_QUALITY, MJPEG_QUALITY])
        if ok:
            with cls._frame_lock:
                cls._latest_jpeg = buf.tobytes()

    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        if self.path == "/stream":
            self._serve_stream()
        elif self.path in ("/", "/index.html"):
            self._serve_index()
        else:
            self.send_response(404)
            self.end_headers()

    def _serve_index(self):
        html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Traffic Vision Live</title>
  <style>
    body {{ margin:0; background:#111; display:flex; flex-direction:column;
            align-items:center; justify-content:center; min-height:100vh; color:#eee; font-family:sans-serif; }}
    img  {{ max-width:100%; border:2px solid #444; border-radius:6px; }}
    h2   {{ margin-bottom:10px; color:#adf; }}
    p    {{ color:#888; font-size:0.85rem; }}
  </style>
</head>
<body>
  <h2>Traffic Vision — Live Stream</h2>
  <img src="/stream" alt="stream">
  <p>MJPEG stream · kualitas {MJPEG_QUALITY}% · port {MJPEG_PORT}</p>
</body>
</html>"""
        body = html.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            while True:
                with MJPEGStreamHandler._frame_lock:
                    jpeg = MJPEGStreamHandler._latest_jpeg
                if jpeg:
                    header = b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                    self.wfile.write(header + jpeg + b"\r\n")
                    self.wfile.flush()
                else:
                    time.sleep(0.03)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            print(f"[MJPEG] Stream error: {e}")


class MJPEGServer:
    def __init__(self):
        self._server = None
        self._thread = None

    def start(self):
        try:
            self._server = ThreadingHTTPServer((MJPEG_HOST, MJPEG_PORT), MJPEGStreamHandler)
            self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
            self._thread.start()
            print(f"[MJPEG] Server aktif → http://localhost:{MJPEG_PORT}/")
            print(f"[MJPEG] Stream URL  → http://localhost:{MJPEG_PORT}/stream")
        except Exception as e:
            print(f"[MJPEG] Gagal start server: {e}")

    def push_frame(self, frame):
        MJPEGStreamHandler.push_frame(frame)

    def stop(self):
        if self._server:
            self._server.shutdown()


# ============================================================
# 6. MQTT PUBLISHER + ROI SUBSCRIBER
# ============================================================

class MQTTPublisher:
    def __init__(self):
        self._client = mqtt.Client(client_id=f"python_traffic_vision_mqtt")
        self._connected = False
        self._roi_ref = None
        if MQTT_USERNAME:
            self._client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message = self._on_message
        self._last_metrics_publish = 0

    def set_roi(self, roi: ROIConfig):
        self._roi_ref = roi

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self._connected = True
            print(f"[MQTT] Terhubung ke broker {MQTT_BROKER}:{MQTT_PORT}")
            client.publish(MQTT_TOPIC_STATUS, json.dumps({
                "camera_id": LARAVEL_CAMERA_ID,
                "status": "active",
                "timestamp": datetime.now().isoformat(),
            }), retain=True)
            client.subscribe(MQTT_TOPIC_ROI, qos=1)
            print(f"[MQTT] Subscribe ROI: {MQTT_TOPIC_ROI}")
        else:
            print(f"[MQTT] Gagal connect, rc={rc}")

    def _on_disconnect(self, client, userdata, rc):
        self._connected = False
        print(f"[MQTT] Terputus (rc={rc})")

    def _on_message(self, client, userdata, msg):
        if msg.topic == MQTT_TOPIC_ROI:
            if self._roi_ref is not None:
                self._roi_ref.apply_from_mqtt(msg.payload.decode("utf-8"))

    def connect(self) -> bool:
        try:
            self._client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
            self._client.loop_start()
            for _ in range(50):
                if self._connected:
                    return True
                time.sleep(0.1)
            print("[MQTT] Timeout connect")
            return False
        except Exception as e:
            print(f"[MQTT] Error connect: {e}")
            return False

    def publish_data(self, stats: dict, fps: float, roi_config: ROIConfig):
        if not self._connected:
            return
        now = datetime.now().isoformat()
        pt1, pt2 = roi_config.get_line()
        payload_data = {
            "camera_id": LARAVEL_CAMERA_ID,
            "timestamp": now,
            "fps": round(fps, 1),
            "confidence": CONFIDENCE,
            "total": stats["total"],
            "active": stats["active"],
            "rate": round(stats["rate"], 2),
            "status": stats["status"],
            "per_vehicle": {
                "car": stats["per_vehicle"].get("car", 0),
                "motorcycle": stats["per_vehicle"].get("motorcycle", 0),
                "bus": stats["per_vehicle"].get("bus", 0),
                "truck": stats["per_vehicle"].get("truck", 0),
            },
            "roi": {
                "x1": pt1[0], "y1": pt1[1],
                "x2": pt2[0], "y2": pt2[1],
            },
        }
        try:
            self._client.publish(MQTT_TOPIC_DATA, json.dumps(payload_data), qos=1)
        except Exception as e:
            print(f"[MQTT] Publish error: {e}")

    def publish_metrics(self, fps: float, latency_ms: float):
        if not self._connected:
            return
        if time.time() - self._last_metrics_publish < 3:
            return
        self._last_metrics_publish = time.time()
        payload = {
            "camera_id": LARAVEL_CAMERA_ID,
            "fps": round(fps),
            "latency_ms": round(latency_ms),
            "timestamp": datetime.now().isoformat(),
        }
        try:
            self._client.publish(MQTT_TOPIC_METRICS, json.dumps(payload), qos=0)
        except Exception as e:
            print(f"[MQTT] Publish metrics error: {e}")

    def publish_detections(self, detections: list):
        if not self._connected or not detections:
            return
        payload = {
            "camera_id": LARAVEL_CAMERA_ID,
            "timestamp": datetime.now().isoformat(),
            "detections": detections,
        }
        try:
            self._client.publish(MQTT_TOPIC_DETECTION, json.dumps(payload), qos=0)
        except Exception as e:
            print(f"[MQTT] Publish detections error: {e}")

    def publish_status(self, status: str):
        payload = {
            "camera_id": LARAVEL_CAMERA_ID,
            "status": status,
            "timestamp": datetime.now().isoformat(),
        }
        try:
            self._client.publish(MQTT_TOPIC_STATUS, json.dumps(payload), retain=True)
        except Exception:
            pass

    def disconnect(self):
        self.publish_status("inactive")
        time.sleep(0.3)
        self._client.loop_stop()
        self._client.disconnect()


# ============================================================
# 7. VEHICLE TRACKER
# ============================================================

class VehicleTracker:
    def __init__(self, roi: ROIConfig):
        self.roi = roi
        self.tracks = {}
        self.counted_ids = set()
        self.counts = defaultdict(int)
        self.crossing_events = []
        self.frame_count = 0
        self.crossed_buffer = {}
        self.position_history = defaultdict(lambda: deque(maxlen=15))

    def update(self, tid: int, label: str,
               x1: int, y1: int, x2: int, y2: int, conf: float = 0.0) -> bool:
        if conf < 25:
            return False

        cx = (x1 + x2) // 2
        box_bottom_cy = y2
        is_below_now = self.roi.is_below_line(cx, box_bottom_cy)

        if tid not in self.tracks:
            self.tracks[tid] = {
                "id": tid,
                "label": label,
                "boxes": deque(maxlen=25),
                "crossed": False,
                "last_seen": self.frame_count,
                "prev_side": None,
                "confirm_count": 0,
                "first_seen": self.frame_count,
                "is_active": True,
            }
            if DEBUG_TRACKING:
                print(f"[DEBUG] Track baru ID#{tid} ({label})")

        track = self.tracks[tid]
        track["last_seen"] = self.frame_count
        track["label"] = label
        track["boxes"].append((x1, y1, x2, y2))
        track["is_active"] = True
        self.position_history[tid].append((cx, box_bottom_cy))

        if tid in self.crossed_buffer:
            if self.frame_count - self.crossed_buffer[tid] < 15:
                return False

        crossed = False
        if not track["crossed"]:
            eligible = track["prev_side"] in (None, "above")
            
            if eligible and is_below_now:
                track["confirm_count"] += 1
                
                if track["confirm_count"] >= COUNT_CONFIRM_FRAMES:
                    crossed = True
                    track["crossed"] = True
                    self.counted_ids.add(tid)
                    self.counts[label] += 1
                    self.crossed_buffer[tid] = self.frame_count
                    
                    self.crossing_events.append({
                        "id": tid,
                        "label": label,
                        "time": datetime.now(),
                        "frame": self.frame_count,
                        "cx": cx,
                    })
                    print(f"[✓] {label.upper()} #{tid} | Total: {self.counts[label]}")
            elif not is_below_now:
                track["confirm_count"] = 0

        track["prev_side"] = "below" if is_below_now else "above"
        return crossed

    def cleanup(self):
        stale = [tid for tid, t in self.tracks.items()
                 if self.frame_count - t["last_seen"] > MAX_TRACK_AGE]
        for tid in stale:
            if DEBUG_TRACKING and not self.tracks[tid]["crossed"]:
                print(f"[DEBUG] Track ID#{tid} ({self.tracks[tid]['label']}) DIHAPUS (stale)")
            self.tracks.pop(tid, None)
            if tid in self.position_history:
                del self.position_history[tid]

    def get_active_count(self) -> int:
        return sum(
            1 for t in self.tracks.values()
            if t["is_active"] and self.frame_count - t["last_seen"] < 20
        )

    # ============================================================
    # PERUBAHAN: LANCAR < 5, PADAT 5-15, MACET > 15
    # ============================================================
    def get_congestion_status(self) -> str:
        """
        Status berdasarkan ACTIVE count:
        - LANCAR: active < 8
        - PADAT: 8 <= active <= 15
        - MACET: active > 15
        """
        active = self.get_active_count()
        
        if active > 15:
            return "macet"
        elif active >= 8:
            return "padat"
        else:
            return "lancar"

    def increment_frame(self):
        self.frame_count += 1

    def get_stats(self) -> dict:
        total = sum(self.counts.values())
        active = self.get_active_count()

        rate = 0.0
        if len(self.crossing_events) > 1:
            t0 = self.crossing_events[0]["time"]
            t1 = self.crossing_events[-1]["time"]
            dur = (t1 - t0).total_seconds() / 60.0
            rate = len(self.crossing_events) / max(dur, 1.0)
        elif len(self.crossing_events) == 1:
            rate = 1.0

        return {
            "total": total,
            "active": active,
            "per_vehicle": dict(self.counts),
            "rate": rate,
            "status": self.get_congestion_status(),
            "track_count": len(self.tracks),
        }


# ============================================================
# 8. MAIN APPLICATION
# ============================================================

class TrafficVisionApp:
    def __init__(self):
        self.model = None
        self.tracker = None
        self.source = None
        self.roi = None
        self.mqtt = MQTTPublisher()
        self.mjpeg = MJPEGServer()
        self.fps_history = deque(maxlen=30)
        self.latency_history = deque(maxlen=30)
        self.running = True
        self.show_line = True
        self._no_frame_count = 0
        self._last_stat_time = 0.0
        self._last_mqtt_time = 0.0
        self._start_time = time.time()
        self.frame_width = 640
        self.frame_height = 480

    def load_model(self):
        print("Memuat model YOLOv8n...")
        t0 = time.time()
        self.model = YOLO("yolov8n.pt")
        self.model.fuse()
        print(f"Model siap ({time.time() - t0:.1f}s)")

    def _init_source(self) -> bool:
        if SOURCE_MODE == "VIDEO":
            self.source = VideoSource(VIDEO_PATH, VIDEO_LOOP)
        elif SOURCE_MODE == "ESP32-CAM":
            self.source = ESP32Source(ESP32_URL)
        else:
            raise ValueError(f"Mode tidak dikenal: {SOURCE_MODE}")
        return self.source.connect()

    def _init_roi(self):
        self.roi = ROIConfig(self.frame_width, self.frame_height)
        if not self.roi.fetch():
            print("[ROI] Pakai default ROI (65% tinggi frame)")
        self.mqtt.set_roi(self.roi)
        self.tracker = VehicleTracker(self.roi)

    def process_frame(self, frame):
        if SOURCE_MODE == "ESP32-CAM":
            frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=15)
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            frame = cv2.filter2D(frame, -1, kernel)

        if self.roi:
            target_w = self.roi.frame_w
            target_h = self.roi.frame_h
            if frame.shape[1] != target_w or frame.shape[0] != target_h:
                frame = cv2.resize(frame, (target_w, target_h))
                self.frame_width = target_w
                self.frame_height = target_h

        t0 = time.time()
        effective_imgsz = LOCAL_IMGSZ if LOCAL_IMGSZ is not None else IMGSZ
        results = self.model.track(
            frame,
            persist=True,
            imgsz=effective_imgsz,
            conf=LOCAL_CONFIDENCE,
            iou=0.45,
            classes=DETECT_CLASSES,
            verbose=False,
        )[0]

        inf_ms = (time.time() - t0) * 1000
        fps = 1000.0 / inf_ms if inf_ms > 0 else 0
        self.fps_history.append(fps)
        self.latency_history.append(inf_ms)
        return frame, results, round(inf_ms, 2)

    # ============================================================
    # SATU GARIS COUNTING LINE
    # ============================================================
    def draw_roi(self, frame):
        if not self.show_line or self.roi is None:
            return frame

        pt1, pt2 = self.roi.get_line()
        
        # HANYA SATU GARIS
        cv2.line(frame, pt1, pt2, (0, 0, 255), 3)
        
        cv2.circle(frame, pt1, 6, (0, 200, 255), -1)
        cv2.circle(frame, pt2, 6, (0, 200, 255), -1)

        mid_x = (pt1[0] + pt2[0]) // 2
        mid_y = (pt1[1] + pt2[1]) // 2
        cv2.putText(frame, "COUNTING LINE",
                    (mid_x - 60, mid_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return frame

    def draw_detections(self, frame, results):
        if results is None or results.boxes is None or self.tracker is None:
            return frame, []

        has_ids = results.boxes.id is not None
        detections = []
        LABEL_ID = {
            "car": "mobil",
            "truck": "truk",
            "motorcycle": "motor",
            "bus": "bis",
        }

        boxes_list = []
        for box in results.boxes:
            conf = int(round(float(box.conf[0]) * 100))
            cls = int(box.cls[0])
            if cls in VEHICLE_CLASSES and conf >= 18:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                boxes_list.append((x1, y1, x2, y2, conf, cls))
        boxes_list.sort(key=lambda b: b[1])

        for x1, y1, x2, y2, conf, cls in boxes_list:
            label = VEHICLE_CLASSES[cls]
            label_id = LABEL_ID.get(label, label)
            color = VEHICLE_COLORS.get(label, (0, 255, 0))
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            if label == "motorcycle" and conf < 18:
                continue
            elif label == "car" and conf < 20:
                continue
            elif label == "bus" and conf < 22:
                continue
            elif label == "truck" and conf < 20:
                continue

            tid = None
            if has_ids and results.boxes.id is not None:
                for i, box in enumerate(results.boxes):
                    if int(box.cls[0]) == cls and abs(box.xyxy[0][0] - x1) < 15:
                        tid = int(results.boxes.id[i])
                        break

            crossed = False
            if tid is not None:
                crossed = self.tracker.update(tid, label, x1, y1, x2, y2, conf)

            if crossed:
                detections.append({
                    "vehicle_type": label_id,
                    "confidence_score": conf,
                })
                cv2.putText(frame, f"+1 {label.upper()}", (cx - 40, cy - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.circle(frame, (cx, cy), 15, (0, 255, 255), 2)

            is_counted = tid is not None and tid in self.tracker.counted_ids
            color_box = (0, 255, 255) if is_counted else color
            thickness = 3 if is_counted else 2

            cv2.rectangle(frame, (x1, y1), (x2, y2), color_box, thickness)
            cv2.circle(frame, (cx, cy), 3, color, -1)

            txt = f"{label.upper()} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color_box, -1)
            cv2.putText(frame, txt, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

            if tid is not None:
                status = "V" if is_counted else "O"
                cv2.putText(frame, f"{status}#{tid}", (x1, y2 + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, color_box, 1)

        return frame, detections

    def draw_panel(self, frame):
        if self.tracker is None:
            return frame

        stats = self.tracker.get_stats()
        avg_fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0

        panel_w = 260
        panel_h = 280
        xs, ys = 8, 8

        overlay = frame.copy()
        cv2.rectangle(overlay, (xs, ys), (xs + panel_w, ys + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
        cv2.rectangle(frame, (xs, ys), (xs + panel_w, ys + panel_h), (100, 100, 100), 1)

        y = ys + 22

        def put(text, yy, scale=0.45, color=(255, 255, 255), bold=False):
            cv2.putText(frame, text, (xs + 10, yy),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2 if bold else 1)

        mode_text = "ESP32" if SOURCE_MODE == "ESP32-CAM" else "VIDEO"
        put(f"{mode_text} | TRAFFIC VISION", y, 0.5, (255, 255, 255), True); y += 20

        mqtt_dot = "●" if self.mqtt._connected else "○"
        put(f"FPS: {avg_fps:.1f}  MQTT:{mqtt_dot}  MJPEG:●", y, 0.4, (200, 200, 200)); y += 18

        put(f"TOTAL: {stats['total']}", y, 0.5, (255, 255, 255), True); y += 22

        for key in ["car", "motorcycle", "bus", "truck"]:
            cnt = stats["per_vehicle"].get(key, 0)
            put(f"{key.upper():12s}: {cnt:3d}", y, 0.4); y += 17

        y += 2
        put(f"AKTIF: {stats['active']}", y, 0.4, (200, 200, 200)); y += 18

        pt1, pt2 = self.roi.get_line()
        put(f"ROI ({pt1[0]},{pt1[1]})-({pt2[0]},{pt2[1]})", y, 0.35, (180, 180, 255)); y += 16

        put(f"Stream: :{MJPEG_PORT}/stream", y, 0.35, (150, 255, 150)); y += 18

        status = stats["status"]
        y += 2
        if status == "macet":
            cv2.rectangle(frame, (xs + 8, y - 2), (xs + panel_w - 8, y + 22), (0, 0, 200), -1)
            put("STATUS: MACET!", y + 16, 0.5, (255, 255, 255), True)
        elif status == "padat":
            cv2.rectangle(frame, (xs + 8, y - 2), (xs + panel_w - 8, y + 22), (0, 165, 255), -1)
            put("STATUS: PADAT", y + 16, 0.5, (255, 255, 255), True)
        else:
            cv2.rectangle(frame, (xs + 8, y - 2), (xs + panel_w - 8, y + 22), (0, 200, 0), -1)
            put("STATUS: LANCAR", y + 16, 0.5, (255, 255, 255), True)

        y += 28
        cv2.putText(frame, f"Tracks: {len(self.tracker.tracks)} | Counted: {len(self.tracker.counted_ids)}",
                    (xs + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 0), 1)

        return frame

    def print_stats(self):
        if self.tracker is None:
            return
        s = self.tracker.get_stats()
        now = datetime.now().strftime("%H:%M:%S")
        status = s["status"].upper()
        fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0
        print(
            f"[{now}] {status} | FPS:{fps:.1f} | "
            f"Total:{s['total']:3d} | "
            f"Car:{s['per_vehicle'].get('car', 0):3d} | "
            f"Moto:{s['per_vehicle'].get('motorcycle', 0):3d} | "
            f"Bus:{s['per_vehicle'].get('bus', 0):3d} | "
            f"Truck:{s['per_vehicle'].get('truck', 0):3d} | "
            f"Aktif:{s['active']:2d} | "
            f"Track:{s['track_count']:2d}"
        )

    def save_report(self):
        if self.tracker is None:
            return
        os.makedirs(REPORT_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(REPORT_DIR, f"traffic_report_{ts}.txt")
        stats = self.tracker.get_stats()
        elapsed = int(time.time() - self._start_time)

        lines = [
            "=" * 50,
            "TRAFFIC VISION SYSTEM - REPORT",
            "=" * 50,
            f"Waktu      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Sumber     : {SOURCE_MODE}",
            f"Durasi     : {elapsed//3600:02d}h {(elapsed%3600)//60:02d}m {elapsed%60:02d}s",
            "-" * 50,
            f"TOTAL KENDARAAN : {stats['total']}",
            "-" * 50,
        ]
        for key in ["car", "motorcycle", "bus", "truck"]:
            lines.append(f"{key.upper():15s} : {stats['per_vehicle'].get(key, 0):4d}")

        lines += [
            "-" * 50,
            f"STATUS LALU LINTAS: {stats['status'].upper()}",
            "=" * 50,
            "", "LOG CROSSING:", "-" * 50,
        ]
        for ev in self.tracker.crossing_events[-50:]:
            lines.append(f"  [{ev['time'].strftime('%H:%M:%S')}] ID#{ev['id']:04d}  {ev['label'].upper()}")
        if not self.tracker.crossing_events:
            lines.append("  (Tidak ada kendaraan tercatat)")
        lines.append("=" * 50)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"\n[REPORT] Tersimpan ke {filepath}")

    def run(self):
        print("=" * 60)
        print("  TRAFFIC VISION SYSTEM")
        print(f"  Mode       : {SOURCE_MODE}")
        print(f"  Confidence : {CONFIDENCE}")
        print(f"  IMGSZ      : {IMGSZ}")
        print("=" * 60)

        self.load_model()

        if not self._init_source():
            print("[ERROR] Gagal membuka sumber video!")
            return

        self.frame_width, self.frame_height = self.source.get_frame_size()
        print(f"Frame: {self.frame_width}x{self.frame_height}")

        self._init_roi()

        mqtt_ok = self.mqtt.connect()
        if not mqtt_ok:
            print("[MQTT] Berjalan tanpa MQTT")

        self.mjpeg.start()

        print("\nKontrol: Q=Keluar | S=Statistik | R=Reset | L=Line | F=Fetch ROI\n")

        self._last_stat_time = time.time()
        self._last_mqtt_time = time.time()
        self._start_time = time.time()

        while self.running:
            frame = self.source.get_frame()

            if frame is None:
                self._no_frame_count += 1
                time.sleep(0.01)
                if SOURCE_MODE == "ESP32-CAM" and self._no_frame_count >= 50:
                    print("[ESP32] Reconnecting...")
                    self.source.reconnect()
                    self._no_frame_count = 0
                    time.sleep(2)
                continue

            self._no_frame_count = 0

            if SOURCE_MODE == "VIDEO" and self.source.is_done():
                print("\n[VIDEO] Selesai.")
                break

            self.roi.fetch_if_needed()

            frame, results, latency_ms = self.process_frame(frame)
            frame = self.draw_roi(frame)
            frame, detections = self.draw_detections(frame, results)
            frame = self.draw_panel(frame)

            self.tracker.cleanup()
            self.tracker.increment_frame()

            self.mjpeg.push_frame(frame)

            now = time.time()
            if now - self._last_stat_time >= 3:
                self.print_stats()
                self._last_stat_time = now

            if now - self._last_mqtt_time >= MQTT_PUBLISH_INTERVAL:
                avg_fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0
                self.mqtt.publish_data(self.tracker.get_stats(), avg_fps, self.roi)
                self._last_mqtt_time = now

            avg_fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0
            self.mqtt.publish_metrics(avg_fps, latency_ms)
            self.mqtt.publish_detections(detections)

            cv2.imshow("Traffic Vision", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False
            elif key == ord('s'):
                self.print_stats()
            elif key == ord('r'):
                self.tracker = VehicleTracker(self.roi)
                self._start_time = time.time()
                print("[RESET] Semua hitungan direset!")
            elif key == ord('l'):
                self.show_line = not self.show_line
            elif key == ord('f'):
                self.roi.fetch()
                print("[ROI] Config diperbarui dari API")

        self.mqtt.disconnect()
        self.mjpeg.stop()
        self.source.release()
        cv2.destroyAllWindows()

        print("\n" + "=" * 60)
        print("  SISTEM DIHENTIKAN")
        self.print_stats()
        self.save_report()
        print("=" * 60)


# ============================================================
# 9. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"  MODE AKTIF : {SOURCE_MODE}")
    if SOURCE_MODE == "VIDEO":
        print(f"  Video Path : {VIDEO_PATH}")
    else:
        print(f"  ESP32 URL  : {ESP32_URL}")
    print(f"  MQTT       : {MQTT_BROKER}:{MQTT_PORT}")
    print(f"  Laravel    : {LARAVEL_API_URL}")
    print(f"  MJPEG      : http://localhost:{MJPEG_PORT}/")
    print(f"{'='*60}\n")

    app = TrafficVisionApp()
    app.run()