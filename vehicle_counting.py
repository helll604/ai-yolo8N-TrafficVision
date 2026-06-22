import cv2
import os
import time
import threading
import urllib.request
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
from datetime import datetime

# ============================================================
# 1. KONFIGURASI - PILIH MODE DI SINI!
# ============================================================

# ===== PILIH SALAH SATU =====
#SOURCE_MODE = "VIDEO"        # UNCOMMENT untuk mode video
SOURCE_MODE = "ESP32-CAM"     # UNCOMMENT untuk mode ESP32

# Konfigurasi Video
VIDEO_PATH = "videos/traffic5.mp4"
VIDEO_LOOP = True

# Konfigurasi ESP32-CAM
ESP32_URL = "http://10.136.172.178:81/stream"
ESP32_TIMEOUT = 10
ESP32_RECONNECT_S = 3

# Parameter Deteksi (sama untuk kedua mode)
CONFIDENCE = 0.35
IMGSZ = 640

# Parameter Counting
CONFIRM_FRAMES = 3
MAX_TRACK_AGE = 50

# Threshold Kemacetan
CONGESTION_THRESHOLD = 10
TOTAL_CONGESTION = 15

COUNT_DIRECTION = "down"

VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
VEHICLE_COLORS = {
    "car": (0, 255, 0),
    "motorcycle": (0, 255, 255),
    "bus": (255, 100, 100),
    "truck": (0, 165, 255),
}

DETECT_CLASSES = [2, 3, 5, 7]
REPORT_DIR = "outputs/reports"


# ============================================================
# 2. ESP32-CAM STREAM READER (TIDAK BERUBAH)
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
            t = threading.Thread(target=self._read_loop, daemon=True)
            t.start()
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
            if self._frame is None:
                return None
            return self._frame.copy()

    def is_connected(self) -> bool:
        return self._connected

    def release(self):
        self._stop.set()
        if self._stream:
            try:
                self._stream.close()
            except:
                pass


# ============================================================
# 3. VIDEO SOURCE - DIPISAHKAN UNTUK VIDEO & ESP32
# ============================================================

class VideoSource:
    """Class untuk membaca dari file video"""
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
        
        # Resize proporsional
        if w > h:
            self._frame_width = 640
            self._frame_height = int(h * 640 / w)
        else:
            self._frame_height = 640
            self._frame_width = int(w * 640 / h)
        
        print(f"[VIDEO] {self._total_frames} frame | {self._fps:.1f} FPS | {w}x{h} -> {self._frame_width}x{self._frame_height}")
        return True

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
    """Class untuk membaca dari ESP32-CAM"""
    def __init__(self, url: str):
        self.url = url
        self._esp32 = None
        self._frame_width = 640
        self._frame_height = 480

    def connect(self) -> bool:
        self._esp32 = ESP32StreamReader(self.url)
        if self._esp32.connect():
            # Coba dapatkan ukuran frame (default 640x480 untuk ESP32-CAM)
            self._frame_width = 640
            self._frame_height = 480
            print(f"[ESP32] Resolusi: {self._frame_width}x{self._frame_height}")
            return True
        return False

    def get_frame_size(self):
        return self._frame_width, self._frame_height

    def get_frame(self):
        if self._esp32 and self._esp32.is_connected():
            frame = self._esp32.read()
            if frame is not None:
                # Resize jika perlu
                if frame.shape[1] != self._frame_width or frame.shape[0] != self._frame_height:
                    frame = cv2.resize(frame, (self._frame_width, self._frame_height))
                return frame
        return None

    def is_done(self) -> bool:
        return False  # ESP32 tidak pernah selesai

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
# 4. VEHICLE TRACKER (TIDAK BERUBAH)
# ============================================================

class VehicleTracker:
    def __init__(self, line_y):
        self.tracks = {}
        self.counted_ids = set()
        self.counts = defaultdict(int)
        self.crossing_events = []
        self.frame_count = 0
        self.line_y = line_y

    def update(self, tid: int, label: str, x1: int, y1: int, x2: int, y2: int, conf: float = 0.0) -> bool:
        if conf < 0.25:
            return False
        
        box_bottom = y2
        
        if tid not in self.tracks:
            self.tracks[tid] = {
                "id": tid,
                "label": label,
                "boxes": deque(maxlen=20),
                "crossed": False,
                "last_seen": self.frame_count,
                "crossing_frame": None,
                "has_been_above": False,
            }
        
        track = self.tracks[tid]
        track["last_seen"] = self.frame_count
        track["label"] = label
        track["boxes"].append((x1, y1, x2, y2))
        
        for (bx1, by1, bx2, by2) in track["boxes"]:
            if by2 < self.line_y:
                track["has_been_above"] = True
                break
        
        crossed = False
        
        if not track["crossed"]:
            if track["has_been_above"] and box_bottom >= self.line_y:
                if track["crossing_frame"] is None:
                    track["crossing_frame"] = self.frame_count
                
                if (self.frame_count - track["crossing_frame"]) >= CONFIRM_FRAMES:
                    crossed = True
                    track["crossed"] = True
                    self.counted_ids.add(tid)
                    self.counts[label] += 1
                    
                    self.crossing_events.append({
                        "id": tid,
                        "label": label,
                        "time": datetime.now(),
                        "frame": self.frame_count,
                    })
                    
                    print(f"  COUNTED [{label.upper()}] ID#{tid} | Total: {self.counts[label]}")
        
        return crossed
    
    def cleanup(self):
        stale = []
        for tid, track in self.tracks.items():
            age = self.frame_count - track["last_seen"]
            if age > MAX_TRACK_AGE:
                stale.append(tid)
        
        for tid in stale:
            if tid in self.tracks:
                del self.tracks[tid]
    
    def get_active_count(self) -> int:
        active = 0
        for tid, track in self.tracks.items():
            if not track["crossed"]:
                age = self.frame_count - track["last_seen"]
                if age < 10:
                    active += 1
        return active
    
    def is_congested(self) -> bool:
        total = sum(self.counts.values())
        for key in ["car", "motorcycle", "bus", "truck"]:
            if self.counts.get(key, 0) >= CONGESTION_THRESHOLD:
                return True
        if total >= TOTAL_CONGESTION:
            return True
        return False
    
    def increment_frame(self):
        self.frame_count += 1
    
    def get_stats(self) -> dict:
        total = sum(self.counts.values())
        active = self.get_active_count()
        is_congested = self.is_congested()
        
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
            "is_congested": is_congested,
            "track_count": len(self.tracks),
        }


# ============================================================
# 5. MAIN APPLICATION (DIPISAHKAN LOGIKA VIDEO & ESP32)
# ============================================================

class TrafficVisionApp:
    def __init__(self):
        self.model = None
        self.tracker = None
        self.source = None  # Akan diinisialisasi berdasarkan mode
        self.fps_history = deque(maxlen=30)
        self.running = True
        self.show_line = True
        self._no_frame_count = 0
        self._last_stat_time = 0.0
        self._start_time = time.time()
        
        self.frame_width = 640
        self.frame_height = 480
        self.line_y = 312

    def load_model(self):
        print("Memuat model YOLOv8n...")
        t0 = time.time()
        self.model = YOLO("yolov8n.pt")
        self.model.fuse()
        print(f"Model siap ({time.time() - t0:.1f}s)")

    def _init_source(self):
        """Inisialisasi sumber video berdasarkan SOURCE_MODE"""
        if SOURCE_MODE == "VIDEO":
            self.source = VideoSource(VIDEO_PATH, VIDEO_LOOP)
        elif SOURCE_MODE == "ESP32-CAM":
            self.source = ESP32Source(ESP32_URL)
        else:
            raise ValueError(f"Mode tidak dikenal: {SOURCE_MODE}")
        
        return self.source.connect()

    def process_frame(self, frame):
        """Proses deteksi dengan YOLO"""
        # Enhancement khusus ESP32
        if SOURCE_MODE == "ESP32-CAM":
            frame = cv2.convertScaleAbs(frame, alpha=1.15, beta=10)
        
        t0 = time.time()
        results = self.model.track(
            frame,
            persist=True,
            imgsz=IMGSZ,
            conf=CONFIDENCE,
            classes=DETECT_CLASSES,
            verbose=False,
        )[0]
        
        inf_ms = (time.time() - t0) * 1000
        fps = 1000.0 / inf_ms if inf_ms > 0 else 0
        self.fps_history.append(fps)
        
        return frame, results

    def draw_roi(self, frame):
        """Gambar garis counting"""
        if not self.show_line or self.tracker is None:
            return frame
        
        line_y = self.tracker.line_y
        
        # Garis merah tebal untuk counting
        cv2.line(frame, (0, line_y), (self.frame_width, line_y), (0, 0, 255), 3)
        
        # Label
        cv2.putText(frame, "COUNTING LINE", (self.frame_width - 170, line_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame

    def draw_detections(self, frame, results):
        """Gambar bounding box dan label deteksi"""
        if results is None or results.boxes is None or self.tracker is None:
            return frame
        
        has_ids = results.boxes.id is not None
        
        for box in results.boxes:
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            
            if cls not in VEHICLE_CLASSES or conf < 0.25:
                continue
            
            label = VEHICLE_CLASSES[cls]
            color = VEHICLE_COLORS.get(label, (0, 255, 0))
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            
            tid = int(box.id[0]) if has_ids else None
            
            crossed = False
            if tid is not None:
                crossed = self.tracker.update(tid, label, x1, y1, x2, y2, conf)
            
            is_counted = tid in self.tracker.counted_ids
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
            
            if crossed:
                cv2.putText(frame, f"+1 {label.upper()}", (cx - 40, cy - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.circle(frame, (cx, cy), 15, (0, 255, 255), 2)
        
        return frame

    def draw_panel(self, frame):
        """Gambar panel informasi"""
        if self.tracker is None:
            return frame
            
        stats = self.tracker.get_stats()
        avg_fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0
        
        panel_w = 230
        panel_h = 210
        x_start = 8
        y_start = 8
        
        # Background panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (x_start, y_start), (x_start + panel_w, y_start + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
        cv2.rectangle(frame, (x_start, y_start), (x_start + panel_w, y_start + panel_h), (100, 100, 100), 1)
        
        y_pos = y_start + 22
        def put(text, y, scale=0.45, color=(255, 255, 255), bold=False):
            cv2.putText(frame, text, (x_start + 10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                        2 if bold else 1)
        
        # Mode indicator
        mode_text = "ESP32" if SOURCE_MODE == "ESP32-CAM" else "VIDEO"
        put(f"{mode_text} | TRAFFIC VISION", y_pos, 0.5, (255, 255, 255), True)
        y_pos += 20
        put(f"FPS: {avg_fps:.1f}", y_pos, 0.4, (200, 200, 200))
        y_pos += 18
        
        # TOTAL
        put(f"TOTAL: {stats['total']}", y_pos, 0.5, (255, 255, 255), True)
        y_pos += 22
        
        # PER JENIS
        for key in ["car", "motorcycle", "bus", "truck"]:
            cnt = stats["per_vehicle"].get(key, 0)
            put(f"{key.upper():12s}: {cnt:3d}", y_pos, 0.4, (255, 255, 255))
            y_pos += 17
        
        # AKTIF
        y_pos += 2
        put(f"AKTIF: {stats['active']}", y_pos, 0.4, (200, 200, 200))
        y_pos += 18
        
        # STATUS
        if stats["is_congested"]:
            cv2.rectangle(frame, (x_start + 8, y_pos - 2), 
                         (x_start + panel_w - 8, y_pos + 22), (0, 0, 200), -1)
            put("STATUS: MACET!", y_pos + 16, 0.5, (255, 255, 255), True)
        else:
            put("STATUS: LANCAR", y_pos + 16, 0.5, (0, 255, 0), True)
        
        return frame

    def print_stats(self):
        if self.tracker is None:
            return
        s = self.tracker.get_stats()
        now = datetime.now().strftime("%H:%M:%S")
        status = "MACET" if s["is_congested"] else "LANCAR"
        print(
            f"[{now}] {status} | "
            f"Total:{s['total']:3d} | "
            f"Car:{s['per_vehicle'].get('car', 0):3d} | "
            f"Moto:{s['per_vehicle'].get('motorcycle', 0):3d} | "
            f"Bus:{s['per_vehicle'].get('bus', 0):3d} | "
            f"Truck:{s['per_vehicle'].get('truck', 0):3d} | "
            f"Aktif:{s['active']:2d}"
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
            cnt = stats["per_vehicle"].get(key, 0)
            lines.append(f"{key.upper():15s} : {cnt:4d}")
        
        lines += [
            "-" * 50,
            f"STATUS KEMACETAN: {'MACET' if stats['is_congested'] else 'LANCAR'}",
            "=" * 50,
            "",
            "LOG CROSSING:",
            "-" * 50,
        ]
        
        for ev in self.tracker.crossing_events[-30:]:
            lines.append(
                f"  [{ev['time'].strftime('%H:%M:%S')}] "
                f"ID#{ev['id']:04d}  {ev['label'].upper()}"
            )
        
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
        
        # Load model
        self.load_model()
        
        # Inisialisasi sumber video
        if not self._init_source():
            print("[ERROR] Gagal membuka sumber video!")
            return
        
        # Dapatkan ukuran frame
        self.frame_width, self.frame_height = self.source.get_frame_size()
        self.line_y = int(self.frame_height * 0.65)
        self.tracker = VehicleTracker(self.line_y)
        
        print(f"Frame: {self.frame_width}x{self.frame_height}")
        print(f"LINE Y: {self.line_y}")
        print("\nKontrol: Q=Keluar | S=Statistik | R=Reset | L=Line | U/D=Atur garis\n")
        
        self._last_stat_time = time.time()
        self._start_time = time.time()
        
        while self.running:
            # --- AMBIL FRAME ---
            frame = self.source.get_frame()
            
            # --- HANDLE ERROR ESP32 ---
            if frame is None:
                self._no_frame_count += 1
                time.sleep(0.01)
                
                if SOURCE_MODE == "ESP32-CAM" and self._no_frame_count >= 100:
                    print("[ESP32] Tidak ada frame, mencoba reconnect...")
                    self.source.reconnect()
                    self._no_frame_count = 0
                continue
            
            self._no_frame_count = 0
            
            # --- HANDLE VIDEO SELESAI ---
            if SOURCE_MODE == "VIDEO" and self.source.is_done():
                print("\n[VIDEO] Selesai.")
                break
            
            # --- PROSES FRAME ---
            frame, results = self.process_frame(frame)
            frame = self.draw_roi(frame)
            frame = self.draw_detections(frame, results)
            frame = self.draw_panel(frame)
            
            # --- UPDATE TRACKER ---
            self.tracker.cleanup()
            self.tracker.increment_frame()
            
            # --- PRINT STATS ---
            if time.time() - self._last_stat_time >= 3:
                self.print_stats()
                self._last_stat_time = time.time()
            
            # --- TAMPILKAN ---
            cv2.imshow("Traffic Vision", frame)
            
            # --- KEYBOARD CONTROL ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False
            elif key == ord('s'):
                self.print_stats()
            elif key == ord('r'):
                self.tracker = VehicleTracker(self.line_y)
                self._start_time = time.time()
                print("[RESET] Semua hitungan direset!")
            elif key == ord('l'):
                self.show_line = not self.show_line
            elif key == ord('u'):
                self.line_y = max(50, self.line_y - 10)
                self.tracker.line_y = self.line_y
                print(f"[LINE] Y={self.line_y}")
            elif key == ord('d'):
                self.line_y = min(self.frame_height - 50, self.line_y + 10)
                self.tracker.line_y = self.line_y
                print(f"[LINE] Y={self.line_y}")
        
        # --- CLEANUP ---
        self.source.release()
        cv2.destroyAllWindows()
        
        print("\n" + "=" * 60)
        print("  SISTEM DIHENTIKAN")
        self.print_stats()
        self.save_report()
        print("=" * 60)


# ============================================================
# 6. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    # TAMPILKAN MODE YANG SEDANG AKTIF
    print(f"\n{'='*60}")
    print(f"  MODE AKTIF: {SOURCE_MODE}")
    if SOURCE_MODE == "VIDEO":
        print(f"  Video Path: {VIDEO_PATH}")
    else:
        print(f"  ESP32 URL : {ESP32_URL}")
    print(f"{'='*60}\n")
    
    app = TrafficVisionApp()
    app.run()