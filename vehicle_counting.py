import cv2
import time
import threading
import urllib.request
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
from datetime import datetime

# ─────────────────────────────────────────────
#  KONFIGURASI — ubah di sini
# ─────────────────────────────────────────────
SOURCE_MODE   = "VIDEO"                    # "VIDEO" atau "ESP32-CAM"
VIDEO_PATH    = "videos/traffic1.mp4"
ESP32_URL     = "http://10.188.168.178:81/stream"

FRAME_WIDTH   = 800
FRAME_HEIGHT  = 600
LINE_Y        = 450   
LINE_X        = 400 # vertikal ditengah                         # garis hitung kendaraan

CONFIDENCE        = 0.20
TRACKING_DISTANCE = 80
MAX_TRACK_AGE     = 30

# Warna bounding box per kelas (BGR)
VEHICLE_COLORS = {
    "car":        (0, 255, 0),
    "motorcycle": (0, 255, 255),
    "bus":        (255, 0, 0),
    "truck":      (0, 165, 255),
}

# ID kelas YOLO → nama
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


# ─────────────────────────────────────────────
#  ESP32-CAM READER (thread terpisah)
#  - Selalu ambil frame terbaru, buang yang lama
#  - Tidak block main loop saat jaringan lambat
# ─────────────────────────────────────────────
class ESP32StreamReader:
    def __init__(self, url: str):
        self.url    = url
        self._frame = None
        self._lock  = threading.Lock()
        self._stop  = threading.Event()
        self._connected = False

    def connect(self) -> bool:
        try:
            self._stream = urllib.request.urlopen(self.url, timeout=5)
            self._connected = True
            t = threading.Thread(target=self._read_loop, daemon=True)
            t.start()
            print(f"[ESP32] Terhubung: {self.url}")
            return True
        except Exception as e:
            print(f"[ESP32] Gagal connect: {e}")
            return False

    def _read_loop(self):
        buf = b""
        while not self._stop.is_set():
            try:
                chunk = self._stream.read(16384)   # baca lebih besar → lebih lancar
                if not chunk:
                    break
                buf += chunk

                # Cari JPEG lengkap di buffer
                while True:
                    start = buf.find(b"\xff\xd8")
                    end   = buf.find(b"\xff\xd9")
                    if start == -1 or end == -1 or end <= start:
                        break

                    jpg = buf[start:end + 2]
                    # Langsung buang semua data lama setelah frame ini
                    # → efek "skip frame": selalu pakai frame terbaru
                    buf = buf[end + 2:]

                    frame = cv2.imdecode(
                        np.frombuffer(jpg, dtype=np.uint8),
                        cv2.IMREAD_COLOR,
                    )
                    if frame is not None:
                        with self._lock:
                            self._frame = frame   # simpan frame terbaru

            except Exception:
                break

        self._connected = False

    def read(self):
        """Kembalikan frame terbaru. None jika belum ada."""
        with self._lock:
            if self._frame is None:
                return None
            frame = self._frame.copy()
            self._frame = None   # reset agar caller tahu kapan ada frame baru
            return frame

    def is_connected(self) -> bool:
        return self._connected

    def release(self):
        self._stop.set()
        try:
            self._stream.close()
        except Exception:
            pass


# ─────────────────────────────────────────────
#  VIDEO SOURCE — wrapper VIDEO & ESP32
# ─────────────────────────────────────────────
class VideoSource:
    def __init__(self, mode: str):
        self.mode  = mode
        self._cap   = None
        self._esp32 = None

    def connect(self) -> bool:
        if self.mode == "VIDEO":
            self._cap = cv2.VideoCapture(VIDEO_PATH)
            if not self._cap.isOpened():
                print(f"[VIDEO] Tidak bisa buka: {VIDEO_PATH}")
                return False
            # Kurangi buffer internal OpenCV agar tidak delay
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            print(f"[VIDEO] Loaded: {VIDEO_PATH}")
            return True

        elif self.mode == "ESP32-CAM":
            self._esp32 = ESP32StreamReader(ESP32_URL)
            return self._esp32.connect()

        print(f"[SOURCE] Mode tidak dikenal: {self.mode} (gunakan 'VIDEO' atau 'ESP32-CAM')")
        return False

    def get_frame(self):
        if self.mode == "VIDEO":
            ret, frame = self._cap.read()
            if not ret:
                # Loop video dari awal
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self._cap.read()
            return frame if ret else None

        elif self.mode == "ESP32-CAM":
            if not self._esp32.is_connected():
                return None
            return self._esp32.read()   # bisa None kalau belum ada frame baru

        return None

    def release(self):
        if self._cap:
            self._cap.release()
        if self._esp32:
            self._esp32.release()


# ─────────────────────────────────────────────
#  VEHICLE TRACKER
# ─────────────────────────────────────────────
class VehicleTracker:
    def __init__(self):
        self.tracks          = {}
        self.counted_ids     = set()
        self.counts          = defaultdict(int)
        self.crossing_events = []
        self.next_id         = 0
        self.frame_count     = 0

    # ── helpers ──────────────────────────────
    def _new_id(self) -> int:
        self.next_id += 1
        return self.next_id

    def _find_match(self, cx: int, cy: int, label: str):
        best_id   = None
        best_dist = TRACKING_DISTANCE

        for tid, track in self.tracks.items():
            if track["crossed"] or track["label"] != label:
                continue
            if track["positions"]:
                px, py = track["positions"][-1]
                dist = np.hypot(px - cx, py - cy)
                if dist < best_dist:
                    best_dist = dist
                    best_id   = tid

        return best_id

    # ── public API ───────────────────────────
    def update(self, label: str, cx: int, cy: int) -> bool:
        tid = self._find_match(cx, cy, label)

        if tid is None:
            tid = self._new_id()
            self.tracks[tid] = {
                "id":        tid,
                "label":     label,
                "positions": deque(maxlen=30),
                "crossed":   False,
                "last_seen": self.frame_count,
            }

        track = self.tracks[tid]
        track["last_seen"] = self.frame_count

        crossed = False
        if not track["crossed"] and track["positions"]:
            prev_cx, prev_cy = track["positions"][-1]
            if (prev_cy < LINE_Y <= cy) or (prev_cy > LINE_Y >= cy):
                track["crossed"] = True
                self.counted_ids.add(tid)
                self.counts[label] += 1
                crossed = True
                self.crossing_events.append({
                    "id":       tid,
                    "label":    label,
                    "time":     datetime.now(),
                    "position": (cx, cy),
                })
                print(f"  [CROSSED] {label.upper()} | Total {self.counts[label]}")

        track["positions"].append((cx, cy))
        return crossed

    def cleanup_old_tracks(self):
        stale = [
            tid for tid, t in self.tracks.items()
            if not t["crossed"] and (self.frame_count - t["last_seen"]) > MAX_TRACK_AGE
        ]
        for tid in stale:
            if tid not in self.counted_ids:
                del self.tracks[tid]

    def increment_frame(self):
        self.frame_count += 1

    def get_stats(self) -> dict:
        total  = sum(self.counts.values())
        active = sum(1 for t in self.tracks.values() if not t["crossed"])

        rate = 0.0
        if len(self.crossing_events) > 1:
            t0  = self.crossing_events[0]["time"]
            t1  = self.crossing_events[-1]["time"]
            dur = (t1 - t0).total_seconds() / 60.0
            rate = len(self.crossing_events) / max(dur, 1.0)
        elif len(self.crossing_events) == 1:
            rate = 1.0

        return {
            "total":       total,
            "active":      active,
            "per_vehicle": dict(self.counts),
            "rate":        rate,
        }

    def reset(self):
        self.__init__()
        print("[TRACKER] Reset!")


# ─────────────────────────────────────────────
#  MAIN APP
# ─────────────────────────────────────────────
class TrafficVisionApp:
    def __init__(self):
        self.model       = None
        self.tracker     = VehicleTracker()
        self.source      = VideoSource(SOURCE_MODE)
        self.fps_history = deque(maxlen=30)
        self.running     = True

        # Untuk mode ESP32: jangan paksa 30 fps, ikuti kecepatan stream
        # Untuk mode VIDEO : bisa lebih tinggi
        self._is_esp32 = (SOURCE_MODE == "ESP32-CAM")

    # ── model ────────────────────────────────
    def load_model(self):
        print("Memuat model YOLOv8n …")
        t0 = time.time()
        self.model = YOLO("yolov8n.pt")
        print(f"Model siap ({time.time() - t0:.1f} detik)")

    # ── pemrosesan frame ─────────────────────
    def process_frame(self, frame):
        # Resize dulu
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        # Untuk ESP32: sedikit enhance kontras (opsional, ringan)
        if self._is_esp32:
            frame = cv2.convertScaleAbs(frame, alpha=1.1, beta=15)

        t0 = time.time()
        results = self.model(
            frame,
            imgsz=320,      # ← 320 jauh lebih cepat dari 640, cukup untuk deteksi
            conf=CONFIDENCE,
            iou=0.45,
            verbose=False,
        )[0]
        inf_ms = (time.time() - t0) * 1000

        fps = 1000.0 / inf_ms if inf_ms > 0 else 0
        self.fps_history.append(fps)
        return frame, results

    # ── gambar ───────────────────────────────
    def _draw_line(self, frame):
        cv2.line(frame, (0, LINE_Y), (FRAME_WIDTH, LINE_Y), (0, 0, 255), 3)
        return frame

    def _draw_detections(self, frame, results):
        if results is None or results.boxes is None:
            return frame

        for box in results.boxes:
            conf = float(box.conf[0])
            cls  = int(box.cls[0])
            if conf < CONFIDENCE or cls not in VEHICLE_CLASSES:
                continue

            label = VEHICLE_CLASSES[cls]
            color = VEHICLE_COLORS[label]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            crossed = self.tracker.update(label, cx, cy)

            # Bounding box + titik tengah
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, (cx, cy), 4, color, -1)

            # Label
            text = f"{label.upper()} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
            cv2.putText(frame, text, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            # Notifikasi crossing
            if crossed:
                cv2.putText(frame, f"+1 {label.upper()}", (cx - 40, cy - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        return frame

    def _draw_panel(self, frame):
        stats   = self.tracker.get_stats()
        avg_fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0

        # Latar semi-transparan
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (285, 225), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
        cv2.rectangle(frame, (0, 0), (285, 225), (200, 200, 200), 1)

        def put(text, x, y, scale=0.55, color=(255, 255, 255), thick=1):
            cv2.putText(frame, text, (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)

        put("TRAFFIC VISION", 15, 30, 0.7, (0, 255, 255), 2)

        fps_color = (0, 255, 0) if avg_fps >= 20 else (0, 165, 255)
        put(f"FPS: {avg_fps:.1f}", 15, 55, color=fps_color)
        put(f"TOTAL: {stats['total']}", 15, 80, 0.65, (255, 255, 255), 2)

        y = 105
        for i, (key, label) in enumerate(zip(
            ["car", "motorcycle", "bus", "truck"],
            ["CAR", "MOTORCYCLE", "BUS", "TRUCK"],
        )):
            count = stats["per_vehicle"].get(key, 0)
            pct   = (count / stats["total"] * 100) if stats["total"] > 0 else 0
            put(f"{label}: {count} ({pct:.1f}%)", 15, y + i * 22)

        put(f"ACTIVE: {stats['active']}",         15, y + 4 * 22)
        put(f"RATE: {stats['rate']:.1f}/min",      15, y + 5 * 22, 0.5, (200, 200, 200))

        # Label source mode
        mode_label = f"SRC: {SOURCE_MODE}"
        cv2.putText(frame, mode_label, (FRAME_WIDTH - 130, FRAME_HEIGHT - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

        return frame

    def _print_stats(self):
        s = self.tracker.get_stats()
        now = datetime.now().strftime("%H:%M:%S")
        print(f"\n[{now}] STATS — Total:{s['total']} "
              f"Car:{s['per_vehicle'].get('car',0)} "
              f"Truck:{s['per_vehicle'].get('truck',0)} "
              f"Bus:{s['per_vehicle'].get('bus',0)} "
              f"Moto:{s['per_vehicle'].get('motorcycle',0)} "
              f"| Active:{s['active']} Rate:{s['rate']:.1f}/min")

    # ── main loop ────────────────────────────
    def run(self):
        print("=" * 50)
        print(" AI TRAFFIC VISION SYSTEM")
        print(f" Mode  : {SOURCE_MODE}")
        if SOURCE_MODE == "VIDEO":            print(f" File  : {VIDEO_PATH}")
        else:
            print(f" URL   : {ESP32_URL}")
        print("=" * 50)

        self.load_model()

        if not self.source.connect():
            print("[ERROR] Tidak bisa membuka sumber video!")
            return

        print("Sistem SIAP | Q=Keluar  S=Statistik  R=Reset\n")

        last_stat_time = time.time()
        no_frame_count = 0   # hitung berapa kali tidak dapat frame (ESP32)

        while self.running:
            frame = self.source.get_frame()

            # Untuk ESP32: kalau belum ada frame baru, tunggu sebentar
            if frame is None:
                if self._is_esp32:
                    no_frame_count += 1
                    if no_frame_count > 200:
                        print("[ESP32] Tidak menerima frame, cek koneksi!")
                        no_frame_count = 0
                    time.sleep(0.01)
                    continue
                else:
                    continue

            no_frame_count = 0

            # Proses
            frame, results = self.process_frame(frame)
            frame = self._draw_line(frame)
            frame = self._draw_detections(frame, results)
            frame = self._draw_panel(frame)

            self.tracker.cleanup_old_tracks()
            self.tracker.increment_frame()

            # Print statistik setiap 5 detik
            if time.time() - last_stat_time >= 5:
                self._print_stats()
                last_stat_time = time.time()

            cv2.imshow("AI Traffic Vision System", frame)

            # Delay: VIDEO pakai 1ms, ESP32 pakai 1ms juga (frame rate dikontrol stream)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                self.running = False
            elif key == ord("s"):
                self._print_stats()
            elif key == ord("r"):
                self.tracker.reset()

        # Bersih-bersih
        self.source.release()
        cv2.destroyAllWindows()
        print("\nSistem dihentikan.")
        self._print_stats()


# ─────────────────────────────────────────────
if __name__ == "__main__":
    app = TrafficVisionApp()
    app.run()