# ============================================================
#  KONFIGURASI
# ============================================================

# ===== PILIH SALAH SATU =====
#SOURCE_MODE = "VIDEO"         # mode video file
SOURCE_MODE = "ESP32-CAM"   # mode ESP32-CAM

# Konfigurasi Video
VIDEO_PATH  = "videos/traffic8.mp4"
VIDEO_LOOP  = True

# Konfigurasi ESP32-CAM
ESP32_URL         = "http://10.136.172.178:/stream"
ESP32_TIMEOUT     = 10
ESP32_RECONNECT_S = 3

# Parameter Deteksi
CONFIDENCE = 0.35
IMGSZ      = 640

# Parameter Counting
CONFIRM_FRAMES = 3
MAX_TRACK_AGE  = 50

# ============================================================
# KONFIGURASI STATUS LALU LINTAS
# ============================================================
# Threshold MACET
CONGESTION_THRESHOLD = 10      # Per jenis kendaraan (misal: jika mobil > 10 = macet)
TOTAL_CONGESTION     = 15      # Total semua kendaraan (misal: jika total > 15 = macet)

# Threshold PADAT (persentase dari threshold macet)
PADAT_THRESHOLD_PERCENT = 0.5  # 50% dari TOTAL_CONGESTION
# Berarti padat jika total >= 7.5 (dibulatkan ke 8)
# Status: 
# - LANCAR: total < 8
# - PADAT: 8 <= total < 15
# - MACET: total >= 15

VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
VEHICLE_COLORS  = {
    "car":        (0, 255, 0),
    "motorcycle": (0, 255, 255),
    "bus":        (255, 100, 100),
    "truck":      (0, 165, 255),
}
DETECT_CLASSES = [2, 3, 5, 7]
REPORT_DIR     = "outputs/reports"

# ============================================================
# KONFIGURASI MQTT
# ============================================================
MQTT_BROKER   = "localhost"
MQTT_PORT     = 1883
MQTT_USERNAME = ""
MQTT_PASSWORD = ""

# Topic PUBLISH (Python → Laravel)
MQTT_TOPIC_DATA      = "traffic/data"
MQTT_TOPIC_STATUS    = "traffic/status"
MQTT_TOPIC_METRICS   = "traffic/metrics"    # fps + latency_ms → camera_metrics
MQTT_TOPIC_DETECTION = "traffic/detection"  # per-box detection → detection_log

# Topic SUBSCRIBE (Laravel → Python)
# Laravel publish ROI ke topic ini setiap user simpan config di dashboard.
# Format JSON yang dikirim Laravel:
# {
#   "camera_id": 1,
#   "x1": 100, "y1": 200,
#   "x2": 540, "y2": 220,
#   "width": 640,    <- resolusi referensi saat ROI digambar di Laravel
#   "height": 480
# }
MQTT_TOPIC_ROI = "traffic-vision/camera-config"

# Interval publish data (detik)
MQTT_PUBLISH_INTERVAL = 2.0

# ============================================================
# KONFIGURASI LARAVEL API
# (dipakai untuk fetch ROI awal saat startup)
# ============================================================
LARAVEL_API_URL         = "http://localhost:8000/api/camera-config"
LARAVEL_CAMERA_ID       = 1
LARAVEL_API_TOKEN       = ""
CONFIG_REFRESH_INTERVAL = 30   # fallback polling jika MQTT ROI tidak aktif

# ============================================================
# KONFIGURASI MJPEG RESTREAM
# ============================================================
MJPEG_HOST    = "0.0.0.0"   # 0.0.0.0 = bisa diakses dari jaringan lain
MJPEG_PORT    = 8081
MJPEG_QUALITY = 75           # kualitas JPEG 1-100
# Akses stream : http://<ip>:8081/stream
# Akses halaman: http://<ip>:8081/