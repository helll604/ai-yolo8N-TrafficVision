# AI Traffic Vision 🚦

Sistem monitoring lalu lintas berbasis AI menggunakan **YOLOv8n** untuk
deteksi kendaraan realtime dari dua sumber: **file video** (kamera HP / CCTV)
dan **streaming langsung ESP32-CAM**.

---

## Fitur

- Deteksi 4 kelas kendaraan: **Car, Motorcycle, Bus, Truck**
- Penghitungan otomatis saat kendaraan melewati garis ROI
- Tracker kustom dengan matching berbasis jarak + kelas
- Panel statistik realtime di layar (total, per-kelas, FPS, rate/menit)
- Auto-reconnect saat koneksi ESP32-CAM terputus
- Simpan report `.txt` otomatis saat program ditutup
- Kontrol arah hitung: ke bawah / ke atas / dua arah

---

## Arsitektur Sistem

```
[Sumber Video]
  ESP32-CAM (MJPEG stream)  atau  File Video (.mp4 / .avi)
        ↓
  VideoSource.get_frame()
        ↓
  Resize + Enhance (ESP32)
        ↓
  YOLOv8n Inference  (imgsz=320, conf=0.35)
        ↓
  Filter kelas kendaraan  (car/motorcycle/bus/truck)
        ↓
  VehicleTracker.update()
    ├─ Matching by distance + class label
    ├─ Cek crossing ROI line
    └─ Update counts & log
        ↓
  Draw: bounding box, panel, garis ROI
        ↓
  cv2.imshow()  +  tombol keyboard
        ↓  (saat keluar)
  Simpan report → outputs/reports/vehicle_report_YYYYMMDD_HHMMSS.txt
```

---

## Instalasi

```bash
pip install -r requirements.txt
```

> Pastikan `yolov8n.pt` ada di folder yang sama dengan `vehicle_counting.py`.
> Ultralytics akan otomatis download jika tidak ada.

---

## Konfigurasi (vehicle_counting.py)

Semua pengaturan ada di bagian **KONFIGURASI** di atas file:

| Parameter | Default | Keterangan |
|-----------|---------|------------|
| `SOURCE_MODE` | `"VIDEO"` | `"VIDEO"` atau `"ESP32-CAM"` |
| `VIDEO_PATH` | `videos/traffic1.mp4` | Path file video |
| `VIDEO_LOOP` | `True` | Loop video atau berhenti di akhir |
| `ESP32_URL` | `http://...` | Alamat IP stream ESP32-CAM |
| `CONFIDENCE` | `0.35` | Threshold deteksi (jangan di bawah 0.30) |
| `COUNT_DIRECTION` | `"down"` | `"down"`, `"up"`, atau `"both"` |
| `LINE_Y` | otomatis | 65% tinggi frame, tidak perlu diubah manual |

---

## Cara Pakai

**Mode VIDEO:**
```python
SOURCE_MODE = "VIDEO"
VIDEO_PATH  = "videos/traffic1.mp4"
VIDEO_LOOP  = True    # True = loop, False = hentikan saat habis
```

**Mode ESP32-CAM:**
```python
SOURCE_MODE = "ESP32-CAM"
ESP32_URL   = "http://192.168.x.x:81/stream"   # ganti IP sesuai ESP32
```

**Jalankan:**
```bash
python vehicle_counting.py
```

---

## Kontrol Keyboard

| Tombol | Fungsi |
|--------|--------|
| `Q` | Keluar & simpan report |
| `S` | Cetak statistik ke terminal |
| `R` | Reset semua hitungan |
| `L` | Tampilkan / sembunyikan garis ROI |

---

## Tips ESP32-CAM

1. Set resolusi ke **VGA (640×480)** di firmware ESP32 — jangan UXGA/SVGA,
   terlalu berat untuk streaming WiFi.
2. Pastikan ESP32 dan laptop/PC di **jaringan WiFi yang sama**.
3. Kalau gambar sering freeze, perkecil `FRAME_RATE` di sketch Arduino.
4. Sinar di belakang kendaraan (backlight) menurunkan akurasi —
   tempatkan kamera menghadap sumber cahaya jika memungkinkan.

---

## Struktur Folder

```
project/
├── vehicle_counting.py   ← file utama
├── utils.py              ← helper functions
├── bytetrack_temp.yaml   ← config tracker (disiapkan untuk upgrade)
├── requirements.txt
├── yolov8n.pt            ← model YOLO
├── videos/
│   └── traffic1.mp4
└── outputs/
    ├── reports/          ← report .txt tersimpan di sini
    └── results/          ← output video (jika aktifkan writer)
```

---

## Teknologi

- Python 3.10+
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- OpenCV
- NumPy
- ESP32-CAM (AI Thinker / TTGO T-Journal)