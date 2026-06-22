"""
utils.py — Helper functions untuk AI Traffic Vision System
"""

import os
import cv2


def create_video_writer(video_cap, filename: str, output_dir: str = "outputs/results"):
    """
    Buat VideoWriter dari sumber video yang sudah dibuka.
    Output disimpan ke output_dir/filename.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    frame_width  = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps          = video_cap.get(cv2.CAP_PROP_FPS)

    # Fallback FPS kalau tidak terbaca (sering terjadi di ESP32-CAM)
    if fps <= 0 or fps != fps:
        fps = 25.0

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    return writer


def draw_corner_rect(img, bbox, line_length: int = 28, thickness: int = 2,
                     color: tuple = (0, 255, 255)):
    """
    Gambar bounding box gaya 'sudut saja' (tanpa sisi penuh).
    bbox = (x, y, w, h)
    """
    x, y, w, h = bbox
    pts = [
        # TL
        ((x, y), (x + line_length, y)),
        ((x, y), (x, y + line_length)),
        # TR
        ((x + w, y), (x + w - line_length, y)),
        ((x + w, y), (x + w, y + line_length)),
        # BL
        ((x, y + h), (x + line_length, y + h)),
        ((x, y + h), (x, y + h - line_length)),
        # BR
        ((x + w, y + h), (x + w - line_length, y + h)),
        ((x + w, y + h), (x + w, y + h - line_length)),
    ]
    for p1, p2 in pts:
        cv2.line(img, p1, p2, color, thickness)
    return img


def enhance_frame(frame, alpha: float = 1.15, beta: int = 10):
    """
    Tingkatkan kontras & kecerahan frame.
    Berguna untuk gambar ESP32-CAM yang sering gelap.
    alpha: kontras (1.0 = normal, >1 = lebih kontras)
    beta : kecerahan tambahan (0–50 wajar)
    """
    return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)


def put_text_with_bg(img, text: str, pos: tuple,
                     font_scale: float = 0.5,
                     text_color: tuple = (255, 255, 255),
                     bg_color: tuple = (0, 0, 0),
                     thickness: int = 1,
                     padding: int = 4):
    """
    Tulis teks dengan latar belakang agar mudah terbaca di segala kondisi.
    pos = (x, y) sudut kiri atas teks.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    cv2.rectangle(img,
                  (x - padding, y - th - padding),
                  (x + tw + padding, y + baseline + padding),
                  bg_color, -1)
    cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness)
    return img