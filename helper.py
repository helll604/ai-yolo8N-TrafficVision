import cv2
import os

def create_video_writer(video_cap, output_filename):
    os.makedirs("results", exist_ok=True)
    frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps != fps:
        fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))
    return writer

def cornerRect(img, bbox, l=30, t=5, rt=1, colorR=(255,255,255), colorC=(0,0,255)):
    x, y, w, h = bbox
    cv2.line(img, (x, y), (x + l, y), colorC, t)
    cv2.line(img, (x, y), (x, y + l), colorC, t)
    cv2.line(img, (x + w, y), (x + w - l, y), colorC, t)
    cv2.line(img, (x + w, y), (x + w, y + l), colorC, t)
    cv2.line(img, (x, y + h), (x + l, y + h), colorC, t)
    cv2.line(img, (x, y + h), (x, y + h - l), colorC, t)
    cv2.line(img, (x + w, y + h), (x + w - l, y + h), colorC, t)
    cv2.line(img, (x + w, y + h), (x + w, y + h - l), colorC, t)
    return img