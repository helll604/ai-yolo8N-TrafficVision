import os
import cv2


def create_video_writer(video_cap, filename):

    output_dir = "outputs/results"
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, filename)

    frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps = video_cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0 or fps != fps:
        fps = 30

    fourcc = cv2.VideoWriter_fourcc(*"XVID")

    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (frame_width, frame_height)
    )

    return writer


def draw_corner_rect(
    img,
    bbox,
    line_length=30,
    thickness=3,
    color=(0, 255, 255)
):

    x, y, w, h = bbox

    # Top Left
    cv2.line(img, (x, y), (x + line_length, y), color, thickness)
    cv2.line(img, (x, y), (x, y + line_length), color, thickness)

    # Top Right
    cv2.line(
        img,
        (x + w, y),
        (x + w - line_length, y),
        color,
        thickness
    )

    cv2.line(
        img,
        (x + w, y),
        (x + w, y + line_length),
        color,
        thickness
    )

    # Bottom Left
    cv2.line(
        img,
        (x, y + h),
        (x + line_length, y + h),
        color,
        thickness
    )

    cv2.line(
        img,
        (x, y + h),
        (x, y + h - line_length),
        color,
        thickness
    )

    # Bottom Right
    cv2.line(
        img,
        (x + w, y + h),
        (x + w - line_length, y + h),
        color,
        thickness
    )

    cv2.line(
        img,
        (x + w, y + h),
        (x + w, y + h - line_length),
        color,
        thickness
    )

    return img