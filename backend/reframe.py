"""
Face-aware auto-reframing. Uses OpenCV's bundled Haar cascade (ships inside
opencv-python — no download needed) to find the dominant speaker face across
a clip and center the crop on it. Falls back to a plain center-crop when no
face is detected, which is the common, legitimate fallback path (slides,
screen-share, wide shots, etc.).
"""
from __future__ import annotations

import cv2
import numpy as np

_face_cascade = None


def _get_cascade():
    global _face_cascade
    if _face_cascade is None:
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _face_cascade = cv2.CascadeClassifier(path)
    return _face_cascade


def detect_face_center_x(video_path: str, start: float, end: float, samples: int = 8):
    """Samples frames across [start,end] and returns the median normalized
    (0-1) x-position of the largest detected face, plus source width/height.
    Returns (None, w, h) if no face was found in any sampled frame.
    """
    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    cascade = _get_cascade()

    centers = []
    span = max(0.001, end - start)
    for k in range(samples):
        t = start + span * (k + 0.5) / samples
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        min_size = max(24, int(w * 0.06))
        faces = cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5,
                                          minSize=(min_size, min_size))
        if len(faces):
            fx, fy, fw, fh = max(faces, key=lambda r: r[2] * r[3])
            centers.append((fx + fw / 2) / w)
    cap.release()

    if not centers:
        return None, w, h
    return float(np.median(centers)), w, h


def crop_rect_for_aspect(src_w: int, src_h: int, target_w: int, target_h: int,
                          center_x_norm: float | None):
    """Returns (crop_w, crop_h, x, y) to crop src_w x src_h down to the given
    aspect ratio, centered horizontally on center_x_norm (0-1, defaults to 0.5)
    and vertically in the middle.
    """
    target_ratio = target_w / target_h
    src_ratio = src_w / src_h

    if src_ratio > target_ratio:
        # source is wider than target -> crop width, keep full height
        crop_h = src_h
        crop_w = int(round(crop_h * target_ratio))
    else:
        # source is taller/narrower -> crop height, keep full width
        crop_w = src_w
        crop_h = int(round(crop_w / target_ratio))

    cx = 0.5 if center_x_norm is None else center_x_norm
    x = int(round(cx * src_w - crop_w / 2))
    x = max(0, min(x, src_w - crop_w))
    y = int(round((src_h - crop_h) / 2))
    y = max(0, min(y, src_h - crop_h))
    return crop_w, crop_h, x, y
