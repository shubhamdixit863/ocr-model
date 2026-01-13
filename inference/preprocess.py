import logging

import cv2
import numpy as np
from PIL import Image

LOGGER = logging.getLogger(__name__)


def _deskew(image: np.ndarray) -> np.ndarray:
    """Estimate and correct skew using minimum area rectangle."""
    coords = np.column_stack(np.where(image < 255))
    if coords.size == 0:
        return image
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    if abs(angle) < 0.5:
        return image
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_CUBIC, borderValue=255)


def _segment_lines(binary: np.ndarray) -> list[tuple[int, int]]:
    """Return row ranges for text lines based on horizontal projection."""
    height, width = binary.shape
    row_sum = binary.sum(axis=1)
    min_pixels = max(10, int(0.01 * width))
    min_height = max(10, int(0.02 * height))
    lines: list[tuple[int, int]] = []
    in_line = False
    start = 0
    for idx, value in enumerate(row_sum):
        if value > min_pixels and not in_line:
            start = idx
            in_line = True
        elif value <= min_pixels and in_line:
            end = idx
            if end - start >= min_height:
                lines.append((start, end))
            in_line = False
    if in_line:
        end = height
        if end - start >= min_height:
            lines.append((start, end))
    return lines


def preprocess_image(image_bytes: bytes) -> Image.Image:
    """Decode bytes to RGB PIL image after thresholding and deskew."""
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    deskewed = _deskew(thresh)
    return Image.fromarray(deskewed).convert("RGB")


def preprocess_lines(image_bytes: bytes) -> list[Image.Image]:
    """Decode bytes and return a list of line images for OCR."""
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    deskewed = _deskew(thresh)
    binary = deskewed < 128
    lines = _segment_lines(binary)
    if not lines:
        return [Image.fromarray(deskewed).convert("RGB")]
    padded = []
    pad = 3
    height = deskewed.shape[0]
    for start, end in lines:
        top = max(0, start - pad)
        bottom = min(height, end + pad)
        cropped = deskewed[top:bottom, :]
        padded.append(Image.fromarray(cropped).convert("RGB"))
    return padded
