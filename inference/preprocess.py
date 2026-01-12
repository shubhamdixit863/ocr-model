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
