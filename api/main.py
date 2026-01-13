import logging
import os
from pathlib import Path
from uuid import uuid4

import json
import subprocess
import sys
import threading
import time

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from inference.ocr import KaithiOCR
from inference.preprocess import preprocess_lines

LOGGER = logging.getLogger(__name__)

app = FastAPI(title="Kaithi OCR API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
ocr_model: KaithiOCR | None = None
DATA_DIR = Path(os.getenv("KAITHI_DATA_DIR", "data"))
FEEDBACK_IMAGES_DIR = DATA_DIR / "feedback_images"
FEEDBACK_LABELS_DIR = DATA_DIR / "feedback_labels"
FEEDBACK_LINE_IMAGES_DIR = DATA_DIR / "feedback_line_images"
FEEDBACK_LINE_LABELS_DIR = DATA_DIR / "feedback_line_labels"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_LOCK = DATA_DIR / "training.lock"


def _acquire_train_lock() -> bool:
    try:
        fd = os.open(TRAIN_LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(f"{os.getpid()} {int(time.time())}\n")
    return True


def _release_train_lock() -> None:
    try:
        TRAIN_LOCK.unlink()
    except FileNotFoundError:
        pass


def _count_feedback_samples() -> int:
    if not FEEDBACK_LINE_LABELS_DIR.exists():
        return 0
    return len(list(FEEDBACK_LINE_LABELS_DIR.glob("*.txt")))


def _start_retrain() -> None:
    base_images = Path(os.getenv("KAITHI_BASE_IMAGES_DIR", DATA_DIR / "raw_images"))
    base_labels = Path(os.getenv("KAITHI_BASE_LABELS_DIR", DATA_DIR / "labels"))
    line_images = Path(os.getenv("KAITHI_LINE_IMAGES_DIR", FEEDBACK_LINE_IMAGES_DIR))
    line_labels = Path(os.getenv("KAITHI_LINE_LABELS_DIR", FEEDBACK_LINE_LABELS_DIR))
    model_dir = os.getenv("KAITHI_MODEL_DIR", "models/kaithi_trocr")
    epochs = os.getenv("KAITHI_RETRAIN_EPOCHS", "1")
    batch_size = os.getenv("KAITHI_RETRAIN_BATCH_SIZE", "4")
    fp16 = os.getenv("KAITHI_RETRAIN_FP16", "0").lower() in {"1", "true", "yes"}

    cmd = [sys.executable, "-m", "training.retrain"]
    if base_images.exists() and base_labels.exists():
        cmd += ["--images-dir", str(base_images), "--labels-dir", str(base_labels)]
    if line_images.exists() and line_labels.exists():
        cmd += ["--images-dir", str(line_images), "--labels-dir", str(line_labels)]
    cmd += ["--images-dir", str(FEEDBACK_IMAGES_DIR), "--labels-dir", str(FEEDBACK_LABELS_DIR)]
    if os.getenv("KAITHI_RETRAIN_SMALL_DATA_BOOST", "1").lower() in {"1", "true", "yes"}:
        feedback_count = _count_feedback_samples()
        if feedback_count > 0 and feedback_count < 20:
            epochs = str(max(int(epochs), 50))
    cmd += ["--output-dir", model_dir, "--epochs", epochs, "--batch-size", batch_size]
    if fp16:
        cmd.append("--fp16")

    process = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT))

    def _wait_and_release() -> None:
        try:
            process.wait()
        finally:
            _release_train_lock()

    thread = threading.Thread(target=_wait_and_release, daemon=True)
    thread.start()


@app.on_event("startup")
def load_model() -> None:
    """Load the OCR model on startup."""
    global ocr_model
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    FEEDBACK_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    FEEDBACK_LABELS_DIR.mkdir(parents=True, exist_ok=True)
    FEEDBACK_LINE_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    FEEDBACK_LINE_LABELS_DIR.mkdir(parents=True, exist_ok=True)
    ocr_model = KaithiOCR(model_dir="models/kaithi_trocr")


@app.post("/ocr")
async def ocr_endpoint(file: UploadFile = File(...)) -> JSONResponse:
    """Run OCR on an uploaded image file."""
    if ocr_model is None:
        return JSONResponse(status_code=503, content={"error": "Model not loaded"})
    image_bytes = await file.read()
    try:
        text = ocr_model.predict(image_bytes)
        return JSONResponse(content={"text": text})
    except Exception as exc:
        LOGGER.exception("OCR failed")
        return JSONResponse(status_code=400, content={"error": str(exc)})


@app.post("/ocr_with_feedback")
async def ocr_with_feedback_endpoint(
    file: UploadFile = File(...), entropy_threshold: float = 4.0
) -> JSONResponse:
    """Run OCR and return a correction hint for the client."""
    if ocr_model is None:
        return JSONResponse(status_code=503, content={"error": "Model not loaded"})
    image_bytes = await file.read()
    try:
        text, entropy = ocr_model.predict_with_entropy(image_bytes)
        needs_correction = entropy > entropy_threshold
        return JSONResponse(
            content={
                "text": text,
                "needs_correction": needs_correction,
                "entropy": entropy,
            }
        )
    except Exception as exc:
        LOGGER.exception("OCR failed")
        return JSONResponse(status_code=400, content={"error": str(exc)})


@app.post("/feedback")
async def feedback_endpoint(
    file: UploadFile = File(...),
    text: str = Form(...),
    predicted_text: str | None = Form(None),
) -> JSONResponse:
    """Store image and corrected text for retraining."""
    image_bytes = await file.read()
    suffix = Path(file.filename or "upload.png").suffix or ".png"
    sample_id = uuid4().hex
    image_path = FEEDBACK_IMAGES_DIR / f"{sample_id}{suffix}"
    label_path = FEEDBACK_LABELS_DIR / f"{sample_id}.txt"
    meta_path = FEEDBACK_LABELS_DIR / f"{sample_id}.json"
    try:
        image_path.write_bytes(image_bytes)
        label_path.write_text(text, encoding="utf-8")
        meta = {"corrected_text": text}
        if predicted_text is not None:
            meta["predicted_text"] = predicted_text
        meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        _save_line_feedback(sample_id, image_bytes, text)
        if _acquire_train_lock():
            _start_retrain()
            return JSONResponse(
                content={"id": sample_id, "status": "training_started"}
            )
        return JSONResponse(content={"id": sample_id, "status": "training_in_progress"})
    except Exception as exc:
        LOGGER.exception("Feedback save failed")
        return JSONResponse(status_code=400, content={"error": str(exc)})


def _save_line_feedback(sample_id: str, image_bytes: bytes, text: str) -> None:
    """Split image into lines and save line-level crops/labels."""
    lines = preprocess_lines(image_bytes)
    text_lines = [line.strip() for line in text.splitlines() if line.strip()]
    line_count = min(len(lines), len(text_lines))
    for idx in range(line_count):
        line_id = f"{sample_id}_l{idx:02d}"
        image_path = FEEDBACK_LINE_IMAGES_DIR / f"{line_id}.png"
        label_path = FEEDBACK_LINE_LABELS_DIR / f"{line_id}.txt"
        lines[idx].save(image_path)
        label_path.write_text(text_lines[idx], encoding="utf-8")
