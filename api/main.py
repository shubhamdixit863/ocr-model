import logging
import os
from pathlib import Path
from uuid import uuid4

import json

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from inference.ocr import KaithiOCR

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


@app.on_event("startup")
def load_model() -> None:
    """Load the OCR model on startup."""
    global ocr_model
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    FEEDBACK_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    FEEDBACK_LABELS_DIR.mkdir(parents=True, exist_ok=True)
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
        return JSONResponse(content={"id": sample_id, "status": "saved"})
    except Exception as exc:
        LOGGER.exception("Feedback save failed")
        return JSONResponse(status_code=400, content={"error": str(exc)})
