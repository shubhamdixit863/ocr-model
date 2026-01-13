# Kaithi to Hindi TrOCR OCR

End-to-end OCR pipeline for printed Kaithi Lipi using a fine-tuned TrOCR model that outputs Hindi (Devanagari) text.

## Setup

1. Create a Python 3.10+ virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download the Noto Sans Kaithi font and note its path.

## Synthetic Data Generation (Kaithi image -> Hindi label)

```bash
python data/synthetic_generator.py \
  --font-path /path/to/NotoSansKaithi-Regular.ttf \
  --output-images data/raw_images \
  --output-labels data/labels \
  --count 5000
```

By default, Kaithi-to-Devanagari label mapping is loaded from
`data/kaithi_to_devanagari.json`. You can override it:

```bash
python data/synthetic_generator.py \
  --font-path /path/to/NotoSansKaithi-Regular.ttf \
  --mapping-path /path/to/kaithi_to_devanagari.json \
  --output-images data/raw_images \
  --output-labels data/labels \
  --count 5000
```

## Training

```bash
python -m training.train_trocr \
  --images-dir data/raw_images \
  --labels-dir data/labels \
  --output-dir models/kaithi_trocr \
  --epochs 10 \
  --batch-size 8 \
  --fp16
```

## Evaluation

```bash
python -m training.eval_model \
  --model-dir models/kaithi_trocr \
  --images-dir data/raw_images \
  --labels-dir data/labels
```

## Upload Model to S3

Set credentials via env and upload the trained model directory:

```bash
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_DEFAULT_REGION=us-east-1

python scripts/push_model_to_s3.py \
  --model-dir models/kaithi_trocr \
  --bucket your-bucket \
  --prefix kaithi/models/kaithi_trocr
```

## Inference API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Example curl:

```bash
curl -X POST "http://localhost:8000/ocr" \
  -F "file=@/path/to/kaithi_sample.png"
```

Multi-line images are automatically segmented into lines before OCR.

OCR with correction hint (entropy-based):

```bash
curl -X POST "http://localhost:8000/ocr_with_feedback?entropy_threshold=4.0" \
  -F "file=@/path/to/kaithi_sample.png"
```

## Feedback Collection (Optional)

Save user-corrected OCR samples for retraining:

```bash
curl -X POST "http://localhost:8000/feedback" \
  -F "file=@/path/to/kaithi_sample.png" \
  -F "text=𑂍𑂰𑂠𑂲"
```

This stores data under `data/feedback_images/` and `data/feedback_labels/` by default,
and also saves line-level crops under `data/feedback_line_images/` and
`data/feedback_line_labels/`.
You can set `KAITHI_DATA_DIR` to change the base data directory.

When feedback is submitted, the API triggers a background retrain if one is not
already running:

- `status: "training_started"` when a retrain is launched.
- `status: "training_in_progress"` when a retrain is already running.

Retrain behavior can be tuned via env vars:

- `KAITHI_BASE_IMAGES_DIR` / `KAITHI_BASE_LABELS_DIR`
- `KAITHI_LINE_IMAGES_DIR` / `KAITHI_LINE_LABELS_DIR`
- `KAITHI_MODEL_DIR`
- `KAITHI_RETRAIN_EPOCHS`
- `KAITHI_RETRAIN_BATCH_SIZE`
- `KAITHI_RETRAIN_FP16` (`1` to enable)
- `KAITHI_RETRAIN_SMALL_DATA_BOOST` (`1` by default to boost epochs for <20 feedback samples)

## Periodic Retraining (Optional)

Use the feedback data along with the base dataset:

```bash
python -m training.retrain \
  --images-dir data/raw_images \
  --labels-dir data/labels \
  --images-dir data/feedback_images \
  --labels-dir data/feedback_labels \
  --output-dir models/kaithi_trocr \
  --epochs 3
```

## Project Layout

- `data/synthetic_generator.py`: render Kaithi text images with Hindi labels
- `training/dataset.py`: dataset loader returning `{"image": PIL.Image, "text": str}`
- `training/train_trocr.py`: fine-tune TrOCR with Devanagari vocabulary
- `training/eval_model.py`: CER evaluation and sample predictions
- `inference/`: preprocessing, OCR inference, postprocessing
- `api/main.py`: FastAPI endpoint
- `models/kaithi-trocr/`: model output directory
