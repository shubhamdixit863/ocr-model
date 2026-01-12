# Kaithi TrOCR OCR

End-to-end OCR pipeline for printed Kaithi Lipi using a fine-tuned TrOCR model.

## Setup

1. Create a Python 3.10+ virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download the Noto Sans Kaithi font and note its path.

## Synthetic Data Generation

```bash
python data/synthetic_generator.py \
  --font-path /path/to/NotoSansKaithi-Regular.ttf \
  --output-images data/raw_images \
  --output-labels data/labels \
  --count 5000
```

## Training

```bash
python -m training.train_trocr \
  --images-dir data/raw_images \
  --labels-dir data/labels \
  --output-dir models/kaithi-trocr \
  --epochs 10 \
  --batch-size 8 \
  --fp16
```

## Evaluation

```bash
python -m training.eval_model \
  --model-dir models/kaithi-trocr \
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
  --model-dir models/kaithi-trocr \
  --bucket your-bucket \
  --prefix kaithi/models/kaithi-trocr
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

## Feedback Collection (Optional)

Save user-corrected OCR samples for retraining:

```bash
curl -X POST "http://localhost:8000/feedback" \
  -F "file=@/path/to/kaithi_sample.png" \
  -F "text=𑂍𑂰𑂠𑂲"
```

This stores data under `data/feedback_images/` and `data/feedback_labels/` by default.
You can set `KAITHI_DATA_DIR` to change the base data directory.

## Periodic Retraining (Optional)

Use the feedback data along with the base dataset:

```bash
python -m training.retrain \
  --images-dir data/raw_images \
  --labels-dir data/labels \
  --images-dir data/feedback_images \
  --labels-dir data/feedback_labels \
  --output-dir models/kaithi-trocr \
  --epochs 3
```

## Project Layout

- `data/synthetic_generator.py`: render Kaithi text images with augmentations
- `training/dataset.py`: dataset loader returning `{"image": PIL.Image, "text": str}`
- `training/train_trocr.py`: fine-tune TrOCR with Kaithi vocabulary
- `training/eval_model.py`: CER evaluation and sample predictions
- `inference/`: preprocessing, OCR inference, postprocessing
- `api/main.py`: FastAPI endpoint
- `models/kaithi-trocr/`: model output directory
