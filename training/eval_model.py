import argparse
import logging
from typing import Dict, List

import evaluate
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from training.dataset import load_kaithi_dataset

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate Kaithi TrOCR")
    parser.add_argument("--model-dir", type=str, default="models/kaithi-trocr")
    parser.add_argument("--images-dir", type=str, default="data/raw_images")
    parser.add_argument("--labels-dir", type=str, default="data/labels")
    parser.add_argument("--max-samples", type=int, default=100)
    return parser.parse_args()


def predict(
    model: VisionEncoderDecoderModel,
    processor: TrOCRProcessor,
    images: List[Image.Image],
    device: torch.device,
) -> List[str]:
    """Run model generation on a list of images."""
    pixel_values = processor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)
    with torch.no_grad():
        generated = model.generate(pixel_values)
    return processor.batch_decode(generated, skip_special_tokens=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = TrOCRProcessor.from_pretrained(args.model_dir)
    model = VisionEncoderDecoderModel.from_pretrained(args.model_dir).to(device)
    dataset = load_kaithi_dataset(args.images_dir, args.labels_dir)

    cer = evaluate.load("cer")
    samples = dataset.select(range(min(args.max_samples, len(dataset))))

    images = [sample["image"].convert("RGB") for sample in samples]
    references = [sample["text"] for sample in samples]
    predictions = predict(model, processor, images, device)

    score = cer.compute(predictions=predictions, references=references)
    LOGGER.info("CER: %.4f", score)

    for idx, (pred, ref) in enumerate(zip(predictions[:10], references[:10])):
        LOGGER.info("Sample %d", idx)
        LOGGER.info("  Pred: %s", pred)
        LOGGER.info("  Ref : %s", ref)


if __name__ == "__main__":
    main()
