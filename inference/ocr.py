import logging
from typing import Optional

import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from inference.postprocess import postprocess_text
from inference.preprocess import preprocess_image

LOGGER = logging.getLogger(__name__)


class KaithiOCR:
    """Load a TrOCR model and run OCR on Kaithi images."""
    def __init__(self, model_dir: str, device: Optional[str] = None) -> None:
        self.processor = TrOCRProcessor.from_pretrained(model_dir)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_dir)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, image_bytes: bytes) -> str:
        """Run preprocessing, inference, and postprocessing."""
        image = preprocess_image(image_bytes)
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)
        with torch.no_grad():
            generated = self.model.generate(pixel_values)
        text = self.processor.batch_decode(generated, skip_special_tokens=True)[0]
        return postprocess_text(text)
