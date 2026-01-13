import logging
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from inference.postprocess import postprocess_text
from inference.preprocess import preprocess_lines

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

    def _entropy_score(self, scores) -> float:
        """Compute average token entropy from generate() scores."""
        if not scores:
            return 0.0
        entropies = []
        for step_scores in scores:
            probs = F.softmax(step_scores, dim=-1)
            entropy = -(probs * probs.log()).sum(dim=-1)
            entropies.append(entropy)
        per_batch = torch.stack(entropies, dim=0).mean(dim=0)
        return per_batch.mean().item()

    def predict(self, image_bytes: bytes) -> str:
        """Run preprocessing, inference, and postprocessing."""
        images = preprocess_lines(image_bytes)
        pixel_values = self.processor(images=images, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)
        with torch.no_grad():
            generated = self.model.generate(pixel_values)
        texts = self.processor.batch_decode(generated, skip_special_tokens=True)
        cleaned = [postprocess_text(text) for text in texts]
        return "\n".join([text for text in cleaned if text])

    def predict_with_entropy(self, image_bytes: bytes) -> tuple[str, float]:
        """Run OCR and return text plus average token entropy."""
        images = preprocess_lines(image_bytes)
        pixel_values = self.processor(images=images, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                pixel_values, output_scores=True, return_dict_in_generate=True
            )
        texts = self.processor.batch_decode(
            outputs.sequences, skip_special_tokens=True
        )
        cleaned = [postprocess_text(text) for text in texts]
        entropy = self._entropy_score(outputs.scores)
        return "\n".join([text for text in cleaned if text]), entropy
