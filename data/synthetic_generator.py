import argparse
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont

KAITHI_START = 0x11080
KAITHI_END = 0x110CF

LOGGER = logging.getLogger(__name__)


def kaithi_characters() -> List[str]:
    """Return the Kaithi Unicode block as a list of characters."""
    return [chr(code) for code in range(KAITHI_START, KAITHI_END + 1)]


def random_kaithi_text(min_len: int, max_len: int, rng: random.Random) -> str:
    """Sample a random Kaithi string with length in [min_len, max_len]."""
    length = rng.randint(min_len, max_len)
    chars = kaithi_characters()
    return "".join(rng.choice(chars) for _ in range(length))


@dataclass
class RenderConfig:
    width: int
    height: int
    font_size: int
    min_text_len: int
    max_text_len: int
    rotation_deg: float
    noise_sigma: float
    blur_kernel: int
    contrast_range: Tuple[float, float]


class SyntheticKaithiGenerator:
    """Render Kaithi strings into images with light OCR-style augmentations."""
    def __init__(self, font_path: Path, config: RenderConfig, seed: int) -> None:
        self.font_path = font_path
        self.config = config
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.font = ImageFont.truetype(str(font_path), config.font_size)

    def _render_text(self, text: str) -> Image.Image:
        """Render text centered in a grayscale image."""
        image = Image.new("L", (self.config.width, self.config.height), color=255)
        draw = ImageDraw.Draw(image)
        bbox = draw.textbbox((0, 0), text, font=self.font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        x = max(0, (self.config.width - text_w) // 2)
        y = max(0, (self.config.height - text_h) // 2)
        draw.text((x, y), text, font=self.font, fill=0)
        return image

    def _augment(self, image: Image.Image) -> Image.Image:
        """Apply rotation, contrast, noise, and blur."""
        angle = self.rng.uniform(-self.config.rotation_deg, self.config.rotation_deg)
        image = image.rotate(angle, expand=False, fillcolor=255)

        contrast = self.rng.uniform(*self.config.contrast_range)
        image = ImageEnhance.Contrast(image).enhance(contrast)

        img_np = np.array(image).astype(np.float32)
        if self.config.noise_sigma > 0:
            img_np += self.np_rng.normal(0, self.config.noise_sigma, img_np.shape)
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)

        if self.config.blur_kernel > 0:
            k = self.config.blur_kernel
            if k % 2 == 0:
                k += 1
            img_np = cv2.GaussianBlur(img_np, (k, k), 0)

        return Image.fromarray(img_np)

    def generate(self, output_images: Path, output_labels: Path, count: int) -> None:
        """Generate image/label pairs into the given directories."""
        output_images.mkdir(parents=True, exist_ok=True)
        output_labels.mkdir(parents=True, exist_ok=True)
        for idx in range(count):
            text = random_kaithi_text(
                self.config.min_text_len, self.config.max_text_len, self.rng
            )
            image = self._render_text(text)
            image = self._augment(image)

            filename = f"sample_{idx:06d}.png"
            image_path = output_images / filename
            label_path = output_labels / f"sample_{idx:06d}.txt"
            image.save(image_path)
            label_path.write_text(text, encoding="utf-8")

            if idx % 100 == 0:
                LOGGER.info("Generated %s", filename)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic Kaithi OCR data")
    parser.add_argument("--font-path", type=str, required=True)
    parser.add_argument("--output-images", type=str, default="data/raw_images")
    parser.add_argument("--output-labels", type=str, default="data/labels")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--font-size", type=int, default=48)
    parser.add_argument("--min-text-len", type=int, default=4)
    parser.add_argument("--max-text-len", type=int, default=12)
    parser.add_argument("--rotation-deg", type=float, default=3.0)
    parser.add_argument("--noise-sigma", type=float, default=8.0)
    parser.add_argument("--blur-kernel", type=int, default=3)
    parser.add_argument("--contrast-min", type=float, default=0.7)
    parser.add_argument("--contrast-max", type=float, default=1.3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    config = RenderConfig(
        width=args.width,
        height=args.height,
        font_size=args.font_size,
        min_text_len=args.min_text_len,
        max_text_len=args.max_text_len,
        rotation_deg=args.rotation_deg,
        noise_sigma=args.noise_sigma,
        blur_kernel=args.blur_kernel,
        contrast_range=(args.contrast_min, args.contrast_max),
    )
    generator = SyntheticKaithiGenerator(Path(args.font_path), config, args.seed)
    generator.generate(
        Path(args.output_images),
        Path(args.output_labels),
        args.count,
    )


if __name__ == "__main__":
    main()
