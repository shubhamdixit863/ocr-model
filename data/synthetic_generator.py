import argparse
import json
import logging
import os
import random
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont

KAITHI_START = 0x11080
KAITHI_END = 0x110CF
DEVANAGARI_START = 0x0900
DEVANAGARI_END = 0x097F
DEFAULT_MAPPING_PATH = Path(__file__).with_name("kaithi_to_devanagari.json")

LOGGER = logging.getLogger(__name__)


def _build_devanagari_name_map() -> dict[str, str]:
    """Map Devanagari Unicode names (suffix) to characters."""
    name_map: dict[str, str] = {}
    for code in range(DEVANAGARI_START, DEVANAGARI_END + 1):
        ch = chr(code)
        name = unicodedata.name(ch, "")
        if not name.startswith("DEVANAGARI "):
            continue
        name_map[name.replace("DEVANAGARI ", "", 1)] = ch
    return name_map


def _kaithi_to_devanagari_map() -> dict[str, str]:
    """Build a Kaithi->Devanagari mapping using Unicode name matching."""
    devanagari_by_name = _build_devanagari_name_map()
    mapping: dict[str, str] = {}
    for code in range(KAITHI_START, KAITHI_END + 1):
        ch = chr(code)
        name = unicodedata.name(ch, "")
        if not name.startswith("KAITHI "):
            continue
        suffix = name.replace("KAITHI ", "", 1)
        target = devanagari_by_name.get(suffix)
        if target:
            mapping[ch] = target
    return mapping


def _load_mapping(mapping_path: Path) -> dict[str, str]:
    if mapping_path.exists():
        with mapping_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return {str(k): str(v) for k, v in data.items()}
    return _kaithi_to_devanagari_map()


def kaithi_characters(mapping: dict[str, str]) -> List[str]:
    """Return Kaithi characters that have a Devanagari mapping."""
    return sorted(mapping.keys())


def random_kaithi_text(
    min_len: int, max_len: int, rng: random.Random, mapping: dict[str, str]
) -> str:
    """Sample a random Kaithi string with length in [min_len, max_len]."""
    length = rng.randint(min_len, max_len)
    chars = kaithi_characters(mapping)
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
    """Render Kaithi strings into images with Hindi labels."""
    def __init__(
        self, font_path: Path, config: RenderConfig, seed: int, mapping_path: Path
    ) -> None:
        self.font_path = font_path
        self.config = config
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.font = ImageFont.truetype(str(font_path), config.font_size)
        self.mapping = _load_mapping(mapping_path)

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
        """Generate Kaithi image + Hindi text label pairs."""
        output_images.mkdir(parents=True, exist_ok=True)
        output_labels.mkdir(parents=True, exist_ok=True)
        for idx in range(count):
            kaithi_text = random_kaithi_text(
                self.config.min_text_len,
                self.config.max_text_len,
                self.rng,
                self.mapping,
            )
            hindi_text = "".join(self.mapping.get(ch, "") for ch in kaithi_text)
            image = self._render_text(kaithi_text)
            image = self._augment(image)

            filename = f"sample_{idx:06d}.png"
            image_path = output_images / filename
            label_path = output_labels / f"sample_{idx:06d}.txt"
            image.save(image_path)
            label_path.write_text(hindi_text, encoding="utf-8")

            if idx % 100 == 0:
                LOGGER.info("Generated %s", filename)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic Kaithi OCR data")
    parser.add_argument("--font-path", type=str, required=True)
    parser.add_argument("--output-images", type=str, default="data/raw_images")
    parser.add_argument("--output-labels", type=str, default="data/labels")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--mapping-path", type=str, default=str(DEFAULT_MAPPING_PATH))
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
    generator = SyntheticKaithiGenerator(
        Path(args.font_path), config, args.seed, Path(args.mapping_path)
    )
    generator.generate(
        Path(args.output_images),
        Path(args.output_labels),
        args.count,
    )


if __name__ == "__main__":
    main()
