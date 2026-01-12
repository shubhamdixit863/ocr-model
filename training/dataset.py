import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from datasets import Dataset, concatenate_datasets
from PIL import Image

LOGGER = logging.getLogger(__name__)


@dataclass
class DataPaths:
    images_dir: Path
    labels_dir: Path


def _collect_samples(paths: DataPaths) -> List[Tuple[str, str]]:
    """Collect image paths and label strings paired by filename."""
    image_files = sorted(paths.images_dir.glob("*.png"))
    samples = []
    for image_path in image_files:
        label_path = paths.labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            LOGGER.warning("Missing label for %s", image_path.name)
            continue
        text = label_path.read_text(encoding="utf-8").strip()
        samples.append((str(image_path), text))
    return samples


def load_kaithi_dataset(images_dir: str, labels_dir: str) -> Dataset:
    """Load a Hugging Face Dataset that yields PIL images and Kaithi text."""
    paths = DataPaths(Path(images_dir), Path(labels_dir))
    samples = _collect_samples(paths)
    LOGGER.info("Loaded %d samples", len(samples))

    def _gen():
        for image_path, text in samples:
            image = Image.open(image_path).convert("RGB")
            yield {"image": image, "text": text}

    return Dataset.from_generator(_gen)


def load_multi_kaithi_dataset(images_dirs: List[str], labels_dirs: List[str]) -> Dataset:
    """Load and concatenate multiple Kaithi datasets."""
    if len(images_dirs) != len(labels_dirs):
        raise ValueError("images_dirs and labels_dirs must have the same length")
    datasets = []
    for images_dir, labels_dir in zip(images_dirs, labels_dirs):
        datasets.append(load_kaithi_dataset(images_dir, labels_dir))
    if not datasets:
        raise ValueError("No datasets provided")
    if len(datasets) == 1:
        return datasets[0]
    return concatenate_datasets(datasets)
