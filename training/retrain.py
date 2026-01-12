import argparse
import logging
from dataclasses import dataclass
from typing import Any, Dict, List

import evaluate
import torch
from datasets import Dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
)

from training.dataset import load_multi_kaithi_dataset

LOGGER = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    model_name: str
    output_dir: str
    images_dirs: List[str]
    labels_dirs: List[str]
    max_target_length: int
    per_device_batch_size: int
    num_train_epochs: int
    learning_rate: float
    fp16: bool
    seed: int


def add_kaithi_tokens(processor: TrOCRProcessor) -> int:
    """Add Kaithi Unicode block characters to the tokenizer."""
    chars = [chr(code) for code in range(0x11080, 0x110D0)]
    added = processor.tokenizer.add_tokens(chars)
    LOGGER.info("Added %d Kaithi tokens", added)
    return added


def preprocess_dataset(dataset: Dataset, processor: TrOCRProcessor, max_target_length: int) -> Dataset:
    """Convert images/text into model-ready pixel values and labels."""
    def _map(example: Dict[str, Any]) -> Dict[str, Any]:
        pixel = processor(images=example["image"], return_tensors="pt").pixel_values[0]
        labels = processor.tokenizer(
            example["text"],
            padding="max_length",
            max_length=max_target_length,
            truncation=True,
        ).input_ids
        labels = [label if label != processor.tokenizer.pad_token_id else -100 for label in labels]
        return {"pixel_values": pixel, "labels": labels}

    mapped = dataset.map(_map, remove_columns=["image", "text"])
    return mapped.with_format("torch")


def build_compute_metrics(processor: TrOCRProcessor):
    """Build a CER metrics function bound to the provided processor."""
    cer = evaluate.load("cer")

    def _compute(eval_pred) -> Dict[str, float]:
        predictions, labels = eval_pred
        labels = [
            [
                label if label != -100 else processor.tokenizer.pad_token_id
                for label in seq
            ]
            for seq in labels
        ]
        pred_texts = processor.tokenizer.batch_decode(
            predictions, skip_special_tokens=True
        )
        label_texts = processor.tokenizer.batch_decode(
            labels, skip_special_tokens=True
        )
        value = cer.compute(predictions=pred_texts, references=label_texts)
        return {"cer": value}

    return _compute


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for retraining."""
    parser = argparse.ArgumentParser(description="Retrain Kaithi TrOCR with feedback")
    parser.add_argument("--images-dir", action="append", required=True)
    parser.add_argument("--labels-dir", action="append", required=True)
    parser.add_argument("--output-dir", type=str, default="models/kaithi-trocr")
    parser.add_argument("--model-name", type=str, default="models/kaithi-trocr")
    parser.add_argument("--max-target-length", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-split", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    config = TrainConfig(
        model_name=args.model_name,
        output_dir=args.output_dir,
        images_dirs=args.images_dir,
        labels_dirs=args.labels_dir,
        max_target_length=args.max_target_length,
        per_device_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        fp16=args.fp16,
        seed=args.seed,
    )

    dataset = load_multi_kaithi_dataset(config.images_dirs, config.labels_dirs)
    split = dataset.train_test_split(test_size=args.eval_split, seed=config.seed)

    processor = TrOCRProcessor.from_pretrained(config.model_name)
    add_kaithi_tokens(processor)

    model = VisionEncoderDecoderModel.from_pretrained(config.model_name)
    model.decoder.resize_token_embeddings(len(processor.tokenizer))
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.sep_token_id

    train_ds = preprocess_dataset(split["train"], processor, config.max_target_length)
    eval_ds = preprocess_dataset(split["test"], processor, config.max_target_length)

    def _collate(features: List[Dict[str, Any]]) -> Dict[str, Any]:
        pixel_values = torch.stack([torch.as_tensor(f["pixel_values"]) for f in features])
        labels = torch.stack([torch.as_tensor(f["labels"], dtype=torch.long) for f in features])
        return {"pixel_values": pixel_values, "labels": labels}

    training_args_kwargs = {
        "output_dir": config.output_dir,
        "per_device_train_batch_size": config.per_device_batch_size,
        "per_device_eval_batch_size": config.per_device_batch_size,
        "num_train_epochs": config.num_train_epochs,
        "learning_rate": config.learning_rate,
        "fp16": config.fp16,
        "evaluation_strategy": "steps",
        "eval_steps": 200,
        "save_steps": 200,
        "save_total_limit": 2,
        "predict_with_generate": True,
        "logging_steps": 50,
        "seed": config.seed,
        "load_best_model_at_end": True,
        "metric_for_best_model": "cer",
        "greater_is_better": False,
        "report_to": [],
    }
    try:
        training_args = Seq2SeqTrainingArguments(**training_args_kwargs)
    except TypeError:
        # Older transformers uses do_eval instead of evaluation_strategy.
        training_args_kwargs.pop("evaluation_strategy", None)
        training_args_kwargs["do_eval"] = True
        training_args_kwargs["load_best_model_at_end"] = False
        training_args_kwargs.pop("metric_for_best_model", None)
        training_args_kwargs.pop("greater_is_better", None)
        training_args = Seq2SeqTrainingArguments(**training_args_kwargs)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=_collate,
        tokenizer=processor.tokenizer,
        compute_metrics=build_compute_metrics(processor),
    )

    trainer.train()
    trainer.save_model(config.output_dir)
    processor.save_pretrained(config.output_dir)


if __name__ == "__main__":
    main()
