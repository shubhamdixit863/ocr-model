#!/usr/bin/env python3
"""Upload a trained model directory to S3 using env credentials."""
import argparse
import os
from pathlib import Path
from typing import Iterable

import boto3


def iter_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file():
            yield path


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload model artifacts to S3.")
    parser.add_argument("--model-dir", default="models/kaithi-trocr", help="Local model directory.")
    parser.add_argument("--bucket", required=True, help="Target S3 bucket.")
    parser.add_argument("--prefix", default="models/kaithi-trocr", help="S3 key prefix.")
    args = parser.parse_args()

    model_dir = Path(args.model_dir).resolve()
    if not model_dir.exists():
        raise SystemExit(f"Model dir not found: {model_dir}")

    access_key = os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    if not access_key or not secret_key:
        raise SystemExit("Missing AWS_ACCESS_KEY_ID or AWS_SECRET_ACCESS_KEY in env.")

    session = boto3.Session(
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=os.getenv("AWS_DEFAULT_REGION"),
    )
    s3 = session.client("s3")

    for file_path in iter_files(model_dir):
        rel = file_path.relative_to(model_dir).as_posix()
        key = f"{args.prefix.rstrip('/')}/{rel}"
        s3.upload_file(str(file_path), args.bucket, key)
        print(f"Uploaded s3://{args.bucket}/{key}")


if __name__ == "__main__":
    main()
