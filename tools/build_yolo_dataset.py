#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a YOLO dataset for MapleStory monster/item detection.

Input conventions:

  datasets/synthetic/
    images/
      synthetic_000000.jpg
    labels/
      synthetic_000000.txt

  datasets/real/
    images/
      real_000001.png
    labels/
      real_000001.txt

Output:

  datasets/yolo/
    images/
      train/
      val/
      test/
    labels/
      train/
      val/
      test/
    data.yaml

Classes:
  0: rope
  1: platform
  2: monster
  3: item

Recommended usage:

  python tools/build_yolo_dataset.py ^
    --synthetic-dir datasets/synthetic ^
    --real-dir datasets/real ^
    --output-dir datasets/yolo ^
    --clean ^
    --seed 42

Important:
  - Synthetic samples are copied to train only by default.
  - Real samples are split into train/val/test.
  - Validation/test should be real screenshots whenever possible.
"""

from __future__ import annotations

import argparse
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


@dataclass(frozen=True)
class Sample:
    image_path: Path
    label_path: Optional[Path]
    source: str


def list_images(directory: Path) -> List[Path]:
    if not directory.exists():
        return []
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def paired_samples(dataset_dir: Path, source: str) -> List[Sample]:
    image_root = dataset_dir / "images"
    label_root = dataset_dir / "labels"
    images = list_images(image_root)

    samples: List[Sample] = []
    for image_path in images:
        rel = image_path.relative_to(image_root)
        label_path = (label_root / rel).with_suffix(".txt")
        samples.append(
            Sample(
                image_path=image_path,
                label_path=label_path if label_path.exists() else None,
                source=source,
            )
        )
    return samples


def split_real_samples(
    samples: Sequence[Sample],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[Sample], List[Sample], List[Sample]]:
    if train_ratio < 0 or val_ratio < 0 or test_ratio < 0:
        raise ValueError("Split ratios must be non-negative.")

    total_ratio = train_ratio + val_ratio + test_ratio
    if total_ratio <= 0:
        raise ValueError("At least one split ratio must be positive.")

    normalized_train = train_ratio / total_ratio
    normalized_val = val_ratio / total_ratio

    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)

    total = len(shuffled)
    train_end = int(total * normalized_train)
    val_end = train_end + int(total * normalized_val)

    train = shuffled[:train_end]
    val = shuffled[train_end:val_end]
    test = shuffled[val_end:]
    return train, val, test


def ensure_dirs(output_dir: Path) -> None:
    for split in ("train", "val", "test"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def copy_sample(
    sample: Sample,
    output_dir: Path,
    split: str,
    index: int,
    missing_label_policy: str,
) -> Tuple[bool, str]:
    suffix = sample.image_path.suffix.lower()
    safe_stem = f"{sample.source}_{index:07d}_{sample.image_path.stem}"
    dst_image = output_dir / "images" / split / f"{safe_stem}{suffix}"
    dst_label = output_dir / "labels" / split / f"{safe_stem}.txt"

    if sample.label_path is None:
        if missing_label_policy == "skip":
            return False, "missing_label_skipped"
        if missing_label_policy == "error":
            raise FileNotFoundError(f"Missing label for image: {sample.image_path}")
        if missing_label_policy != "empty":
            raise ValueError(f"Unknown missing label policy: {missing_label_policy}")

    shutil.copy2(sample.image_path, dst_image)

    if sample.label_path is not None:
        shutil.copy2(sample.label_path, dst_label)
    else:
        dst_label.write_text("", encoding="utf-8")

    return True, "copied"


def write_data_yaml(output_dir: Path) -> None:
    data_yaml = """path: .
train: images/train
val: images/val
test: images/test

nc: 4
names:
  0: rope
  1: platform
  2: monster
  3: item
"""
    (output_dir / "data.yaml").write_text(data_yaml, encoding="utf-8")


def count_files(directory: Path) -> int:
    if not directory.exists():
        return 0
    return sum(1 for path in directory.iterdir() if path.is_file())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build YOLO dataset from synthetic and real samples.")
    parser.add_argument("--synthetic-dir", type=Path, default=Path("datasets/synthetic"))
    parser.add_argument("--real-dir", type=Path, default=Path("datasets/real"))
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/yolo"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean", action="store_true", help="Remove output directory before building.")

    parser.add_argument("--real-train-ratio", type=float, default=0.70)
    parser.add_argument("--real-val-ratio", type=float, default=0.20)
    parser.add_argument("--real-test-ratio", type=float, default=0.10)

    parser.add_argument(
        "--synthetic-train-limit",
        type=int,
        default=0,
        help="Maximum synthetic samples copied to train. 0 means all.",
    )
    parser.add_argument(
        "--no-synthetic",
        action="store_true",
        help="Do not include synthetic samples.",
    )
    parser.add_argument(
        "--missing-label-policy",
        choices=("empty", "skip", "error"),
        default="error",
        help=(
            "How to handle images without labels. "
            "Default is error to avoid accidentally treating unlabeled positive "
            "screenshots as negative samples. Use an empty .txt file for real "
            "negative screenshots."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    synthetic_samples = [] if args.no_synthetic else paired_samples(args.synthetic_dir, "synthetic")
    real_samples = paired_samples(args.real_dir, "real")

    if not synthetic_samples and not real_samples:
        raise SystemExit(
            "No samples found. Expected images under datasets/synthetic/images "
            "or datasets/real/images."
        )

    if args.synthetic_train_limit > 0:
        rng = random.Random(args.seed)
        synthetic_samples = list(synthetic_samples)
        rng.shuffle(synthetic_samples)
        synthetic_samples = synthetic_samples[: args.synthetic_train_limit]

    real_train, real_val, real_test = split_real_samples(
        real_samples,
        train_ratio=args.real_train_ratio,
        val_ratio=args.real_val_ratio,
        test_ratio=args.real_test_ratio,
        seed=args.seed,
    )

    split_map: Dict[str, List[Sample]] = {
        "train": list(synthetic_samples) + list(real_train),
        "val": list(real_val),
        "test": list(real_test),
    }

    if args.clean and args.output_dir.exists():
        shutil.rmtree(args.output_dir)

    ensure_dirs(args.output_dir)

    copied = 0
    skipped = 0
    for split, samples in split_map.items():
        for index, sample in enumerate(samples):
            did_copy, status = copy_sample(
                sample,
                args.output_dir,
                split,
                index,
                missing_label_policy=args.missing_label_policy,
            )
            if did_copy:
                copied += 1
            else:
                skipped += 1

    write_data_yaml(args.output_dir)

    print("[DONE] YOLO dataset built")
    print(f"output: {args.output_dir}")
    print(f"synthetic input samples: {len(synthetic_samples)}")
    print(f"real input samples: {len(real_samples)}")
    print(f"copied: {copied}")
    print(f"skipped: {skipped}")
    for split in ("train", "val", "test"):
        image_count = count_files(args.output_dir / "images" / split)
        label_count = count_files(args.output_dir / "labels" / split)
        print(f"{split}: images={image_count} labels={label_count}")
    print(f"data.yaml: {args.output_dir / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
