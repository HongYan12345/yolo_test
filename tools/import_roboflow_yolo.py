#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Import a downloaded Roboflow/YOLO dataset into this project's real dataset folder.

This script does NOT download from Roboflow. Download the dataset manually from
Roboflow Universe first, preferably in YOLOv8/YOLOv5 format, then point this
script at the extracted dataset root.

Expected source layouts include:

  source/
    data.yaml
    train/images/*.jpg
    train/labels/*.txt
    valid/images/*.jpg
    valid/labels/*.txt
    test/images/*.jpg
    test/labels/*.txt

or:

  source/
    data.yaml
    train/images/*.jpg
    val/images/*.jpg
    test/images/*.jpg

Output layout:

  datasets/real/
    images/
    labels/

Current project class mapping follows the downloaded Roboflow data.yaml:

  0: Item
  1: Mob
  2: Platform
  3: Player
  4: Portal
  5: Rope

The source dataset may contain different class names and class ids, so this
script remaps labels by class name. Unknown/non-target classes are skipped.
"""

from __future__ import annotations

import argparse
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import yaml
from PIL import Image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

TARGET_CLASS_IDS: Mapping[str, int] = {
    "item": 0,
    "mob": 1,
    "platform": 2,
    "player": 3,
    "portal": 4,
    "rope": 5,
}

# Conservative aliases. Use --map if the Roboflow dataset uses different names.
DEFAULT_ALIASES: Mapping[str, str] = {
    "item": "item",
    "items": "item",
    "drop": "item",
    "drops": "item",
    "loot": "item",
    "loots": "item",
    "mob": "mob",
    "mobs": "mob",
    "monster": "mob",
    "monsters": "mob",
    "enemy": "mob",
    "enemies": "mob",
    "platform": "platform",
    "platforms": "platform",
    "foothold": "platform",
    "footholds": "platform",
    "ground": "platform",
    "floor": "platform",
    "player": "player",
    "character": "player",
    "portal": "portal",
    "portals": "portal",
    "rope": "rope",
    "ropes": "rope",
    "ladder": "rope",
    "ladders": "rope",
}


@dataclass(frozen=True)
class ImportStats:
    images_seen: int = 0
    images_copied: int = 0
    labels_written: int = 0
    boxes_seen: int = 0
    boxes_kept: int = 0
    boxes_skipped_unknown_class: int = 0
    labels_missing: int = 0


def normalize_name(name: object) -> str:
    text = str(name).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def load_source_names(data_yaml_path: Path) -> Dict[int, str]:
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"Missing source data.yaml: {data_yaml_path}")

    with data_yaml_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    names = data.get("names")
    if names is None:
        raise ValueError(f"data.yaml does not contain names: {data_yaml_path}")

    if isinstance(names, list):
        return {index: str(name) for index, name in enumerate(names)}

    if isinstance(names, dict):
        result: Dict[int, str] = {}
        for key, value in names.items():
            result[int(key)] = str(value)
        return result

    raise ValueError(f"Unsupported names format in data.yaml: {type(names)!r}")


def parse_manual_maps(values: Sequence[str]) -> Dict[str, str]:
    result: Dict[str, str] = {}

    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --map value: {value!r}. Expected SOURCE=TARGET.")

        source, target = value.split("=", 1)
        source_key = normalize_name(source)
        target_key = normalize_name(target)

        if target_key not in TARGET_CLASS_IDS:
            valid_targets = ", ".join(TARGET_CLASS_IDS)
            raise ValueError(
                f"Invalid target class in --map {value!r}. "
                f"Valid targets: {valid_targets}."
            )

        result[source_key] = target_key

    return result


def build_class_id_map(
    source_names: Mapping[int, str],
    manual_maps: Mapping[str, str],
) -> Tuple[Dict[int, int], Dict[int, str]]:
    class_id_map: Dict[int, int] = {}
    skipped: Dict[int, str] = {}

    aliases = dict(DEFAULT_ALIASES)
    aliases.update(manual_maps)

    for source_id, source_name in sorted(source_names.items()):
        normalized = normalize_name(source_name)
        target_name = aliases.get(normalized)

        if target_name is None:
            skipped[source_id] = source_name
            continue

        class_id_map[source_id] = TARGET_CLASS_IDS[target_name]

    return class_id_map, skipped


def iter_split_dirs(source_dir: Path) -> Iterable[Tuple[str, Path]]:
    candidates = [
        ("train", source_dir / "train"),
        ("val", source_dir / "val"),
        ("valid", source_dir / "valid"),
        ("test", source_dir / "test"),
    ]

    for split_name, split_dir in candidates:
        image_dir = split_dir / "images"
        if image_dir.exists():
            normalized_split = "val" if split_name == "valid" else split_name
            yield normalized_split, split_dir


def list_images(image_dir: Path) -> List[Path]:
    return sorted(
        path
        for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def remap_label_lines(
    label_path: Optional[Path],
    class_id_map: Mapping[int, int],
) -> Tuple[List[str], int, int, int]:
    if label_path is None or not label_path.exists():
        return [], 0, 0, 1

    output_lines: List[str] = []
    boxes_seen = 0
    boxes_kept = 0
    boxes_skipped_unknown_class = 0

    for raw_line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 5:
            continue

        try:
            source_class_id = int(float(parts[0]))
        except ValueError:
            continue

        boxes_seen += 1

        target_class_id = class_id_map.get(source_class_id)
        if target_class_id is None:
            boxes_skipped_unknown_class += 1
            continue

        # YOLO labels are: class x_center y_center width height.
        # Ignore optional trailing columns such as segmentation/keypoints here.
        output_lines.append(
            " ".join([str(target_class_id), parts[1], parts[2], parts[3], parts[4]])
        )
        boxes_kept += 1

    return output_lines, boxes_seen, boxes_skipped_unknown_class, 0


def parse_aspect_ratio(value: str) -> Optional[Tuple[int, int]]:
    text = value.strip().lower()
    if text in {"", "none", "off", "false", "0"}:
        return None

    if ":" not in text:
        raise ValueError(f"Invalid aspect ratio: {value!r}. Expected format like 16:9 or none.")

    width_text, height_text = text.split(":", 1)
    width = int(width_text)
    height = int(height_text)

    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid aspect ratio: {value!r}. Width and height must be positive.")

    return width, height


def copy_or_restore_image_aspect(
    source_image: Path,
    destination_image: Path,
    restore_aspect: Optional[Tuple[int, int]],
) -> None:
    if restore_aspect is None:
        shutil.copy2(source_image, destination_image)
        return

    image = Image.open(source_image)
    image = image.convert("RGB")

    source_width, source_height = image.size
    aspect_width, aspect_height = restore_aspect

    target_width = source_width
    target_height = max(1, round(target_width * aspect_height / aspect_width))

    image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
    image.save(destination_image)


def import_dataset(
    source_dir: Path,
    output_real_dir: Path,
    class_id_map: Mapping[int, int],
    *,
    prefix: str,
    clean: bool,
    drop_empty: bool,
    restore_aspect: Optional[Tuple[int, int]],
) -> ImportStats:
    output_image_dir = output_real_dir / "images"
    output_label_dir = output_real_dir / "labels"

    if clean and output_real_dir.exists():
        shutil.rmtree(output_real_dir)

    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)

    stats = ImportStats()

    split_dirs = list(iter_split_dirs(source_dir))
    if not split_dirs:
        raise FileNotFoundError(
            f"No Roboflow split image directories found under {source_dir}. "
            "Expected train/images, valid/images or test/images."
        )

    images_seen = 0
    images_copied = 0
    labels_written = 0
    boxes_seen_total = 0
    boxes_kept_total = 0
    boxes_skipped_total = 0
    labels_missing_total = 0

    for split, split_dir in split_dirs:
        image_dir = split_dir / "images"
        label_dir = split_dir / "labels"

        for image_path in list_images(image_dir):
            images_seen += 1

            relative_image = image_path.relative_to(image_dir)
            relative_label = relative_image.with_suffix(".txt")
            label_path = label_dir / relative_label

            output_lines, boxes_seen, boxes_skipped, labels_missing = remap_label_lines(
                label_path if label_path.exists() else None,
                class_id_map,
            )

            boxes_seen_total += boxes_seen
            boxes_kept_total += len(output_lines)
            boxes_skipped_total += boxes_skipped
            labels_missing_total += labels_missing

            if drop_empty and not output_lines:
                continue

            safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", image_path.stem)
            dst_stem = f"{prefix}_{split}_{images_copied:07d}_{safe_stem}"

            dst_image = output_image_dir / f"{dst_stem}{image_path.suffix.lower()}"
            dst_label = output_label_dir / f"{dst_stem}.txt"

            copy_or_restore_image_aspect(image_path, dst_image, restore_aspect)
            dst_label.write_text("\n".join(output_lines) + ("\n" if output_lines else ""), encoding="utf-8")

            images_copied += 1
            labels_written += 1

    return ImportStats(
        images_seen=images_seen,
        images_copied=images_copied,
        labels_written=labels_written,
        boxes_seen=boxes_seen_total,
        boxes_kept=boxes_kept_total,
        boxes_skipped_unknown_class=boxes_skipped_total,
        labels_missing=labels_missing_total,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import a downloaded Roboflow YOLO dataset into datasets/real with class remapping."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Extracted Roboflow dataset root containing data.yaml and train/valid/test folders.",
    )
    parser.add_argument(
        "--output-real-dir",
        type=Path,
        default=Path("datasets/real"),
        help="Destination real dataset directory. Default: datasets/real.",
    )
    parser.add_argument(
        "--prefix",
        default="roboflow",
        help="Filename prefix for imported samples.",
    )
    parser.add_argument(
        "--map",
        action="append",
        default=[],
        metavar="SOURCE=TARGET",
        help=(
            "Manual class-name mapping. TARGET must be one of item/mob/platform/player/portal/rope. "
            "Example: --map Monster=mob --map Ladder=rope"
        ),
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove output-real-dir before importing. Be careful if you already have local labels.",
    )
    parser.add_argument(
        "--drop-empty",
        action="store_true",
        help="Drop images whose labels contain no target classes after remapping.",
    )
    parser.add_argument(
        "--restore-aspect",
        default="none",
        metavar="RATIO",
        help=(
            "Optionally undo Roboflow Stretch-to-square exports by resizing imported images "
            "back to an expected aspect ratio such as 16:9. YOLO normalized labels remain valid "
            "for whole-image anisotropic resizing. Default: none."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    source_dir = args.source_dir.resolve()
    data_yaml_path = source_dir / "data.yaml"

    source_names = load_source_names(data_yaml_path)
    manual_maps = parse_manual_maps(args.map)
    class_id_map, skipped_classes = build_class_id_map(source_names, manual_maps)
    restore_aspect = parse_aspect_ratio(args.restore_aspect)

    print("[INFO] Source classes:")
    for source_id, source_name in sorted(source_names.items()):
        target_id = class_id_map.get(source_id)
        if target_id is None:
            print(f"  {source_id}: {source_name} -> SKIP")
        else:
            target_name = next(name for name, cid in TARGET_CLASS_IDS.items() if cid == target_id)
            print(f"  {source_id}: {source_name} -> {target_id}:{target_name}")

    if not class_id_map:
        raise SystemExit(
            "No source classes map to this project's target classes. "
            "Use --map SOURCE=TARGET to define mappings manually."
        )

    if skipped_classes:
        print("[WARN] Some source classes are not target classes and will be skipped:")
        for source_id, source_name in sorted(skipped_classes.items()):
            print(f"  {source_id}: {source_name}")

    if restore_aspect is not None:
        print(f"[INFO] Restoring imported image aspect ratio to {restore_aspect[0]}:{restore_aspect[1]}")

    stats = import_dataset(
        source_dir=source_dir,
        output_real_dir=args.output_real_dir,
        class_id_map=class_id_map,
        prefix=args.prefix,
        clean=args.clean,
        drop_empty=args.drop_empty,
        restore_aspect=restore_aspect,
    )

    print("[DONE] Roboflow YOLO dataset imported")
    print(f"source: {source_dir}")
    print(f"output: {args.output_real_dir}")
    print(f"images seen: {stats.images_seen}")
    print(f"images copied: {stats.images_copied}")
    print(f"labels written: {stats.labels_written}")
    print(f"boxes seen: {stats.boxes_seen}")
    print(f"boxes kept: {stats.boxes_kept}")
    print(f"boxes skipped because class is not mapped: {stats.boxes_skipped_unknown_class}")
    print(f"missing label files treated as empty labels: {stats.labels_missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
