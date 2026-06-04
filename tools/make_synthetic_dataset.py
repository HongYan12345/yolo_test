#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate synthetic YOLO detection samples for MapleStory detection.

This script composites crawled transparent PNG assets onto real/empty map
background screenshots and writes YOLO-format labels.

Classes:
  0: rope
  1: platform
  2: monster
  3: item

This synthetic generator only creates labels for monster/item automatically.
Rope/platform labels should come from manually labeled real screenshots.

Recommended usage:

  python tools/make_synthetic_dataset.py ^
    --background-dir datasets/backgrounds ^
    --monster-dir assets/monsters/raw ^
    --item-dir assets/items/raw ^
    --output-dir datasets/synthetic ^
    --count 3000 ^
    --seed 42

Output:
  datasets/synthetic/images/*.jpg
  datasets/synthetic/labels/*.txt

Notes:
  - Synthetic images should mainly be used for train.
  - Validation/test should be real screenshots whenever possible.
  - Do not train only on transparent PNG originals.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageEnhance, ImageFilter, ImageOps


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


@dataclass(frozen=True)
class Box:
    class_id: int
    x1: int
    y1: int
    x2: int
    y2: int

    def to_yolo(self, width: int, height: int) -> str:
        x1 = max(0, min(self.x1, width - 1))
        y1 = max(0, min(self.y1, height - 1))
        x2 = max(0, min(self.x2, width))
        y2 = max(0, min(self.y2, height))
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        cx = x1 + bw / 2
        cy = y1 + bh / 2
        return (
            f"{self.class_id} "
            f"{cx / width:.6f} {cy / height:.6f} "
            f"{bw / width:.6f} {bh / height:.6f}"
        )


def list_images(directory: Path) -> List[Path]:
    if not directory.exists():
        return []
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_rgba(path: Path) -> Optional[Image.Image]:
    try:
        image = Image.open(path)
        return image.convert("RGBA")
    except Exception as exc:
        print(f"[WARN] failed to load image: {path} error={exc}")
        return None


def alpha_bbox(image: Image.Image, min_alpha: int = 10) -> Optional[Tuple[int, int, int, int]]:
    """Return the bounding box of visible pixels in an RGBA image."""
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    alpha = image.getchannel("A")
    mask = alpha.point(lambda value: 255 if value >= min_alpha else 0)
    return mask.getbbox()


def crop_visible(image: Image.Image) -> Optional[Image.Image]:
    bbox = alpha_bbox(image)
    if not bbox:
        return None
    return image.crop(bbox)


def resize_by_height(image: Image.Image, target_height: int) -> Image.Image:
    w, h = image.size
    target_height = max(1, int(target_height))
    target_width = max(1, int(round(w * target_height / max(1, h))))
    return image.resize((target_width, target_height), Image.Resampling.LANCZOS)


def random_asset_transform(
    image: Image.Image,
    target_height: int,
    opacity_range: Tuple[float, float],
    brightness_range: Tuple[float, float],
    contrast_range: Tuple[float, float],
    blur_probability: float,
    flip_probability: float,
) -> Image.Image:
    image = crop_visible(image) or image
    image = resize_by_height(image, target_height)

    if random.random() < flip_probability:
        image = ImageOps.mirror(image)

    brightness = random.uniform(*brightness_range)
    contrast = random.uniform(*contrast_range)
    image = ImageEnhance.Brightness(image).enhance(brightness)
    image = ImageEnhance.Contrast(image).enhance(contrast)

    opacity = random.uniform(*opacity_range)
    if opacity < 0.999:
        r, g, b, a = image.split()
        a = a.point(lambda value: int(value * opacity))
        image = Image.merge("RGBA", (r, g, b, a))

    if random.random() < blur_probability:
        image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.8)))

    return image


def paste_rgba(
    background: Image.Image,
    sprite: Image.Image,
    x: int,
    y: int,
    class_id: int,
    min_visible_ratio: float,
) -> Optional[Box]:
    bg_w, bg_h = background.size
    sp_w, sp_h = sprite.size

    # Allow slight edge clipping.
    paste_x = x
    paste_y = y

    visible_x1 = max(0, paste_x)
    visible_y1 = max(0, paste_y)
    visible_x2 = min(bg_w, paste_x + sp_w)
    visible_y2 = min(bg_h, paste_y + sp_h)

    if visible_x2 <= visible_x1 or visible_y2 <= visible_y1:
        return None

    visible_area = (visible_x2 - visible_x1) * (visible_y2 - visible_y1)
    total_area = sp_w * sp_h
    if total_area <= 0 or visible_area / total_area < min_visible_ratio:
        return None

    crop_left = visible_x1 - paste_x
    crop_top = visible_y1 - paste_y
    crop_right = crop_left + (visible_x2 - visible_x1)
    crop_bottom = crop_top + (visible_y2 - visible_y1)
    cropped = sprite.crop((crop_left, crop_top, crop_right, crop_bottom))

    background.alpha_composite(cropped, (visible_x1, visible_y1))

    bbox = alpha_bbox(cropped)
    if not bbox:
        return None

    bx1, by1, bx2, by2 = bbox
    return Box(
        class_id=class_id,
        x1=visible_x1 + bx1,
        y1=visible_y1 + by1,
        x2=visible_x1 + bx2,
        y2=visible_y1 + by2,
    )


def random_ground_position(
    bg_size: Tuple[int, int],
    sprite_size: Tuple[int, int],
    y_min_ratio: float,
    y_max_ratio: float,
    edge_clip_ratio: float,
) -> Tuple[int, int]:
    bg_w, bg_h = bg_size
    sp_w, sp_h = sprite_size

    clip_x = int(sp_w * edge_clip_ratio)
    x = random.randint(-clip_x, max(-clip_x, bg_w - sp_w + clip_x))

    y_min = int(bg_h * y_min_ratio)
    y_max = int(bg_h * y_max_ratio)
    y_max = max(y_min, min(bg_h - 1, y_max))

    # Interpret selected y as approximate feet/bottom position.
    foot_y = random.randint(y_min, y_max)
    y = foot_y - sp_h
    return x, y


def maybe_add_occluder(
    image: Image.Image,
    box: Box,
    probability: float,
) -> None:
    """Add simple semi-transparent rectangles/number-like bars to simulate effects."""
    if random.random() >= probability:
        return

    w = max(1, box.x2 - box.x1)
    h = max(1, box.y2 - box.y1)

    occ_w = random.randint(max(2, int(w * 0.2)), max(3, int(w * 0.7)))
    occ_h = random.randint(max(2, int(h * 0.08)), max(3, int(h * 0.25)))
    x1 = random.randint(box.x1, max(box.x1, box.x2 - occ_w))
    y1 = random.randint(box.y1, max(box.y1, box.y2 - occ_h))

    color = random.choice(
        [
            (255, 50, 50, 120),
            (255, 220, 50, 120),
            (80, 160, 255, 100),
            (255, 255, 255, 90),
        ]
    )
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    patch = Image.new("RGBA", (occ_w, occ_h), color)
    overlay.alpha_composite(patch, (x1, y1))
    image.alpha_composite(overlay)


def generate_one(
    background_path: Path,
    monster_paths: Sequence[Path],
    item_paths: Sequence[Path],
    args: argparse.Namespace,
) -> Tuple[Image.Image, List[Box]]:
    background = load_rgba(background_path)
    if background is None:
        raise RuntimeError(f"Cannot load background: {background_path}")

    if args.output_width and args.output_height:
        background = ImageOps.fit(
            background,
            (args.output_width, args.output_height),
            method=Image.Resampling.LANCZOS,
        )

    bg_w, bg_h = background.size
    boxes: List[Box] = []

    monster_count = random.randint(args.min_monsters, args.max_monsters)
    item_count = random.randint(args.min_items, args.max_items)

    for _ in range(monster_count):
        if not monster_paths:
            break
        asset = load_rgba(random.choice(monster_paths))
        if asset is None:
            continue

        target_height = random.randint(args.monster_min_height, args.monster_max_height)
        sprite = random_asset_transform(
            asset,
            target_height=target_height,
            opacity_range=(args.monster_min_opacity, args.monster_max_opacity),
            brightness_range=(args.min_brightness, args.max_brightness),
            contrast_range=(args.min_contrast, args.max_contrast),
            blur_probability=args.blur_probability,
            flip_probability=args.flip_probability,
        )

        x, y = random_ground_position(
            (bg_w, bg_h),
            sprite.size,
            args.y_min_ratio,
            args.y_max_ratio,
            args.edge_clip_ratio,
        )
        box = paste_rgba(
            background,
            sprite,
            x,
            y,
            class_id=2,
            min_visible_ratio=args.min_visible_ratio,
        )
        if box:
            boxes.append(box)
            maybe_add_occluder(background, box, args.occlusion_probability)

    for _ in range(item_count):
        if not item_paths:
            break
        asset = load_rgba(random.choice(item_paths))
        if asset is None:
            continue

        target_height = random.randint(args.item_min_height, args.item_max_height)
        sprite = random_asset_transform(
            asset,
            target_height=target_height,
            opacity_range=(args.item_min_opacity, args.item_max_opacity),
            brightness_range=(args.min_brightness, args.max_brightness),
            contrast_range=(args.min_contrast, args.max_contrast),
            blur_probability=args.blur_probability,
            flip_probability=0.0,
        )

        x, y = random_ground_position(
            (bg_w, bg_h),
            sprite.size,
            args.y_min_ratio,
            args.y_max_ratio,
            args.edge_clip_ratio,
        )
        box = paste_rgba(
            background,
            sprite,
            x,
            y,
            class_id=3,
            min_visible_ratio=args.min_visible_ratio,
        )
        if box:
            boxes.append(box)
            maybe_add_occluder(background, box, args.item_occlusion_probability)

    # Mild final image perturbation.
    if random.random() < args.final_blur_probability:
        background = background.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.6)))

    rgb = background.convert("RGB")
    return rgb, boxes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic MapleStory YOLO dataset.")
    parser.add_argument("--background-dir", type=Path, default=Path("datasets/backgrounds"))
    parser.add_argument("--monster-dir", type=Path, default=Path("assets/monsters/raw"))
    parser.add_argument("--item-dir", type=Path, default=Path("assets/items/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/synthetic"))
    parser.add_argument("--count", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--output-width", type=int, default=0, help="Optional fixed output width.")
    parser.add_argument("--output-height", type=int, default=0, help="Optional fixed output height.")

    parser.add_argument("--min-monsters", type=int, default=1)
    parser.add_argument("--max-monsters", type=int, default=6)
    parser.add_argument("--min-items", type=int, default=0)
    parser.add_argument("--max-items", type=int, default=8)

    parser.add_argument("--monster-min-height", type=int, default=28)
    parser.add_argument("--monster-max-height", type=int, default=110)
    parser.add_argument("--item-min-height", type=int, default=12)
    parser.add_argument("--item-max-height", type=int, default=34)

    parser.add_argument("--monster-min-opacity", type=float, default=0.82)
    parser.add_argument("--monster-max-opacity", type=float, default=1.0)
    parser.add_argument("--item-min-opacity", type=float, default=0.85)
    parser.add_argument("--item-max-opacity", type=float, default=1.0)

    parser.add_argument("--min-brightness", type=float, default=0.82)
    parser.add_argument("--max-brightness", type=float, default=1.18)
    parser.add_argument("--min-contrast", type=float, default=0.85)
    parser.add_argument("--max-contrast", type=float, default=1.15)

    parser.add_argument("--blur-probability", type=float, default=0.08)
    parser.add_argument("--final-blur-probability", type=float, default=0.04)
    parser.add_argument("--flip-probability", type=float, default=0.5)

    parser.add_argument("--occlusion-probability", type=float, default=0.12)
    parser.add_argument("--item-occlusion-probability", type=float, default=0.04)
    parser.add_argument("--min-visible-ratio", type=float, default=0.55)
    parser.add_argument("--edge-clip-ratio", type=float, default=0.12)

    parser.add_argument(
        "--y-min-ratio",
        type=float,
        default=0.45,
        help="Minimum approximate ground/feet y ratio in background.",
    )
    parser.add_argument(
        "--y-max-ratio",
        type=float,
        default=0.92,
        help="Maximum approximate ground/feet y ratio in background.",
    )

    parser.add_argument("--jpeg-quality", type=int, default=92)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)

    backgrounds = list_images(args.background_dir)
    monsters = list_images(args.monster_dir)
    items = list_images(args.item_dir)

    if not backgrounds:
        raise SystemExit(
            f"No background images found in {args.background_dir}. "
            "Please put empty/low-target map screenshots there first."
        )
    if not monsters:
        raise SystemExit(f"No monster images found in {args.monster_dir}.")
    if not items:
        print(f"[WARN] no item images found in {args.item_dir}; generating monster-only samples.")

    image_dir = args.output_dir / "images"
    label_dir = args.output_dir / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] backgrounds={len(backgrounds)} monsters={len(monsters)} items={len(items)}")
    print(f"[INFO] output={args.output_dir} count={args.count} seed={args.seed}")

    for index in range(args.count):
        background_path = random.choice(backgrounds)
        image, boxes = generate_one(background_path, monsters, items, args)

        stem = f"synthetic_{index:06d}"
        image_path = image_dir / f"{stem}.jpg"
        label_path = label_dir / f"{stem}.txt"

        image.save(image_path, quality=args.jpeg_quality)

        width, height = image.size
        with label_path.open("w", encoding="utf-8") as file:
            for box in boxes:
                file.write(box.to_yolo(width, height) + "\n")

        if (index + 1) % 100 == 0 or index + 1 == args.count:
            print(f"[INFO] generated {index + 1}/{args.count}")

    print("[DONE] synthetic dataset generated")
    print(f"images: {image_dir}")
    print(f"labels: {label_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
