"""Convert paired endoscopy PNG images/masks into TransUNet npz format.

Example:
python tools/prepare_endoscopy_png_dataset.py \
  --image_dir /data/esophagus/images \
  --mask_dir /data/esophagus/masks \
  --output_root /data/esophagus_transunet \
  --train_ratio 0.8
"""

import argparse
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True, help="Directory of RGB/gray PNG images")
    parser.add_argument("--mask_dir", type=str, required=True, help="Directory of PNG masks with same filenames")
    parser.add_argument("--output_root", type=str, required=True, help="Output root containing train_npz and lists")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Train split ratio")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--binarize_mask",
        action="store_true",
        help="Convert any non-zero mask pixel to class 1 (binary segmentation)",
    )
    return parser.parse_args()


def load_image(path):
    arr = np.array(Image.open(path))
    # Keep RGB if present so the training pipeline can consume 3-channel inputs.
    if arr.ndim == 3 and arr.shape[2] > 3:
        arr = arr[..., :3]
    return arr.astype(np.float32)


def load_mask(path, binarize):
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    if binarize:
        arr = (arr > 0).astype(np.uint8)
    return arr.astype(np.uint8)


def main():
    args = parse_args()
    random.seed(args.seed)

    image_dir = Path(args.image_dir)
    mask_dir = Path(args.mask_dir)
    output_root = Path(args.output_root)
    train_npz_dir = output_root / "train_npz"
    list_dir = output_root / "lists"
    train_npz_dir.mkdir(parents=True, exist_ok=True)
    list_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(image_dir.glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No png files in {image_dir}")

    valid_ids = []
    for img_path in image_paths:
        stem = img_path.stem
        mask_path = mask_dir / f"{stem}.png"
        if not mask_path.exists():
            continue

        image = load_image(img_path)
        label = load_mask(mask_path, args.binarize_mask)
        if image.ndim == 2:
            if image.shape != label.shape:
                raise ValueError(f"Shape mismatch for {stem}: image {image.shape}, mask {label.shape}")
        elif image.ndim == 3:
            if image.shape[:2] != label.shape:
                raise ValueError(f"Shape mismatch for {stem}: image {image.shape}, mask {label.shape}")
        else:
            raise ValueError(f"Unsupported image ndim for {stem}: {image.ndim}")

        np.savez_compressed(train_npz_dir / f"{stem}.npz", image=image, label=label)
        valid_ids.append(stem)

    if not valid_ids:
        raise RuntimeError("No valid image-mask pairs found.")

    random.shuffle(valid_ids)
    train_count = max(1, int(len(valid_ids) * args.train_ratio))
    train_ids = valid_ids[:train_count]
    val_ids = valid_ids[train_count:]

    (list_dir / "train.txt").write_text("\n".join(train_ids) + "\n")
    # Keep a val split for your own bookkeeping (not consumed by current train.py)
    if val_ids:
        (list_dir / "val.txt").write_text("\n".join(val_ids) + "\n")

    print(f"Prepared {len(valid_ids)} samples")
    print(f"train_npz: {train_npz_dir}")
    print(f"list_dir:   {list_dir}")
    print("Use --dataset Custom and point --root_path/--list_dir to these outputs.")


if __name__ == "__main__":
    main()
