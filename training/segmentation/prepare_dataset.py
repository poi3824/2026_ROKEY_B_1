#!/usr/bin/env python3
"""Convert datasets/segmentation/<cam>/ (Isaac Sim Replicator BasicWriter dump) into
Ultralytics YOLO segmentation label format under training/segmentation/data/.

For each frame, reads instance_segmentation_<idx>.png + its
instance_segmentation_semantics_mapping_<idx>.json (instance color -> class name),
extracts one polygon per instance via cv2.findContours + approxPolyDP, and writes a
YOLO-seg label line "class_id x1 y1 x2 y2 ..." (normalized 0-1) per instance. Frames with
no in-frame object of interest are still copied to images/ with no matching .txt -- the
standard YOLO convention for a background/hard-negative example.

Usage:
    python prepare_dataset.py --occlusion both --cams color
    python prepare_dataset.py --occlusion light --cams color --limit 20   # smoke test
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from training.common.classes import CLASS_NAMES, CLASS_TO_ID
from training.common.split import filter_by_occlusion, train_val_split

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = REPO_ROOT / "datasets" / "segmentation"
DEFAULT_OUT = Path(__file__).resolve().parent / "data"

IGNORED_CLASSES = {"BACKGROUND", "UNLABELLED"}


def parse_rgba_key(key: str):
    return tuple(int(v) for v in key.strip("()").split(","))


def frame_indices(dataset_root: Path, cam: str):
    return sorted(int(p.stem.split("_")[-1]) for p in (dataset_root / cam).glob("rgb_*.png"))


def mask_to_polygons(mask: np.ndarray, epsilon_frac: float = 0.005, min_area: float = 20.0):
    """mask: uint8 binary (0/255) single instance. Returns list of (N,2) pixel-coord polygons."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon_frac * peri, True)
        if len(approx) < 3:
            continue
        polygons.append(approx.reshape(-1, 2))
    return polygons


def convert_frame(dataset_root: Path, cam: str, idx: int, out_images: Path, out_labels: Path, out_name: str):
    cam_dir = dataset_root / cam
    rgb_path = cam_dir / f"rgb_{idx:04d}.png"
    inst_png_path = cam_dir / f"instance_segmentation_{idx:04d}.png"
    inst_semantics_path = cam_dir / f"instance_segmentation_semantics_mapping_{idx:04d}.json"

    semantics = json.loads(inst_semantics_path.read_text())
    color_to_class = {}
    for key, val in semantics.items():
        cls = val.get("class") if isinstance(val, dict) else val
        if cls in IGNORED_CLASSES or cls not in CLASS_TO_ID:
            continue
        color_to_class[parse_rgba_key(key)] = cls

    raw = cv2.imread(str(inst_png_path), cv2.IMREAD_UNCHANGED)
    inst_img = cv2.cvtColor(raw, cv2.COLOR_BGRA2RGBA)
    h, w = inst_img.shape[:2]

    label_lines = []
    for color, cls_name in color_to_class.items():
        mask = (np.all(inst_img == np.array(color, dtype=np.uint8), axis=-1).astype(np.uint8)) * 255
        if not mask.any():
            continue
        for poly in mask_to_polygons(mask):
            norm = poly.astype(np.float32)
            norm[:, 0] /= w
            norm[:, 1] /= h
            coords = " ".join(f"{v:.6f}" for v in norm.reshape(-1))
            label_lines.append(f"{CLASS_TO_ID[cls_name]} {coords}")

    # Always copy the image, even with zero labeled instances: an image with no matching
    # .txt is a hard negative in the YOLO convention, and frames with no in-frame object
    # are deliberately kept as background examples to suppress false positives.
    shutil.copyfile(rgb_path, out_images / f"{out_name}.png")
    if label_lines:
        (out_labels / f"{out_name}.txt").write_text("\n".join(label_lines) + "\n")
    return bool(label_lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--cams", nargs="+", default=["color"], choices=["color", "depth", "left", "right"])
    ap.add_argument("--occlusion", choices=["light", "heavy", "both"], default="both")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None, help="only convert the first N frames per cam (smoke test)")
    args = ap.parse_args()

    for split in ["train", "val"]:
        (args.out / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.out / "labels" / split).mkdir(parents=True, exist_ok=True)

    n_positive = {"train": 0, "val": 0}
    n_background = {"train": 0, "val": 0}

    for cam in args.cams:
        indices = filter_by_occlusion(frame_indices(args.dataset_root, cam), args.occlusion)
        if args.limit:
            indices = indices[: args.limit]
        train_idx, val_idx = train_val_split(indices, args.val_ratio, args.seed)
        for split, split_indices in [("train", train_idx), ("val", val_idx)]:
            for idx in split_indices:
                out_name = f"{cam}_{idx:04d}"
                has_labels = convert_frame(
                    args.dataset_root, cam, idx,
                    args.out / "images" / split, args.out / "labels" / split, out_name,
                )
                if has_labels:
                    n_positive[split] += 1
                else:
                    n_background[split] += 1

    data_yaml = args.out / "data.yaml"
    names_block = "".join(f"  {i}: {name}\n" for i, name in enumerate(CLASS_NAMES))
    data_yaml.write_text(
        f"path: {args.out.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n{names_block}"
    )

    for split in ["train", "val"]:
        total = n_positive[split] + n_background[split]
        print(f"{split}: {total} frames ({n_positive[split]} labeled, "
              f"{n_background[split]} background/hard-negative)")
    print(f"data.yaml -> {data_yaml}")


if __name__ == "__main__":
    main()
