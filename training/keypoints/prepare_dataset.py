#!/usr/bin/env python3
"""Convert datasets/keypoints/<cam>/ (custom Isaac Sim PoseWriter dump) into Ultralytics
YOLO-pose label format under training/keypoints/data/.

Each object's 9 cuboid keypoints (Center + 8 corners, see keypoint_order in the source
JSON) come from 3D ground truth projected to pixels -- correct even when the point is
behind the gripper, since it isn't a pixel-level detection. The visibility flag written
here reflects that: a keypoint only gets v=0 (excluded from the loss) when it falls
outside the image frame entirely. A keypoint that's occluded by the gripper but still
in-frame gets v=1 (occluded) or v=2 (visible) based on the object's overall visibility
fraction -- both are still supervised, so the model learns to regress occluded-point
locations from the visible parts of the object rather than never seeing them at all.
This is the mechanism that lets a keypoint model do what a segmentation mask structurally
cannot: report a position for a point it can't currently see.

Frames with no in-frame target object are still copied to images/ with no matching .txt --
the standard YOLO convention for a background/hard-negative example.

Usage:
    python prepare_dataset.py --occlusion both --cams color
    python prepare_dataset.py --occlusion light --cams color --limit 20   # smoke test
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from training.common.classes import CLASS_NAMES, CLASS_TO_ID
from training.common.split import filter_by_occlusion, train_val_split

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = REPO_ROOT / "datasets" / "keypoints"
DEFAULT_OUT = Path(__file__).resolve().parent / "data"

VISIBLE_THRESHOLD = 0.5  # object-level 'visibility' at/above this -> keypoints marked v=2, else v=1


def frame_indices(dataset_root: Path, cam: str):
    return sorted(
        int(p.stem) for p in (dataset_root / cam).glob("*.json")
        if p.stem.isdigit() and len(p.stem) == 6
    )


def compute_flip_idx(keypoint_order):
    """Left/Right keypoint pairs must swap under a horizontal flip augmentation."""
    name_to_idx = {n: i for i, n in enumerate(keypoint_order)}
    flip_idx = []
    for name in keypoint_order:
        if name.startswith("L"):
            mirror = "R" + name[1:]
        elif name.startswith("R"):
            mirror = "L" + name[1:]
        else:
            mirror = name
        flip_idx.append(name_to_idx.get(mirror, name_to_idx[name]))
    return flip_idx


def convert_frame(dataset_root: Path, cam: str, idx: int, out_images: Path, out_labels: Path, out_name: str, w: int, h: int):
    cam_dir = dataset_root / cam
    json_path = cam_dir / f"{idx:06d}.json"
    png_path = cam_dir / f"{idx:06d}.png"

    data = json.loads(json_path.read_text())
    label_lines = []
    for obj in data["objects"]:
        cls_name = obj["label"]
        if cls_name not in CLASS_TO_ID:
            continue
        pts = obj["cuboid_keypoints_projected"]
        obj_visibility = obj.get("visibility", 1.0)

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x_min, x_max = max(0.0, min(xs)), min(float(w), max(xs))
        y_min, y_max = max(0.0, min(ys)), min(float(h), max(ys))
        if x_max <= x_min or y_max <= y_min:
            continue  # entirely outside the frame

        bw, bh = x_max - x_min, y_max - y_min
        cx, cy = x_min + bw / 2, y_min + bh / 2

        kp_fields = []
        for px, py in pts:
            in_frame = 0 <= px < w and 0 <= py < h
            if not in_frame:
                # v=0 keypoints are masked out of the loss -- zero the coords too, so an
                # out-of-frame projection (which can be arbitrarily negative/large) never
                # leaks a bogus value into anything that doesn't check v first.
                kp_fields.append("0.000000 0.000000 0")
                continue
            v = 2 if obj_visibility >= VISIBLE_THRESHOLD else 1
            kp_fields.append(f"{px / w:.6f} {py / h:.6f} {v}")

        line = (
            f"{CLASS_TO_ID[cls_name]} {cx / w:.6f} {cy / h:.6f} {bw / w:.6f} {bh / h:.6f} "
            + " ".join(kp_fields)
        )
        label_lines.append(line)

    # Always copy the image, even with zero in-frame objects: an image with no matching
    # .txt is a hard negative in the YOLO convention, kept deliberately as a background
    # example rather than dropped.
    shutil.copyfile(png_path, out_images / f"{out_name}.png")
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
    keypoint_order = None

    for cam in args.cams:
        indices = filter_by_occlusion(frame_indices(args.dataset_root, cam), args.occlusion)
        if args.limit:
            indices = indices[: args.limit]
        train_idx, val_idx = train_val_split(indices, args.val_ratio, args.seed)
        for split, split_indices in [("train", train_idx), ("val", val_idx)]:
            for idx in split_indices:
                json_path = args.dataset_root / cam / f"{idx:06d}.json"
                data = json.loads(json_path.read_text())
                if keypoint_order is None:
                    keypoint_order = data["keypoint_order"]
                w, h = data["camera_data"]["resolution"]
                out_name = f"{cam}_{idx:06d}"
                has_labels = convert_frame(
                    args.dataset_root, cam, idx,
                    args.out / "images" / split, args.out / "labels" / split, out_name, w, h,
                )
                if has_labels:
                    n_positive[split] += 1
                else:
                    n_background[split] += 1

    flip_idx = compute_flip_idx(keypoint_order) if keypoint_order else list(range(9))
    data_yaml = args.out / "data.yaml"
    names_block = "".join(f"  {i}: {name}\n" for i, name in enumerate(CLASS_NAMES))
    data_yaml.write_text(
        f"path: {args.out.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"kpt_shape: [{len(keypoint_order) if keypoint_order else 9}, 3]\n"
        f"flip_idx: {flip_idx}\n"
        f"names:\n{names_block}"
    )

    print(f"keypoint_order: {keypoint_order}")
    for split in ["train", "val"]:
        total = n_positive[split] + n_background[split]
        print(f"{split}: {total} frames ({n_positive[split]} labeled, "
              f"{n_background[split]} background/hard-negative)")
    print(f"data.yaml -> {data_yaml}")


if __name__ == "__main__":
    main()
