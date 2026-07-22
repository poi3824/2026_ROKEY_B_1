"""Replicator ZIP을 가벼운 Colab용 YOLO dataset ZIP으로 만든다."""

import argparse
import json
import random
import tempfile
import zipfile
from pathlib import Path

import cv2

from prepare_dataset import build_labels, decode_png, find_dataset_root, find_frame_ids


HERE = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Colab용 경량 bolt dataset 생성")
    parser.add_argument("--source", type=Path, default=HERE / "pose_dataset.zip")
    parser.add_argument("--output", type=Path, default=HERE / "bolt_seg_colab.zip")
    parser.add_argument("--quality", type=int, default=85)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-area", type=float, default=20.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.source.is_file():
        raise FileNotFoundError(args.source)
    if not 1 <= args.quality <= 100:
        raise ValueError("--quality는 1~100이어야 합니다.")

    with tempfile.TemporaryDirectory(prefix="bolt_seg_colab_") as temp:
        dataset = Path(temp) / "bolt_seg"
        for split in ("train", "val"):
            (dataset / "images" / split).mkdir(parents=True)
            (dataset / "labels" / split).mkdir(parents=True)

        with zipfile.ZipFile(args.source) as source:
            names = source.namelist()
            root = find_dataset_root(names)
            frame_ids = find_frame_ids(names, root)
            random.Random(args.seed).shuffle(frame_ids)
            val_count = max(1, round(len(frame_ids) * args.val_ratio))
            val_ids = set(frame_ids[:val_count])
            counts = {"train": 0, "val": 0, "objects": 0, "empty": 0}

            for frame_id in frame_ids:
                split = "val" if frame_id in val_ids else "train"
                rgb = decode_png(source, f"{root}rgb_{frame_id}.png", cv2.IMREAD_COLOR)
                mask = decode_png(
                    source, f"{root}instance_segmentation_{frame_id}.png", cv2.IMREAD_UNCHANGED
                )
                semantics = json.loads(
                    source.read(
                        f"{root}instance_segmentation_semantics_mapping_{frame_id}.json"
                    ).decode("utf-8")
                )
                labels = build_labels(mask, semantics, {"bolt": 0}, args.min_area)
                stem = f"rgb_{frame_id}"
                image_path = dataset / "images" / split / f"{stem}.jpg"
                label_path = dataset / "labels" / split / f"{stem}.txt"
                if not cv2.imwrite(
                    str(image_path), rgb, [cv2.IMWRITE_JPEG_QUALITY, args.quality]
                ):
                    raise OSError(f"이미지 저장 실패: {image_path}")
                label_path.write_text(
                    "\n".join(labels) + ("\n" if labels else ""), encoding="utf-8"
                )
                counts[split] += 1
                counts["objects"] += len(labels)
                counts["empty"] += int(not labels)

        (dataset / "data.yaml").write_text(
            "path: /content/bolt_seg\n"
            "train: images/train\n"
            "val: images/val\n"
            "names:\n"
            "  0: bolt\n",
            encoding="utf-8",
        )

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(
            args.output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1
        ) as target:
            for path in dataset.rglob("*"):
                if path.is_file():
                    target.write(path, path.relative_to(dataset.parent))

    size_mb = args.output.stat().st_size / (1024 * 1024)
    print(
        f"created={args.output} size={size_mb:.1f}MB "
        f"train={counts['train']} val={counts['val']} "
        f"objects={counts['objects']} empty={counts['empty']}"
    )


if __name__ == "__main__":
    main()
