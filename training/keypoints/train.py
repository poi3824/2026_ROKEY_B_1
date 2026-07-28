#!/usr/bin/env python3
"""Train a YOLOv8-pose model on the output of prepare_dataset.py.

Reuses ultralytics (same library already used for the segmentation track) with task="pose"
-- no new dependency. The 9 cuboid keypoints per object (see datasets/README.md) give the
model both a grasp point (the "Center" keypoint) and an orientation (e.g. the vector
between two corner keypoints), which the current segmentation-centroid pipeline has neither
of.

Usage:
    python prepare_dataset.py --occlusion both --cams color   # once, to build training/keypoints/data/
    python train.py --epochs 150 --model yolov8n-pose.pt
    python train.py --epochs 1                                 # smoke test
"""
import argparse
import sys
from pathlib import Path

from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from training.common.summarize import write_summary

DEFAULT_DATA = Path(__file__).resolve().parent / "data" / "data.yaml"
DEFAULT_PROJECT = Path(__file__).resolve().parent / "runs"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--model", default="yolov8n-pose.pt")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default=None, help="e.g. 0, cpu; default lets ultralytics auto-pick")
    ap.add_argument("--workers", type=int, default=8,
                     help="CPU dataloader workers (ultralytics' equivalent of sklearn's n_jobs) -- "
                          "raise toward os.cpu_count() if GPU-util stays low waiting on data loading")
    ap.add_argument("--cache", choices=["ram", "disk", "none"], default="ram",
                     help="cache decoded images across epochs instead of re-reading+decoding from disk "
                          "every epoch -- biggest single speedup for a small dataset run for many epochs")
    ap.add_argument("--name", default="busbar_pose")
    ap.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    ap.add_argument("--degrees", type=float, default=15.0)
    ap.add_argument("--mixup", type=float, default=0.1)
    args = ap.parse_args()

    if not args.data.exists():
        raise SystemExit(f"{args.data} not found -- run prepare_dataset.py first")

    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        cache=False if args.cache == "none" else args.cache,
        project=str(args.project),
        name=args.name,
        degrees=args.degrees,
        mixup=args.mixup,
        plots=True,
    )

    # ultralytics saves results.csv/plots/weights on its own but never a plain-text
    # summary -- write one now so it doesn't need a one-off script every time.
    run_dir = Path(model.trainer.save_dir)
    print()
    print(write_summary(run_dir))
    print(f"\nsummary.txt -> {run_dir / 'summary.txt'}")


if __name__ == "__main__":
    main()
