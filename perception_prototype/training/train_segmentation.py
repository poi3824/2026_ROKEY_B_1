"""변환된 데이터로 YOLO bolt segmentation 모델을 학습한다."""

import argparse
from pathlib import Path

from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO bolt segmentation 학습")
    parser.add_argument(
        "--data", type=Path, default=PROJECT_DIR / "datasets" / "bolt_seg" / "data.yaml"
    )
    parser.add_argument("--model", type=Path, default=PROJECT_DIR / "yolov8n-seg.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", help="예: 0, cpu. 생략하면 Ultralytics 자동 선택")
    parser.add_argument("--name", default="bolt_seg")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data.is_file():
        raise FileNotFoundError(f"data.yaml이 없습니다: {args.data}")
    if not args.model.is_file():
        raise FileNotFoundError(f"초기 모델이 없습니다: {args.model}")

    train_args = {
        "data": str(args.data.resolve()),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "project": str((PROJECT_DIR / "runs" / "segment").resolve()),
        "name": args.name,
        "plots": True,
    }
    if args.device is not None:
        train_args["device"] = args.device

    model = YOLO(str(args.model.resolve()))
    results = model.train(**train_args)
    save_dir = Path(results.save_dir)
    print(f"best model: {save_dir / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
