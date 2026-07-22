"""Isaac Sim Replicator ZIP을 YOLO instance-segmentation 형식으로 변환한다."""

import argparse
import ast
import json
import random
import re
import shutil
import zipfile
from pathlib import Path

import cv2
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path.home() / "Downloads" / "pose_dataset.zip"
DEFAULT_OUTPUT = PROJECT_DIR / "datasets" / "bolt_seg"
FRAME_PATTERN = re.compile(r"rgb_(\d+)\.png$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replicator instance masks -> YOLO segmentation labels"
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--classes", nargs="+", default=["bolt"])
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-area", type=float, default=20.0)
    parser.add_argument("--limit", type=int, help="변환 시험용 최대 프레임 수")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def find_dataset_root(names: list[str]) -> str:
    matches = [name for name in names if name.endswith("rgb_0000.png")]
    if len(matches) != 1:
        raise ValueError("ZIP에서 rgb_0000.png의 고유한 위치를 찾지 못했습니다.")
    return matches[0][: -len("rgb_0000.png")]


def find_frame_ids(names: list[str], root: str) -> list[str]:
    frame_ids = []
    for name in names:
        if not name.startswith(root):
            continue
        match = FRAME_PATTERN.fullmatch(name[len(root) :])
        if match:
            frame_ids.append(match.group(1))
    return sorted(frame_ids)


def decode_png(archive: zipfile.ZipFile, name: str, flags: int) -> np.ndarray:
    encoded = np.frombuffer(archive.read(name), dtype=np.uint8)
    image = cv2.imdecode(encoded, flags)
    if image is None:
        raise ValueError(f"PNG를 읽지 못했습니다: {name}")
    return image


def color_key_to_rgba(key: str) -> tuple[int, int, int, int]:
    color = ast.literal_eval(key)
    if not isinstance(color, tuple) or len(color) != 4:
        raise ValueError(f"잘못된 Replicator 색상 키: {key}")
    return tuple(int(channel) for channel in color)


def mask_to_polygon(mask: np.ndarray, min_area: float) -> list[tuple[float, float]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [contour for contour in contours if cv2.contourArea(contour) >= min_area]
    if not contours:
        return []

    contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(contour, True)
    polygon = cv2.approxPolyDP(contour, 0.002 * perimeter, True).reshape(-1, 2)
    if len(polygon) < 3:
        return []
    return [(float(x), float(y)) for x, y in polygon]


def build_labels(
    instance_bgra: np.ndarray,
    semantics: dict,
    class_to_id: dict[str, int],
    min_area: float,
) -> list[str]:
    if instance_bgra.ndim != 3 or instance_bgra.shape[2] != 4:
        raise ValueError(f"인스턴스 마스크가 BGRA 4채널이 아닙니다: {instance_bgra.shape}")

    instance_rgba = cv2.cvtColor(instance_bgra, cv2.COLOR_BGRA2RGBA)
    height, width = instance_rgba.shape[:2]
    rows = []

    for color_key, metadata in semantics.items():
        class_name = metadata.get("class")
        if class_name not in class_to_id:
            continue

        rgba = np.array(color_key_to_rgba(color_key), dtype=np.uint8)
        binary_mask = np.all(instance_rgba == rgba, axis=2).astype(np.uint8) * 255
        polygon = mask_to_polygon(binary_mask, min_area)
        if not polygon:
            continue

        normalized = []
        for x, y in polygon:
            normalized.extend(
                (min(max(x / width, 0.0), 1.0), min(max(y / height, 0.0), 1.0))
            )
        coordinates = " ".join(f"{value:.6f}" for value in normalized)
        rows.append(f"{class_to_id[class_name]} {coordinates}")

    return rows


def prepare_output(output: Path, overwrite: bool) -> None:
    if output.exists() and any(output.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"출력 폴더가 비어 있지 않습니다: {output}\n"
                "다시 만들려면 --overwrite를 사용하세요."
            )
        shutil.rmtree(output)

    for split in ("train", "val"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)


def write_data_yaml(output: Path, classes: list[str]) -> None:
    names = "\n".join(f"  {index}: {name}" for index, name in enumerate(classes))
    content = (
        f"path: {output.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        f"{names}\n"
    )
    (output / "data.yaml").write_text(content, encoding="utf-8")


def convert(args: argparse.Namespace) -> None:
    if not args.source.is_file():
        raise FileNotFoundError(f"원본 ZIP이 없습니다: {args.source}")
    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError("--val-ratio는 0과 1 사이여야 합니다.")
    if len(set(args.classes)) != len(args.classes):
        raise ValueError("--classes에 중복된 이름이 있습니다.")

    prepare_output(args.output, args.overwrite)
    class_to_id = {name: index for index, name in enumerate(args.classes)}

    with zipfile.ZipFile(args.source) as archive:
        names = archive.namelist()
        root = find_dataset_root(names)
        frame_ids = find_frame_ids(names, root)
        if args.limit is not None:
            frame_ids = frame_ids[: args.limit]
        if len(frame_ids) < 2:
            raise ValueError("학습/검증 분리에 필요한 프레임이 부족합니다.")

        rng = random.Random(args.seed)
        rng.shuffle(frame_ids)
        val_count = max(1, round(len(frame_ids) * args.val_ratio))
        val_ids = set(frame_ids[:val_count])
        counts = {"train": 0, "val": 0, "objects": 0, "empty": 0}

        for frame_id in frame_ids:
            split = "val" if frame_id in val_ids else "train"
            rgb_name = f"{root}rgb_{frame_id}.png"
            mask_name = f"{root}instance_segmentation_{frame_id}.png"
            map_name = f"{root}instance_segmentation_semantics_mapping_{frame_id}.json"

            rgb = decode_png(archive, rgb_name, cv2.IMREAD_COLOR)
            mask = decode_png(archive, mask_name, cv2.IMREAD_UNCHANGED)
            semantics = json.loads(archive.read(map_name).decode("utf-8"))
            labels = build_labels(mask, semantics, class_to_id, args.min_area)

            stem = f"rgb_{frame_id}"
            image_path = args.output / "images" / split / f"{stem}.png"
            label_path = args.output / "labels" / split / f"{stem}.txt"
            if not cv2.imwrite(str(image_path), rgb):
                raise OSError(f"이미지를 저장하지 못했습니다: {image_path}")
            label_path.write_text(
                "\n".join(labels) + ("\n" if labels else ""), encoding="utf-8"
            )

            counts[split] += 1
            counts["objects"] += len(labels)
            counts["empty"] += int(not labels)

    write_data_yaml(args.output, args.classes)
    print(f"dataset: {args.output.resolve()}")
    print(
        f"train={counts['train']} val={counts['val']} "
        f"objects={counts['objects']} empty_images={counts['empty']}"
    )


def main() -> None:
    convert(parse_args())


if __name__ == "__main__":
    main()
