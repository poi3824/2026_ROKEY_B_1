#!/usr/bin/env python3
"""Generate a single-frame "does detection/grasp-point work" check, in the same visual
style as the original detector screenshot (yellow bbox, green mask contour, red grasp
point, black px/cam/world readout card) -- but using a real trained checkpoint's actual
prediction on a held-out frame, not ground truth.

Works with either track's checkpoint -- auto-detects segmentation vs pose from the loaded
model. Segmentation: cleans the predicted mask (training/segmentation/postprocess.py) and
uses its centroid as the grasp point. Pose: uses the predicted "Center" keypoint directly
as the grasp point (no mask exists for this task, so the mask contour is skipped and the
raw keypoints are drawn instead). Either way the grasp point is backprojected through real
depth + camera intrinsics to camera/world coordinates
(training/eval/camera_geometry.py, validated against this dataset's own ground truth).

Usage:
    python make_demo_frame.py                                  # auto-picks a held-out frame with the target class
    python make_demo_frame.py --idx 517 --occlusion heavy       # a specific frame
    python make_demo_frame.py --weights ../segmentation/runs/busbar_seg_batch8/weights/best.pt
    python make_demo_frame.py --weights ../keypoints/runs/busbar_pose/weights/best.pt
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from training.common.classes import CLASS_TO_ID
from training.common.split import occlusion_of, train_val_split, filter_by_occlusion
from training.segmentation.postprocess import extract_grasp_point
from camera_geometry import pixel_to_world_point

REPO_ROOT = Path(__file__).resolve().parents[2]
SEG_ROOT = REPO_ROOT / "datasets" / "segmentation"
KP_ROOT = REPO_ROOT / "datasets" / "keypoints"
DEFAULT_WEIGHTS = Path(__file__).resolve().parents[1] / "keypoints" / "runs" / "busbar_pose" / "weights" / "best.pt"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "results" / "demo_frames"


def candidate_indices(occlusion: str, val_ratio: float, seed: int):
    all_indices = sorted(
        int(p.stem) for p in (KP_ROOT / "color").glob("*.json") if p.stem.isdigit() and len(p.stem) == 6
    )
    _, val_indices = train_val_split(all_indices, val_ratio, seed)
    return filter_by_occlusion(val_indices, occlusion)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    ap.add_argument("--target-class", default="busbar")
    ap.add_argument("--idx", type=int, default=None, help="specific frame index; default auto-picks a held-out frame")
    ap.add_argument("--occlusion", choices=["light", "heavy", "any"], default="heavy",
                     help="which scene to pull the auto-picked frame from (default: heavy, the harder case)")
    ap.add_argument("--cam", default="color", choices=["color", "depth", "left", "right"])
    ap.add_argument("--conf", type=float, default=0.6, help="NMS confidence threshold (matches perception_node's default)")
    ap.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold (matches perception_node's default)")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if not args.weights.exists():
        raise SystemExit(f"{args.weights} not found -- train the segmentation model first")

    occlusion_filter = "both" if args.occlusion == "any" else args.occlusion
    candidates = [args.idx] if args.idx is not None else candidate_indices(occlusion_filter, args.val_ratio, args.seed)
    if not candidates:
        raise SystemExit(f"no held-out frames for occlusion={args.occlusion}")

    from ultralytics import YOLO
    from overlay import save_detection_card
    model = YOLO(str(args.weights))
    target_id = CLASS_TO_ID[args.target_class]
    is_pose = model.task == "pose"

    keypoint_order = None
    if is_pose:
        # any frame's JSON has the same fixed order (see datasets/README.md)
        sample = json.loads((KP_ROOT / args.cam / f"{candidates[0]:06d}.json").read_text())
        keypoint_order = sample["keypoint_order"]
        center_idx = keypoint_order.index("Center")

    for idx in candidates:
        img_path = SEG_ROOT / args.cam / f"rgb_{idx:04d}.png"
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path))
        r = model.predict(img, conf=args.conf, iou=args.iou, verbose=False)[0]
        if r.boxes is None or (is_pose and r.keypoints is None) or (not is_pose and r.masks is None):
            continue
        cls_ids = r.boxes.cls.cpu().numpy().astype(int)
        matches = np.where(cls_ids == target_id)[0]
        if len(matches) == 0:
            continue

        best = matches[np.argmax(r.boxes.conf.cpu().numpy()[matches])]
        conf = float(r.boxes.conf[best])
        bbox_xyxy = r.boxes.xyxy[best].cpu().numpy()

        mask_used = None
        keypoints_px = None
        if is_pose:
            keypoints_px = r.keypoints.xy[best].cpu().numpy()
            u, v = keypoints_px[center_idx]
        else:
            h, w = img.shape[:2]
            poly = r.masks.xy[best].astype(np.int32)
            raw_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(raw_mask, [poly], 255)
            result = extract_grasp_point(raw_mask, apply_cleanup=True)
            if result["centroid"] is None:
                continue
            u, v = result["centroid"]
            mask_used = result["mask_used"]

        depth_map = np.load(SEG_ROOT / args.cam / f"distance_to_image_plane_{idx:04d}.npy")
        depth = float(depth_map[int(round(v)), int(round(u))])
        if depth <= 0:
            continue

        camera_data = json.loads((KP_ROOT / args.cam / f"{idx:06d}.json").read_text())["camera_data"]
        cam_pt, world_pt = pixel_to_world_point(u, v, depth, camera_data)

        out_path = args.out or DEFAULT_OUT_DIR / f"{idx:06d}_{occlusion_of(idx)}_{model.task}.png"
        save_detection_card(
            img, mask_used, bbox_xyxy, (u, v), args.target_class, conf,
            cam_pt, world_pt, out_path, keypoints=keypoints_px,
        )
        print(f"frame {idx} ({occlusion_of(idx)}, task={model.task}): {args.target_class} {conf:.2f}, "
              f"px=({u:.0f},{v:.0f}) depth={depth:.3f}m cam={cam_pt.round(3).tolist()} "
              f"world={world_pt.round(3).tolist()}")
        print(f"saved -> {out_path}")
        return

    raise SystemExit(f"no frame among {len(candidates)} candidates produced a {args.target_class} detection "
                      f"-- try --occlusion any, a different --idx, or lowering --conf")


if __name__ == "__main__":
    main()
