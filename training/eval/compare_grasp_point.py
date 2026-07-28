#!/usr/bin/env python3
"""Compare the segmentation track and the keypoint track on the same held-out frames, for
a single target class (default: busbar -- the class in the original screenshot).

Requires trained checkpoints from both tracks:
    training/segmentation/train.py -> runs/<name>/weights/best.pt
    training/keypoints/train.py    -> runs/<name>/weights/best.pt

For each held-out frame it computes a predicted grasp point + orientation three ways:
  - seg_raw:   segmentation mask -> raw moment centroid (matches the original baseline)
  - seg_clean: same mask, after training/segmentation/postprocess.py's morphology+PCA
  - pose:      keypoint model's predicted "Center" keypoint + a corner-pair orientation

...and compares each to ground truth (the simulator's projected "Center" keypoint /
corner pair, from datasets/keypoints/<cam>/<idx>.json -- available regardless of which
track is being scored, since it's independent of both models).

Held-out frames are regenerated with training.common.split.train_val_split using the same
defaults prepare_dataset.py used, so no dependency on the (gitignored) training/*/data/
folders -- only on datasets/ and the two checkpoints.

NOTE ON "frame-to-frame jitter": these are independent synthetic scenes, not a video of one
static scene, so the "variance" reported here is the spread of the pixel error across the
held-out set, not literal temporal jitter of a fixed object. It's still the right proxy for
"how much would the reported grasp point move around for a similar-looking input" -- a model
whose error variance is high will also jitter across consecutive real camera frames -- but
it isn't measured on video.

Usage:
    python compare_grasp_point.py \\
        --seg-weights ../segmentation/runs/busbar_seg/weights/best.pt \\
        --pose-weights ../keypoints/runs/busbar_pose/weights/best.pt \\
        --target-class busbar
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from training.common.classes import CLASS_TO_ID
from training.common.split import occlusion_of, train_val_split
from training.segmentation.postprocess import extract_grasp_point
from overlay import save_comparison_overlay

REPO_ROOT = Path(__file__).resolve().parents[2]
SEG_ROOT = REPO_ROOT / "datasets" / "segmentation"
KP_ROOT = REPO_ROOT / "datasets" / "keypoints"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

# corner pair used to define a signed orientation vector from the 9 cuboid keypoints
ORIENTATION_PAIR = ("LUB", "RUB")


def ground_truth_for(idx: int, target_class: str):
    data = json.loads((KP_ROOT / "color" / f"{idx:06d}.json").read_text())
    order = data["keypoint_order"]
    for obj in data["objects"]:
        if obj["label"] != target_class:
            continue
        pts = obj["cuboid_keypoints_projected"]
        center = np.array(pts[order.index("Center")], dtype=np.float64)
        a = np.array(pts[order.index(ORIENTATION_PAIR[0])], dtype=np.float64)
        b = np.array(pts[order.index(ORIENTATION_PAIR[1])], dtype=np.float64)
        axis = b - a
        angle = float(np.arctan2(axis[1], axis[0]))
        return center, angle
    return None, None


def angle_diff_deg(a, b):
    """Undirected line-angle difference in degrees (0-90), since PCA orientation has no sign."""
    if a is None or b is None:
        return None
    d = np.degrees(abs(a - b)) % 180.0
    return min(d, 180.0 - d)


def predict_seg(model, target_id, img_bgr, conf=0.6, iou=0.45):
    t0 = time.perf_counter()
    r = model.predict(img_bgr, conf=conf, iou=iou, verbose=False)[0]
    dt = time.perf_counter() - t0
    if r.masks is None or r.boxes is None:
        return None, dt
    cls_ids = r.boxes.cls.cpu().numpy().astype(int)
    matches = np.where(cls_ids == target_id)[0]
    if len(matches) == 0:
        return None, dt
    best = matches[np.argmax(r.boxes.conf.cpu().numpy()[matches])]
    h, w = img_bgr.shape[:2]
    poly = r.masks.xy[best].astype(np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    raw = extract_grasp_point(mask, apply_cleanup=False)
    clean = extract_grasp_point(mask, apply_cleanup=True)
    return {"raw": raw, "clean": clean}, dt


def predict_pose(model, target_id, keypoint_order, img_bgr, conf=0.6, iou=0.45):
    t0 = time.perf_counter()
    r = model.predict(img_bgr, conf=conf, iou=iou, verbose=False)[0]
    dt = time.perf_counter() - t0
    if r.keypoints is None or r.boxes is None:
        return None, dt
    cls_ids = r.boxes.cls.cpu().numpy().astype(int)
    matches = np.where(cls_ids == target_id)[0]
    if len(matches) == 0:
        return None, dt
    best = matches[np.argmax(r.boxes.conf.cpu().numpy()[matches])]
    kpts = r.keypoints.xy[best].cpu().numpy()  # (K, 2)
    center = kpts[keypoint_order.index("Center")]
    a = kpts[keypoint_order.index(ORIENTATION_PAIR[0])]
    b = kpts[keypoint_order.index(ORIENTATION_PAIR[1])]
    axis = b - a
    angle = float(np.arctan2(axis[1], axis[0]))
    return {"center": center, "angle": angle}, dt


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seg-weights", type=Path, required=True)
    ap.add_argument("--pose-weights", type=Path, required=True)
    ap.add_argument("--target-class", default="busbar")
    ap.add_argument("--conf", type=float, default=0.6, help="NMS confidence threshold (matches perception_node's default)")
    ap.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold (matches perception_node's default)")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-overlays", type=int, default=8, help="representative overlay images to save")
    ap.add_argument("--out", type=Path, default=RESULTS_DIR)
    args = ap.parse_args()

    from ultralytics import YOLO
    seg_model = YOLO(str(args.seg_weights))
    pose_model = YOLO(str(args.pose_weights))
    target_id = CLASS_TO_ID[args.target_class]

    all_indices = sorted(
        int(p.stem) for p in (KP_ROOT / "color").glob("*.json") if p.stem.isdigit() and len(p.stem) == 6
    )
    _, val_indices = train_val_split(all_indices, args.val_ratio, args.seed)

    keypoint_order = json.loads((KP_ROOT / "color" / f"{val_indices[0]:06d}.json").read_text())["keypoint_order"]

    rows = []
    overlays_saved = 0
    (args.out / "overlays").mkdir(parents=True, exist_ok=True)
    (args.out / "plots").mkdir(parents=True, exist_ok=True)

    for idx in val_indices:
        gt_center, gt_angle = ground_truth_for(idx, args.target_class)
        if gt_center is None:
            continue

        # datasets/segmentation/ only covers index 0-999 -- keypoints-only capture batches
        # (e.g. the busbar-occlusion-focused 1000-1299 range, see datasets/README.md) have
        # no segmentation counterpart, so seg evaluation is skipped for those frames while
        # pose evaluation (which only needs the keypoints tree) still runs normally.
        seg_img_path = SEG_ROOT / "color" / f"rgb_{idx:04d}.png"
        pose_img = cv2.imread(str(KP_ROOT / "color" / f"{idx:06d}.png"))

        if seg_img_path.exists():
            seg_img = cv2.imread(str(seg_img_path))
            seg_pred, seg_dt = predict_seg(seg_model, target_id, seg_img, args.conf, args.iou)
        else:
            seg_pred, seg_dt = None, None
        pose_pred, pose_dt = predict_pose(pose_model, target_id, keypoint_order, pose_img, args.conf, args.iou)

        row = {"idx": idx, "occlusion": occlusion_of(idx)}
        for tag, pred in [("seg_raw", seg_pred and seg_pred["raw"]), ("seg_clean", seg_pred and seg_pred["clean"])]:
            if pred and pred["centroid"] is not None:
                err = float(np.linalg.norm(np.array(pred["centroid"]) - gt_center))
                ang_err = angle_diff_deg(pred["angle_rad"], gt_angle)
            else:
                err, ang_err = None, None
            row[f"{tag}_px_err"] = err
            row[f"{tag}_angle_err_deg"] = ang_err
        row["seg_infer_s"] = seg_dt

        if pose_pred:
            row["pose_px_err"] = float(np.linalg.norm(pose_pred["center"] - gt_center))
            row["pose_angle_err_deg"] = angle_diff_deg(pose_pred["angle"], gt_angle)
        else:
            row["pose_px_err"] = None
            row["pose_angle_err_deg"] = None
        row["pose_infer_s"] = pose_dt

        rows.append(row)

        if overlays_saved < args.num_overlays and seg_pred and pose_pred:
            save_comparison_overlay(
                seg_img, pose_img, seg_pred, pose_pred, gt_center,
                args.out / "overlays" / f"{idx:06d}_{row['occlusion']}.png",
            )
            overlays_saved += 1

    _write_outputs(rows, args.out)


def _write_outputs(rows, out_dir: Path):
    import csv
    with open(out_dir / "raw_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    groups = {}
    for occlusion in ["light", "heavy"]:
        subset = [r for r in rows if r["occlusion"] == occlusion]
        for tag in ["seg_raw", "seg_clean", "pose"]:
            errs = [r[f"{tag}_px_err"] for r in subset if r.get(f"{tag}_px_err") is not None]
            angs = [r[f"{tag}_angle_err_deg"] for r in subset if r.get(f"{tag}_angle_err_deg") is not None]
            groups[(tag, occlusion)] = {
                "n": len(subset),
                "detected": len(errs),
                "px_err_mean": float(np.mean(errs)) if errs else None,
                "px_err_std": float(np.std(errs)) if errs else None,
                "angle_err_mean_deg": float(np.mean(angs)) if angs else None,
            }

    lines = ["# Grasp-point comparison: {seg_raw, seg_clean, pose} x {light, heavy}\n"]
    lines.append("| track | occlusion | n frames | detected | px err mean | px err std | angle err mean (deg) |")
    lines.append("|---|---|---|---|---|---|---|")
    for (tag, occ), g in groups.items():
        lines.append(
            f"| {tag} | {occ} | {g['n']} | {g['detected']} | "
            f"{g['px_err_mean']:.2f} | {g['px_err_std']:.2f} | {g['angle_err_mean_deg']:.2f} |"
            if g["px_err_mean"] is not None else
            f"| {tag} | {occ} | {g['n']} | {g['detected']} | - | - | - |"
        )
    infer_seg = [r["seg_infer_s"] for r in rows if r.get("seg_infer_s") is not None]
    infer_pose = [r["pose_infer_s"] for r in rows if r.get("pose_infer_s") is not None]
    if infer_seg:
        lines.append(f"\nmean seg inference: {np.mean(infer_seg) * 1000:.1f} ms/frame")
    if infer_pose:
        lines.append(f"mean pose inference: {np.mean(infer_pose) * 1000:.1f} ms/frame")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    _save_plots(groups, out_dir / "plots")


def _save_plots(groups, plots_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tags = ["seg_raw", "seg_clean", "pose"]
    occlusions = ["light", "heavy"]
    colors = {"light": "#5B8FF9", "heavy": "#F6903D"}

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(tags))
    width = 0.35
    for i, occ in enumerate(occlusions):
        means = [groups[(t, occ)]["px_err_mean"] or 0 for t in tags]
        stds = [groups[(t, occ)]["px_err_std"] or 0 for t in tags]
        ax.bar(x + (i - 0.5) * width, means, width, yerr=stds, capsize=4, label=occ, color=colors[occ])
    ax.set_xticks(x)
    ax.set_xticklabels(tags)
    ax.set_ylabel("grasp point pixel error (mean ± std)")
    ax.set_title("Segmentation vs Keypoint grasp-point accuracy")
    ax.legend(title="occlusion")
    fig.tight_layout()
    fig.savefig(plots_dir / "pixel_error.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, occ in enumerate(occlusions):
        vals = [groups[(t, occ)]["angle_err_mean_deg"] or 0 for t in tags]
        ax.bar(x + (i - 0.5) * width, vals, width, label=occ, color=colors[occ])
    ax.set_xticks(x)
    ax.set_xticklabels(tags)
    ax.set_ylabel("orientation error (deg, mean)")
    ax.set_title("Segmentation vs Keypoint orientation accuracy")
    ax.legend(title="occlusion")
    fig.tight_layout()
    fig.savefig(plots_dir / "orientation_error.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
