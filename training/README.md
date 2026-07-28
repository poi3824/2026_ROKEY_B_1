# training/

Two tracks for the busbar grasp-point problem (see `datasets/README.md` for the data these
read), plus a shared comparison harness. Neither track trains anything by itself when you
`prepare_dataset.py` -- that step only builds the YOLO-format dataset; running `train.py` is
a separate, possibly-GPU-bound step left to you (same as the existing Colab-notebook workflow
this repo already used for the original baseline).

```
training/
├── common/            # CLASS_NAMES/CLASS_TO_ID (from datasets/classes.json) + occlusion
│                       # index filtering / train-val split shared by both tracks
├── segmentation/       # Track A: improve the existing YOLOv8n-seg approach
│   ├── prepare_dataset.py
│   ├── train.py
│   └── postprocess.py  # inference-time mask cleanup + PCA orientation (see its docstring)
├── keypoints/           # Track B: new YOLOv8n-pose approach (no prior code existed for this)
│   ├── prepare_dataset.py
│   └── train.py
└── eval/
    ├── compare_grasp_point.py   # scores both trained models on the same held-out frames
    ├── overlay.py                # image-drawing helper used by compare_grasp_point.py
    └── results/                   # generated: raw_results.csv, summary.md, overlays/, plots/
```

## Why two tracks

The original baseline (YOLOv8n-seg, default hyperparameters, light-occlusion data only)
produces a noisy mask -> the grasp point is a raw moment centroid of that noisy contour,
with no orientation at all. Track A fixes the parts of that pipeline that were simply never
tuned (heavy-occlusion data unused, default augmentation, no mask cleanup). Track B tests
whether the underlying approach is the real limit: a keypoint/pose model regresses point
locations directly (including for occluded points, via the visibility-aware labels described
in `training/keypoints/prepare_dataset.py`), which structurally can't be blocked by a noisy
mask contour the way a segmentation model can. `compare_grasp_point.py` is what turns that
into a real answer instead of two competing guesses.

## Running it

```bash
# Track A
python3 training/segmentation/prepare_dataset.py --occlusion both --cams color
python3 training/segmentation/train.py --epochs 150

# Track B
python3 training/keypoints/prepare_dataset.py --occlusion both --cams color
python3 training/keypoints/train.py --epochs 150

# Comparison (after both best.pt exist)
python3 training/eval/compare_grasp_point.py \
    --seg-weights training/segmentation/runs/busbar_seg/weights/best.pt \
    --pose-weights training/keypoints/runs/busbar_pose/weights/best.pt \
    --target-class busbar
```

Requires `pip install -r training/requirements.txt` (`ultralytics` + `matplotlib`) for
`train.py` and `compare_grasp_point.py` -- not installed in this environment, so those two
were verified by code review + `py_compile` only, not by an actual training run.
`prepare_dataset.py` (both tracks) and `postprocess.py` only need `opencv-python`/`numpy`,
which are present, and were smoke-tested end to end against the real dataset.

### Troubleshooting: `_ARRAY_API not found` / numpy version conflict on `train.py`

If `train.py` (or anything importing `ultralytics`) fails with
`AttributeError: _ARRAY_API not found` or `ImportError: numpy.core.multiarray failed to
import` when it tries to import matplotlib: `pip install ultralytics` pulled a numpy 2.x
and an `opencv-python` wheel into `~/.local/lib/python3.10/site-packages` (which
`sys.path` resolves before anything apt-installed), but `matplotlib` on this machine was
still the apt system package, compiled against numpy 1.x -- so importing numpy 2.x breaks
matplotlib's C extensions. Pinning `numpy<2` "fixes" that but then conflicts the other way,
since the pip-installed `opencv-python` requires `numpy>=2`. matplotlib is the only piece
still on the apt/numpy1 side, so bring it onto the same pip/numpy2 side as the rest instead
of downgrading numpy:

```bash
pip install --user "numpy<2"              # undo an earlier numpy<2 pin if you tried one
pip install --user --upgrade numpy matplotlib
```

This updates `~/.local` (not a venv), so it also affects any other script outside this repo
that imports matplotlib/numpy under this user account -- nothing else in this repo depends
on matplotlib, so that's expected to be safe here.

## Reading `compare_grasp_point.py`'s output

`results/summary.md` breaks pixel error and orientation error down by
`{seg_raw, seg_clean, pose} x {light, heavy}` occlusion. `seg_raw` matches the original
baseline's extraction method exactly, so it's the reference point for "did tuning help" and
"did switching to keypoints help", independently. `results/overlays/` has representative
before/after images per occlusion level in the same visual style as the original screenshot
(predicted point + orientation vs. ground truth). `results/plots/` has the same numbers as
bar charts.
