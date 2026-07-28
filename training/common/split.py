"""Shared occlusion-range filtering + train/val split, used by both prepare_dataset.py scripts.

datasets/segmentation and datasets/keypoints both use the same convention:
index 0-499 = light occlusion scene, index 500-999 = heavy occlusion scene
(see datasets/README.md). No filename prefix marks this -- the index range is the label.
"""
import random

HEAVY_OFFSET = 500


def occlusion_of(index: int) -> str:
    return "light" if index < HEAVY_OFFSET else "heavy"


def filter_by_occlusion(indices, occlusion: str):
    if occlusion == "both":
        return list(indices)
    if occlusion not in ("light", "heavy"):
        raise ValueError(f"occlusion must be light/heavy/both, got {occlusion!r}")
    return [i for i in indices if occlusion_of(i) == occlusion]


def train_val_split(indices, val_ratio: float = 0.2, seed: int = 42):
    """Shuffle (fixed seed) then split, so light/heavy end up mixed in both splits
    even though their indices are contiguous blocks."""
    indices = list(indices)
    random.Random(seed).shuffle(indices)
    n_val = max(1, int(len(indices) * val_ratio))
    val = sorted(indices[:n_val])
    train = sorted(indices[n_val:])
    return train, val
