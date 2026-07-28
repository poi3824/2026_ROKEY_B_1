"""Human-readable training summary from ultralytics' results.csv.

Ultralytics auto-saves results.csv/results.png/confusion matrices/weights when training
finishes, but not a plain-text summary -- this fills that gap so every train.py run leaves
a summary.txt behind instead of needing a one-off script each time.
"""
import csv
from pathlib import Path


def _metric_columns(fieldnames):
    return [f for f in fieldnames if f.startswith("metrics/")]


def _fitness(row, metric_cols) -> float:
    """Reconstructs ultralytics' own model-selection criterion: for each task head present
    (box/mask/pose), 0.1*mAP50 + 0.9*mAP50-95, summed across heads. BaseTrainer saves
    weights/best.pt whenever an epoch's fitness strictly exceeds the best seen so far, so
    the epoch that maximizes this is the epoch best.pt came from -- the checkpoint itself
    doesn't retain that number (ultralytics strips epoch/optimizer state via
    strip_optimizer() once training finishes, so it can't be read back off best.pt)."""
    total = 0.0
    for suffix in ("(B)", "(M)", "(P)"):
        map50_col = f"metrics/mAP50{suffix}"
        map5095_col = f"metrics/mAP50-95{suffix}"
        if map50_col in metric_cols and map5095_col in metric_cols:
            total += 0.1 * float(row[map50_col] or 0) + 0.9 * float(row[map5095_col] or 0)
    return total


def summarize_run(run_dir: Path) -> str:
    csv_path = run_dir / "results.csv"
    rows = list(csv.DictReader(csv_path.open()))
    if not rows:
        return "(no epochs logged)"

    metric_cols = _metric_columns(rows[0].keys())
    final = rows[-1]
    best = max(rows, key=lambda r: _fitness(r, metric_cols))

    lines = [f"Run: {run_dir.name}", f"Total epochs: {len(rows)}", ""]
    lines.append(f"=== Final epoch ({final['epoch']}) ===")
    for c in metric_cols:
        lines.append(f"{c}: {float(final[c]):.4f}")

    lines.append("")
    lines.append(f"=== weights/best.pt came from epoch {best['epoch']} "
                  f"(fitness={_fitness(best, metric_cols):.4f}) ===")
    for c in metric_cols:
        lines.append(f"{c}: {float(best[c]):.4f}")
    if best is final:
        lines.append("(best epoch == final epoch)")

    return "\n".join(lines)


def write_summary(run_dir: Path) -> str:
    text = summarize_run(run_dir)
    (run_dir / "summary.txt").write_text(text + "\n")
    return text
