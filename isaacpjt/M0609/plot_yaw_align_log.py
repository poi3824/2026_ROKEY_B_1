"""
assembly_nut_fraction1.py가 저장한 logs/yaw_align_log.csv를 읽어
YawAligner(PI) 실시간 정렬 보정 동작을 시각화하는 후처리 스크립트.

Isaac Sim 프로세스와 분리해서 실행한다 (matplotlib GUI가 SimulationApp
렌더 루프와 충돌하는 것을 피하기 위해 시뮬레이션 종료 후 별도 실행).

사용법:
    python plot_yaw_align_log.py [csv_path]
"""
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt

OUT_LIMIT_DEG = 15.0  # assembly_nut_fraction1.py의 YawAligner(out_limit_deg=15.0)와 동일하게 유지


def load_log(csv_path):
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "t_s": float(r["t_s"]),
                "step": int(r["step"]),
                "phase": r["phase"],
                "error_deg": float(r["error_deg"]),
                "correction_deg": float(r["correction_deg"]),
                "dx_mm": float(r["dx_mm"]),
                "dy_mm": float(r["dy_mm"]),
                "saturated": r["saturated"] == "True",
            })
    return rows


def main():
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "logs" / "yaw_align_log.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"YawAligner 로그 파일을 찾을 수 없습니다: {csv_path}")

    rows = load_log(csv_path)

    t = [r["t_s"] for r in rows]
    error_deg = [r["error_deg"] for r in rows]
    correction_deg = [r["correction_deg"] for r in rows]
    dx_mm = [r["dx_mm"] for r in rows]
    dy_mm = [r["dy_mm"] for r in rows]
    sat_t = [r["t_s"] for r in rows if r["saturated"]]
    sat_corr = [r["correction_deg"] for r in rows if r["saturated"]]

    fig, (ax_deg, ax_xy) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    ax_deg.plot(t, error_deg, label="raw error (dTheta, deg)", color="tab:red", alpha=0.6)
    ax_deg.plot(t, correction_deg, label="PI correction_deg", color="tab:blue")
    if sat_t:
        ax_deg.scatter(sat_t, sat_corr, color="black", marker="x", s=50, zorder=5, label="saturated")
    ax_deg.axhline(OUT_LIMIT_DEG, color="gray", linestyle="--", linewidth=1, label=f"out_limit_deg (±{OUT_LIMIT_DEG})")
    ax_deg.axhline(-OUT_LIMIT_DEG, color="gray", linestyle="--", linewidth=1)
    ax_deg.axhline(0.0, color="black", linewidth=0.5)
    ax_deg.set_ylabel("deg")
    ax_deg.set_title("YawAligner PI: error vs correction (수렴/진동/포화 확인)")
    ax_deg.legend()
    ax_deg.grid(True, alpha=0.3)

    ax_xy.plot(t, dx_mm, label="dx (mm)", color="tab:green")
    ax_xy.plot(t, dy_mm, label="dy (mm)", color="tab:orange")
    ax_xy.axhline(0.0, color="black", linewidth=0.5)
    ax_xy.set_xlabel("elapsed time since yaw_aligner.reset() (s)")
    ax_xy.set_ylabel("mm")
    ax_xy.set_title("busbar_insert_xy 실시간 보정값")
    ax_xy.legend()
    ax_xy.grid(True, alpha=0.3)

    fig.tight_layout()

    out_path = csv_path.parent / "yaw_align_analysis.png"
    fig.savefig(out_path, dpi=150)
    print(f"[INFO] 그래프 저장 완료 -> {out_path}")

    plt.show()


if __name__ == "__main__":
    main()
