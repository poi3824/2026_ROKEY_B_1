"""
assembly_nut_fraction.py가 저장한 logs/screw_log.csv를 읽어
너트 체결(Screwing) 과정을 시각화하는 후처리 스크립트.

Isaac Sim 프로세스와 분리해서 실행한다 (matplotlib GUI가 SimulationApp
렌더 루프와 충돌하는 것을 피하기 위해 시뮬레이션 종료 후 별도 실행).

사용법:
    python plot_screw_log.py [csv_path]
"""
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt

TORQUE_THRESHOLD = 45.0  # assembly_nut_fraction.py의 TORQUE_THRESHOLD와 동일하게 유지
NUT_COLORS = {1: "tab:blue", 2: "tab:orange"}


def load_log(csv_path):
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "nut_id": int(r["nut_id"]),
                "step": int(r["step"]),
                "pass_idx": int(r["pass_idx"]),
                "theta_deg": float(r["theta_deg"]),
                "total_deg": float(r["total_deg"]),
                "depth_mm": float(r["depth_mm"]),
                "tcp_z": float(r["tcp_z"]),
                "torque_nm": float(r["torque_nm"]),
                "stuck_counter": int(r["stuck_counter"]),
                "seated": r["seated"] == "True",
            })
    return rows


def plot_nut(ax_torque, ax_depth, ax_z, rows, nut_id):
    color = NUT_COLORS[nut_id]
    nut_rows = [r for r in rows if r["nut_id"] == nut_id]
    if not nut_rows:
        return

    total_deg = [r["total_deg"] for r in nut_rows]
    torque = [r["torque_nm"] for r in nut_rows]
    depth = [r["depth_mm"] for r in nut_rows]
    step = [r["step"] for r in nut_rows]
    tcp_z = [r["tcp_z"] for r in nut_rows]
    seated_deg = [r["total_deg"] for r in nut_rows if r["seated"]]
    seated_torque = [r["torque_nm"] for r in nut_rows if r["seated"]]

    ax_torque.plot(total_deg, torque, label=f"Nut {nut_id}", color=color)
    if seated_deg:
        ax_torque.scatter(seated_deg, seated_torque, color=color, marker="x", s=80,
                           zorder=5, label=f"Nut {nut_id} Seated")

    ax_depth.plot(total_deg, depth, label=f"Nut {nut_id}", color=color)
    ax_z.plot(step, tcp_z, label=f"Nut {nut_id}", color=color)


def main():
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "logs" / "screw_log.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"체결 로그 파일을 찾을 수 없습니다: {csv_path}")

    rows = load_log(csv_path)

    fig, (ax_torque, ax_depth, ax_z) = plt.subplots(3, 1, figsize=(9, 11))

    for nut_id in (1, 2):
        plot_nut(ax_torque, ax_depth, ax_z, rows, nut_id)

    ax_torque.axhline(TORQUE_THRESHOLD, color="red", linestyle="--", linewidth=1, label="Torque Threshold")
    ax_torque.set_xlabel("Cumulative Rotation (deg)")
    ax_torque.set_ylabel("Joint 6 Torque (Nm)")
    ax_torque.set_title("Torque-Angle Signature")
    ax_torque.legend()
    ax_torque.grid(True, alpha=0.3)

    ax_depth.set_xlabel("Cumulative Rotation (deg)")
    ax_depth.set_ylabel("Engagement Depth (mm)")
    ax_depth.set_title("Depth-Angle Curve")
    ax_depth.legend()
    ax_depth.grid(True, alpha=0.3)

    ax_z.set_xlabel("Simulation Step")
    ax_z.set_ylabel("TCP Z Height (m)")
    ax_z.set_title("TCP Height over Time")
    ax_z.legend()
    ax_z.grid(True, alpha=0.3)

    fig.tight_layout()

    out_path = csv_path.parent / "screw_analysis.png"
    fig.savefig(out_path, dpi=150)
    print(f"[INFO] 그래프 저장 완료 -> {out_path}")

    plt.show()


if __name__ == "__main__":
    main()
