#!/usr/bin/env python3
"""Busbar 작업 셀 회전을 USD root layer에 영구 저장한다."""

import argparse
from pathlib import Path

from pxr import Usd

from workcell_transform import rotate_busbar_workcell


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("usd", type=Path)
    parser.add_argument("--yaw-deg", type=float, required=True)
    args = parser.parse_args()

    usd_path = args.usd.expanduser().resolve()
    if not usd_path.is_file():
        raise FileNotFoundError(f"USD 파일 없음: {usd_path}")

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"USD stage 로드 실패: {usd_path}")

    root_layer = stage.GetRootLayer()
    report = rotate_busbar_workcell(
        stage,
        args.yaw_deg,
        edit_layer=root_layer,
    )
    if not root_layer.Save():
        raise RuntimeError(f"USD 저장 실패: {usd_path}")

    print(
        f"saved={usd_path}\n"
        f"yaw={report.yaw_deg:+.1f}deg\n"
        f"pivot={report.pivot_world}\n"
        f"approach_yaw_rad={report.busbar_approach_yaw_rad:.12f}"
    )
    for path in report.centers_before:
        print(
            f"{path}: {report.centers_before[path]} -> "
            f"{report.centers_after[path]}"
        )


if __name__ == "__main__":
    main()
