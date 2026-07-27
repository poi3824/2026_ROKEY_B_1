#!/usr/bin/env python3
"""같은 SimulationApp에서 물리 AMR 주행과 M0609 전체 조립을 실행한다."""

import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
EXECUTOR = ROOT / "isaacpjt" / "M0609" / "execute_isaac.py"

if not EXECUTOR.is_file():
    raise FileNotFoundError(f"전체 공정 executor가 없습니다: {EXECUTOR}")

runpy.run_path(str(EXECUTOR), run_name="__main__")
