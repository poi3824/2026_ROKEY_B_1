#!/usr/bin/env bash
#
# Isaac Sim 환경에 포함된 spdlog/fmt/Python 라이브러리가 ROS 2 Humble에
# 섞이지 않도록 ROS 관련 경로를 초기화한 뒤 FMS launch를 실행한다.
#
# 사용:
#   ./scripts/launch_fms_clean.sh
#   ./scripts/launch_fms_clean.sh --check

set -euo pipefail

readonly PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly ROS_SETUP="/opt/ros/humble/setup.bash"
readonly WORKSPACE_SETUP="${PROJECT_DIR}/install/setup.bash"

if [[ ! -f "${ROS_SETUP}" ]]; then
    echo "ROS 2 Humble setup을 찾을 수 없습니다: ${ROS_SETUP}" >&2
    exit 1
fi

if [[ ! -f "${WORKSPACE_SETUP}" ]]; then
    echo "워크스페이스가 빌드되지 않았습니다: ${WORKSPACE_SETUP}" >&2
    echo "먼저 colcon build를 실행하세요." >&2
    exit 1
fi

# isaac_ros2 alias 또는 Isaac Python 환경이 남긴 ABI 충돌 경로 제거.
unset LD_LIBRARY_PATH
unset PYTHONPATH
unset AMENT_PREFIX_PATH
unset CMAKE_PREFIX_PATH
unset COLCON_PREFIX_PATH

# ROS 2의 generated setup 스크립트는 정의되지 않은 환경변수를 조회하므로
# source 구간에서만 nounset을 끈다.
set +u
# shellcheck disable=SC1091
source "${ROS_SETUP}"
# shellcheck disable=SC1091
source "${WORKSPACE_SETUP}"
set -u

if [[ "${1:-}" == "--check" ]]; then
    /usr/bin/python3 -c "import rclpy; print('rclpy import OK')"
    ros2 pkg prefix fms_bringup
    ldd /opt/ros/humble/lib/librcl_logging_spdlog.so \
        | grep -E 'libspdlog|libfmt|not found'
    exit 0
fi

exec ros2 launch fms_bringup fms_bringup.launch.py "$@"
