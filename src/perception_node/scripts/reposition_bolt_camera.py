"""Camera_bolt를 수직 하향 -> 사각(오블리크)으로 재배치.

문제: 실측 결과(2026-07-25) Camera_bolt world pos=(1.2598,-0.0054,1.2184)가
볼트쌍 바로 위에서 순수 -Z 방향(수직)으로 내려다보고 있었다. 너트 체결의
탑다운 접근 경로와 카메라 시선이 완전히 같은 축이라, 그리퍼가 볼트에
접근할수록 필연적으로 카메라 시야를 가린다 - 재배치가 아니라 "각도를
틀어야" 해결되는 문제.

계산: 볼트쌍(BOLT1_POS, BOLT2_POS, assembly_nut_physics.py 기준) 중간점을
바라보되, 카메라 자체는 그 중간점에서 -Y 방향으로 0.85m 밀어내 세로축(그리퍼
접근 경로)에서 벗어난 사선 위치로 옮긴다. 결과 수직 대비 기울기 약 39도.
1차 추정치 - Isaac Sim에서 실제 렌더로 볼트/너트 프레이밍과 그리퍼 가림
여부를 육안 확인 후 미세조정 필요(카메라 world pos/orient만 더 조정하면 됨,
ROS2 브릿지는 프림에 붙어 있어 그대로 따라간다).

실행 (Isaac Sim 전용 파이썬으로):
  <isaac_sim_root>/python.sh reposition_bolt_camera.py
"""
import numpy as np
from pxr import Usd, UsdGeom, Gf

USD_PATH = "/home/rokey/EV_combine/src/Collected_World_0123/World0123.usd"
CAMERA_PATH = "/World/Camera_bolt"

# assembly_nut_physics.py의 BOLT1_POS/BOLT2_POS와 동일 (isaacpjt/M0609/assembly_nut_physics.py).
BOLT1_POS = np.array([1.0576, 0.3653, 0.152])
BOLT2_POS = np.array([1.2636, 0.0019, 0.152])
TARGET = (BOLT1_POS + BOLT2_POS) / 2.0

OFFSET_Y = -0.85   # target에서 -Y로 밀어내는 거리(m) - 그리퍼 수직 접근 경로에서 벗어남
CAMERA_Z = 1.20     # 기존(1.2184m)과 비슷한 높이 유지


def look_at_quat_wxyz(position, target, world_up=np.array([0.0, 0.0, 1.0])):
    """position에서 target을 바라보는 쿼터니언(wxyz). USD/OpenGL 컨벤션
    (로컬 -Z가 forward, 로컬 +Y가 up)과 일치하도록 구성한다."""
    forward = target - position
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)
    true_up = np.cross(right, forward)

    rot = np.column_stack([right, true_up, -forward])  # 로컬 X,Y,Z축을 world 기준으로

    tr = rot[0, 0] + rot[1, 1] + rot[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (rot[2, 1] - rot[1, 2]) / s
        y = (rot[0, 2] - rot[2, 0]) / s
        z = (rot[1, 0] - rot[0, 1]) / s
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2
        w = (rot[2, 1] - rot[1, 2]) / s
        x = 0.25 * s
        y = (rot[0, 1] + rot[1, 0]) / s
        z = (rot[0, 2] + rot[2, 0]) / s
    elif rot[1, 1] > rot[2, 2]:
        s = np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2
        w = (rot[0, 2] - rot[2, 0]) / s
        x = (rot[0, 1] + rot[1, 0]) / s
        y = 0.25 * s
        z = (rot[1, 2] + rot[2, 1]) / s
    else:
        s = np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2
        w = (rot[1, 0] - rot[0, 1]) / s
        x = (rot[0, 2] + rot[2, 0]) / s
        y = (rot[1, 2] + rot[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])


def main():
    stage = Usd.Stage.Open(USD_PATH)
    prim = stage.GetPrimAtPath(CAMERA_PATH)
    if not prim.IsValid():
        raise RuntimeError(f"프림 없음: {CAMERA_PATH}")

    new_pos = np.array([TARGET[0], TARGET[1] + OFFSET_Y, CAMERA_Z])
    q = look_at_quat_wxyz(new_pos, TARGET)

    tilt_deg = np.degrees(np.arctan2(
        np.linalg.norm((TARGET - new_pos)[:2]), abs((TARGET - new_pos)[2])))
    print(f"[reposition] target(볼트쌍 중간점)={TARGET}")
    print(f"[reposition] new world pos={new_pos}, 수직 대비 기울기={tilt_deg:.1f}deg")
    print(f"[reposition] quat wxyz={q}")

    xformable = UsdGeom.Xformable(prim)
    ops = {op.GetOpName(): op for op in xformable.GetOrderedXformOps()}
    if "xformOp:translate" not in ops or "xformOp:orient" not in ops:
        raise RuntimeError(
            f"{CAMERA_PATH}에 예상한 xformOp:translate/orient가 없음 - "
            f"실제 순서: {[op.GetOpName() for op in xformable.GetOrderedXformOps()]}")

    ops["xformOp:translate"].Set(Gf.Vec3d(float(new_pos[0]), float(new_pos[1]), float(new_pos[2])))
    ops["xformOp:orient"].Set(Gf.Quatd(float(q[0]), Gf.Vec3d(float(q[1]), float(q[2]), float(q[3]))))

    stage.GetRootLayer().Save()
    print(f"[저장 완료] {USD_PATH}")


if __name__ == "__main__":
    main()
