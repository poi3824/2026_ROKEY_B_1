"""top-down 이미지에서 단자 홀(볼트 포스트)의 픽셀 좌표를 Hough Circle로 검출."""
import math
import os
import sys

import cv2
import numpy as np

MODULE_MIN_AREA_PX = 50000

HOUGH_PARAMS = dict(
    dp=1,
    minDist=200,
    param1=100,
    param2=25,
    minRadius=25,
    maxRadius=55,
)
CIRCULARITY_THRESHOLD = 0.6
# 뭐하는 코드임?: Hough Circle 검출에 쓰이는 설정값들 모아둔 것.
# ---
# 왜 이렇게 짬?: 숫자를 함수 안에 박아넣지 않고 위로 따로 뺀 이유는, 나중에 이 값들만 보고 바로 조정 할 수 있게 하려고.
# minRadius=25, maxRadius=55는 실제 사진에서 구멍 반지름을 측정해서 나온 값이고,
# CIRCULARITY_THRESHOLD=0.6은 진짜 구멍(원형도 0.88~0.90)이랑 가짜 사각형(0.16~0.40)을 실측해서 그 사이 값으로 잡은 것
# ---
# 흐름: 이 자체는 실행되는 코드 X 아래 함수들이 참조하는 "설정판"이다.


def _circularity(gray: np.ndarray, x: int, y: int, r: int) -> float:
    """후보 원 주변만 잘라 원형도(1에 가까울수록 원)를 계산.

    배터리 모듈 표면의 사각형 셀 무늬가 Hough 반지름 범위와 겹쳐서
    같이 검출되는데, 무늬는 원형도가 뚜렷이 낮아(0.16~0.40) 실제
    단자 홀(0.88~0.90)과 구분됨. 밝기 기준 대신 이걸 쓰는 이유는
    조명이 바뀌어도 모양 자체는 유지되어 더 안정적이기 때문.
    """
    pad = int(r * 1.5)
    h, w = gray.shape
    x0, x1 = max(0, x - pad), min(w, x + pad)
    y0, y1 = max(0, y - pad), min(h, y + pad)
    roi = gray[y0:y1, x0:x1]
    if roi.size == 0:
        return 0.0
    _, th = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    peri = cv2.arcLength(c, True)
    if peri == 0:
        return 0.0
    return 4 * np.pi * area / (peri * peri)
# 뭐하는 코드임?: 원 후보 하나가 "진짜 원처럼 생겼는지" 점수(0~1)를 계산.
# ---
# 왜 이렇게 짬?: 처음엔 "안쪽이 얼마나 어두운가"로 진짜/가짜를 구분했는데, 조명이 바뀌면 그 기준이 틀어지는 걸 실험으로 확인했다.
# 그래서 "밝기"가 아니라 "모양"으로 바꿨다.
# > 모양은 조명이 바뀌어도 그대로라 더 안정적이기 때문이다. 이름 앞에 _가 붙은 이유는 "이 파일 안에서만 쓰는 부품이고, 밖에선 직접 부르지 마세요."란 표시
# ---
# 흐름: 혼자 실행 안됨. 아래 detect_terminal_holes() 안에서 원 후보 하나하나마다 불려서 씀.


def detect_terminal_holes(image: np.ndarray) -> list[tuple[int, int, int]]:
    """BGR 이미지에서 단자 홀의 (x, y, r) 픽셀 좌표 리스트를 반환."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, **HOUGH_PARAMS)
    if circles is None:
        return []

    holes = []
    for x, y, r in np.round(circles[0, :]).astype(int):
        if _circularity(gray, x, y, r) >= CIRCULARITY_THRESHOLD:
            holes.append((int(x), int(y), int(r)))
    return holes
# 뭐하는 코드임?: 사진 한 장을 받아서, 진짜 단자 구멍들의(x,y,반지름) 리스트를 반환한다.
# ---
# 왜 이렇게 짬?: 이미지 파일 경로를 몰라도 되게, "이미지 배열"만 받도록 만들었다.
# 그래야 나중에 실제 카메라에서 받은 프레임을 그대로 넣을 수 있으니까.(파일로 저장 안 해도 됨.)
# 내부적으로 cv2.HoughCircles로 후보를 다 찾은 다음(14개) _circularity()로 걸러서(6개만 남김)반환
# ---
# 흐름: 이 파일이 가장 먼저 실행되는 진입점 역할. 사진 -> 이 함수 -> 픽셀 좌표 리스트.


def _detect_module_boxes(gray: np.ndarray) -> list[tuple[int, int, int, int]]:
    """배경(밝은 회색)과 분리된 모듈 사각형 영역 (x, y, w, h) 리스트를 반환."""
    _, th = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [
        cv2.boundingRect(c)
        for c in contours
        if cv2.contourArea(c) > MODULE_MIN_AREA_PX
    ]
    boxes.sort(key=lambda b: b[0])
    return boxes
# 뭐하는 코드임?: 배경(밝은 회색)과 모듈(어두운 부분)을 밝기로 나눠서, 모듈 3개의 사각형 영역(x,y,w,h)을 찾는다.
# ---
# 왜 이렇게 짬?: 구멍끼리 짝지을 때 "제일 가까운 것끼리"로 하면 옆 모듈이랑 헷갈린다는 걸 실측으로 확인.(같은 모듈 대각선 거리 905~913px > 옆 모듈 같은 줄거리 650~656px, 오히려 더 가까움)
# 그래서 거리 대신 "어느 모듈 사각형 안에 있나"로 방식을 바꿈.
# ---
# 흐름: 아래 pair_terminal_holes() 안에서만 쓰이는 부품 함수


def pair_terminal_holes(image: np.ndarray, holes: list[tuple[int, int, int]]) -> list[dict]:
    """구멍들을 모듈 사각형 단위로 묶어서 (중간 좌표, 각도)를 계산.

    거리 기준으로 짝짓지 않는 이유: 같은 모듈의 대각선 두 구멍 사이 거리가
    오히려 옆 모듈의 같은 줄 구멍끼리 거리보다 멀어서, 거리만으로는 옆
    모듈 것과 헷갈릴 수 있음 (실측 확인함). 대신 모듈 사각형 영역 자체를
    찾아서 그 안에 있는 구멍끼리만 묶으면 이 문제가 생기지 않음.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    boxes = _detect_module_boxes(gray)

    pairs = []
    for bx, by, bw, bh in boxes:
        in_box = [
            (x, y, r)
            for (x, y, r) in holes
            if bx <= x <= bx + bw and by <= y <= by + bh
        ]
        if len(in_box) != 2:
            continue  # 이 모듈 안에 구멍이 정확히 2개가 아니면 건너뜀 (오검출 가능성)

        # 항상 "위쪽 구멍 -> 아래쪽 구멍" 방향으로 각도를 재도록 y좌표로 정렬.
        # 정렬 안 하면 검출 순서에 따라 쌍마다 방향이 반대로(180도 차이) 나올 수 있음.
        in_box = sorted(in_box, key=lambda h: h[1])
        (x1, y1, _), (x2, y2, _) = in_box
        mid = ((x1 + x2) / 2, (y1 + y2) / 2)
        angle_deg = math.degrees(math.atan2(y2 - y1, x2 - x1))
        pairs.append({"holes": in_box, "mid": mid, "angle_deg": angle_deg})

    return pairs
# 뭐하는 코드임?: 구멍 6개를 모듈별로 2개씩 묶어서, 각 쌍의 중간 좌표 + 각도 계산.
# ---
# 왜 이렇게 짬?:
# - 모듈 사각형 안에 있는 것끼리만 묶어서 정확하게 3쌍이 나오게 함.
# - 각도를 구할 때 두 구멍을 y좌표로 정렬해서 항상 "위 -> 아래" 방향으로 재도록함.
#   > 처음엔 정렬을 안 해서 어떤 쌍은 60.1도, 어떤 쌍은 -119.8도로 나오는 버그가 있었고, 실행해서 직접 확인한 뒤 고침.
# - 왜 각도까지 필요하냐면, 버스바가 딱딱한 2구멍짜리 부품이라 "어디에"뿐 아니라 "어느 각도로" 놓을지도 로봇이 알아야 하기 때문
# ---
# 흐름: detect_terminal_holes()가 끝난 다음에 그 결과(6개 좌표)를 받아서 실행됨. 이 함수 결과(3쌍)가 다음 단계인 depth_estimation.py로 넘어감

def draw_debug(image: np.ndarray, holes: list[tuple[int, int, int]]) -> np.ndarray:
    """검출된 홀을 원본 위에 초록 원 + 빨간 중심점으로 그려서 반환."""
    debug = image.copy()
    for x, y, r in holes:
        cv2.circle(debug, (x, y), r, (0, 255, 0), 3)
        cv2.circle(debug, (x, y), 4, (0, 0, 255), -1)
    return debug


DEFAULT_IMG_PATH = r"C:\Users\meihi\Downloads\배터리 모듈팩.png"
# 뭐하는 코드임?: 검출된 구멍 위치에 초록 원 + 빨간 점을 그려서 눈으로 확인할 수 있는 이미지를 만듦.
# ---
# 왜 이렇게 짬?: 검출 로직(detect_terminal_holes)이랑 시각화를 분리함. 로봇 노드에서 실제로 쓸 땐 이 함수는 필요 없고(좌표만 쓰면됨.)
# 사람이 눈으로 확인 할 때만 필요하니 따로 빼버림.
# ---
# 흐름: main()에서만 불림. 실제 노드에서 안씀.


def main():
    IMG_PATH = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMG_PATH
    OUT_PATH = os.path.join(os.path.dirname(__file__), "debug_output.png")

    # imread는 한글 경로를 못 읽어서 imdecode로 우회
    img = cv2.imdecode(np.fromfile(IMG_PATH, dtype=np.uint8), cv2.IMREAD_COLOR)
    holes = detect_terminal_holes(img)

    print(f"detected {len(holes)} circles")
    for h in holes:
        print(h)

    pairs = pair_terminal_holes(img, holes)
    print(f"paired into {len(pairs)} sets")
    for p in pairs:
        print(f"  holes={p['holes']} mid={p['mid']} angle_deg={p['angle_deg']:.1f}")

    cv2.imwrite(OUT_PATH, draw_debug(img, holes))


if __name__ == "__main__":
    main()
