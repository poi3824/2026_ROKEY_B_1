import cv2
import numpy as np
import os

IMG_PATH = r"C:\Users\meihi\Downloads\배터리 모듈팩.png"
OUT_DIR = os.path.join(os.path.dirname(__file__), "variants")
os.makedirs(OUT_DIR, exist_ok=True)

img = cv2.imdecode(np.fromfile(IMG_PATH, dtype=np.uint8), cv2.IMREAD_COLOR)
h, w = img.shape[:2]


def make_brighter(im):
    return cv2.convertScaleAbs(im, alpha=1.0, beta=40)


def make_darker(im):
    return cv2.convertScaleAbs(im, alpha=1.0, beta=-40)


def make_rotated(im):
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 5, 1.0)
    return cv2.warpAffine(im, M, (w, h), borderValue=(200, 200, 200))


def make_noisy(im):
    noise = np.random.normal(0, 15, im.shape).astype(np.int16)
    out = im.astype(np.int16) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def make_low_contrast(im):
    return cv2.convertScaleAbs(im, alpha=0.6, beta=40)


variants = {
    "brighter": make_brighter(img),
    "darker": make_darker(img),
    "rotated_5deg": make_rotated(img),
    "noisy": make_noisy(img),
    "low_contrast": make_low_contrast(img),
}


CIRCULARITY_THRESHOLD = 0.6


def circularity(gray, x, y, r):
    # 후보 주변만 잘라서 그 안에서만 밝기 기준(Otsu)을 다시 잡음 ->
    # 사진 전체 밝기가 달라져도 이 지역 대비만 보므로 영향을 덜 받음
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


def detect(im):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=200,
        param1=100, param2=25, minRadius=25, maxRadius=55,
    )
    kept = []
    if circles is not None:
        for (x, y, r) in np.round(circles[0, :]).astype(int):
            if circularity(gray, x, y, r) >= CIRCULARITY_THRESHOLD:
                kept.append((int(x), int(y), int(r)))
    return kept


for name, variant_img in variants.items():
    path = os.path.join(OUT_DIR, f"{name}.png")
    cv2.imwrite(path, variant_img)
    kept = detect(variant_img)
    debug = variant_img.copy()
    for (x, y, r) in kept:
        cv2.circle(debug, (x, y), r, (0, 255, 0), 3)
        cv2.circle(debug, (x, y), 4, (0, 0, 255), -1)
    cv2.imwrite(os.path.join(OUT_DIR, f"{name}_debug.png"), debug)
    status = "OK" if len(kept) == 6 else "MISMATCH"
    print(f"[{status}] {name}: detected {len(kept)} / expected 6")
    for c in kept:
        print("   ", c)
