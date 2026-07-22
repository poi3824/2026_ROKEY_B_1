# Bolt segmentation 학습

`detection.py`는 `results.masks`를 사용하므로 detection이 아니라 segmentation 모델을 학습합니다.

## 1. Replicator ZIP 변환

```powershell
cd C:\Users\meihi\cobot3_ws\perception_prototype
py -3.10 training\prepare_dataset.py
```

생성 결과는 `datasets/bolt_seg`에 저장됩니다. 기본 분할은 train 80%, val 20%입니다.

## 2. 학습

```powershell
py -3.10 training\train_segmentation.py --device 0
```

GPU가 없으면 `--device cpu`를 사용합니다. 기본값은 100 epochs, image size 640, batch 8입니다.

완료 모델:

```text
runs/segment/bolt_seg/weights/best.pt
```

학습 후 `detection.py`의 `MODEL_PATH`를 위 `best.pt` 경로로 교체합니다.
