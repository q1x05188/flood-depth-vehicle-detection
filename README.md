# CCTV 침수 상황 차량 탐지 가능성 검증 (서브 프로젝트)

## 프로젝트 개요

본 프로젝트는 **CCTV 이미지 기반 침수심(침수 깊이) 추정 프로젝트**의 선행 서브 프로젝트입니다.

침수심을 추정하려면 CCTV 영상 속 차량을 먼저 인식하고, 차량이 물에 잠긴 정도를 기준으로 삼아야 합니다.
이 서브 프로젝트의 목적은 **"주어진 CCTV 영상에서 YOLO 기반 Detection 모델로 차량이 실제로 잘 인식되는가"**를
직접 구현/실험하여 확인하는 것입니다.

- 확인하려는 것: 모델 자체의 성능/구현 가능성 (단순 데이터 품질 판별이 아님)
- 클래스: 차종 구분 없이 **단일 클래스 `vehicle`** (차량 존재 유무만 판단하면 되는 과제 범위)
- 특히 확인할 부분: **홍수(flood) 상황과 비홍수(normal) 상황에서 탐지 성능 차이**
  (침수 상황에서 차량 일부가 물에 잠겨 인식이 더 어려울 것으로 예상됨)

## 데이터 특징

- 원본은 CCTV 4분할 화면 영상 (한 화면에 카메라 4대가 2x2로 나뉘어 표시됨)
- 5초 간격으로 캡처 후, 4분할된 각 구역을 개별 이미지로 분리해서 사용
- 홍수 상황 영상과 비홍수 상황 영상이 별도로 존재 → 파일명에 `flood_` / `normal_` 태그를 붙여 구분

## 전체 파이프라인

```
[영상: flood/, normal/ 폴더]
        │  1b_split_quad_and_extract.py  (4분할 분리 + 5초 간격 캡처, --tag로 flood/normal 표시)
        ▼
dataset/raw_frames/  (flood_xxx.jpg, normal_xxx.jpg)
        │  [라벨링] LabelImg / Roboflow 등으로 차량에 박스(vehicle 단일 클래스)
        │  0_auto_label.py 로 반자동 라벨링 가능 (초벌 모델로 자동 박스 생성 → 사람은 수정만)
        ▼
dataset/labeled/  (이미지 + YOLO 포맷 라벨(.txt))
        │  2_split_dataset.py  (홍수/비홍수 비율 유지하며 train/val/test 분할)
        ▼
dataset/images, dataset/labels  (train / val / test / test_flood / test_normal)
        │  3_train.py  (YOLOv8n 파인튜닝 학습)
        ▼
runs_cctv_vehicle/exp1/weights/best.pt
        │  4_infer_and_eval.py --eval   (전체 test셋 mAP/precision/recall)
        │  5_eval_by_group.py           (홍수 vs 비홍수 성능 비교 - 핵심 결과)
        ▼
최종 결과: "이 CCTV 데이터에서 차량 탐지가 어느 정도 가능한가" + "침수 상황이 탐지 성능에 미치는 영향"
```

## 폴더 구조

```
cctv-vehicle-detection/
├── README.md
├── data.yaml                 # YOLO 데이터셋 설정 (클래스: vehicle 1개)
├── src/                       # 파이프라인 스크립트
│   ├── 1b_split_quad_and_extract.py   # 4분할 영상 -> 프레임 추출
│   ├── check_quad_alignment.py         # 4분할 경계 좌표 확인용
│   ├── 0_auto_label.py                 # 반자동(모델 보조) 라벨링
│   ├── 2_split_dataset.py              # train/val/test 분할 (홍수/비홍수 비율 유지)
│   ├── 3_train.py                      # YOLO 학습
│   ├── 4_infer_and_eval.py             # 추론 + 정량 평가
│   └── 5_eval_by_group.py              # 홍수 vs 비홍수 성능 비교
├── dataset/
│   ├── videos/
│   │   ├── flood/            # 원본 홍수 상황 영상 (각자 파일 추가)
│   │   └── normal/           # 원본 비홍수 상황 영상
│   ├── raw_frames/           # 1b 스크립트로 추출된 프레임 (라벨링 전)
│   ├── labeled/              # 라벨링 완료된 이미지+txt (2_split_dataset.py 입력)
│   ├── images/{train,val,test,test_flood,test_normal}/
│   └── labels/{train,val,test,test_flood,test_normal}/
└── runs_cctv_vehicle/         # 학습 결과물 (모델 가중치, 로그, 평가 그래프) 저장 위치
```

현재는 **빈 폴더 구조만 세팅된 상태**입니다. 각자 담당 영상/이미지를 해당 폴더에 채워 넣으면서 진행하시면 됩니다.

## 시작하는 방법 (Quick Start)

```bash
# 0. 필요 패키지 설치
pip install ultralytics opencv-python labelImg pyyaml

# 1. 원본 영상을 dataset/videos/flood, dataset/videos/normal 에 넣기

# 2. 4분할 경계 확인 (영상 1개로 먼저 테스트)
python src/check_quad_alignment.py --video dataset/videos/flood/샘플영상.mp4 --time 5

# 3. 프레임 추출 (경계 좌표는 2번 결과에 맞게 조정)
python src/1b_split_quad_and_extract.py --video_dir dataset/videos/flood  --out dataset/raw_frames --interval 5 --tag flood
python src/1b_split_quad_and_extract.py --video_dir dataset/videos/normal --out dataset/raw_frames --interval 5 --tag normal

# 4. 라벨링 (LabelImg 등, vehicle 단일 클래스)
labelImg dataset/raw_frames

# 5. train/val/test 분할
python src/2_split_dataset.py --src dataset/raw_frames --dst dataset

# 6. 학습
python src/3_train.py --data data.yaml --model yolov8n.pt --epochs 50 --imgsz 640 --batch 8

# 7. 평가
python src/4_infer_and_eval.py --weights runs_cctv_vehicle/exp1/weights/best.pt --data data.yaml --eval
python src/5_eval_by_group.py --weights runs_cctv_vehicle/exp1/weights/best.pt --data_root dataset
```

## 라벨링 규칙 (작업자 필독)

- 클래스는 차종 구분 없이 **`vehicle` 하나만** 사용
- 차량 여러 대가 붙어있어도 **차량 한 대당 박스 하나** (절대 여러 대를 한 박스로 묶지 않음)
- 일부만 가려진 차량은 보이는 부분 기준으로 박스 (완전히 형체를 알아볼 수 없으면 라벨링 스킵)
- 라벨링 파일명 규칙: 원본 프레임 파일명에 이미 `flood_` / `normal_` 태그가 붙어 있으므로
  별도 작업 없이 그대로 저장하면 됨 (2_split_dataset.py가 이 태그로 자동 구분)

## 담당 영역 (예시 - 실제 배정에 맞게 수정)

| 담당자 | 역할 |
|---|---|
| (이름) | 영상 수집 및 4분할 프레임 추출 |
| (이름) | 라벨링 (flood 그룹) |
| (이름) | 라벨링 (normal 그룹) |
| (이름) | 모델 학습 및 평가 |

## 참고

- 목적이 "구현 가능성 확인"이므로, 완벽한 성능보다 **파이프라인이 끝까지 동작하고 결과 수치가 나오는 것**을 1차 목표로 합니다.
- 데이터 양이 적어 성능이 낮게 나와도 괜찮으며, 한계점과 개선 방향을 함께 정리하면 됩니다.
