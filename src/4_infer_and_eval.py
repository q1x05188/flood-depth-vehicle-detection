"""
STEP 4. 학습된 모델로 탐지 테스트 (이미지 폴더 또는 영상 전체)
--------------------------------------------------------------
- 이미지 폴더 또는 영상 파일을 넣으면 탐지 결과가 그려진 이미지/영상이 저장됨
- test 셋에 대한 정량 평가(mAP 등)는 --eval 옵션으로 확인 가능

사용법:
    # test 이미지셋 정량 평가 (mAP50, mAP50-95 등)
    python 4_infer_and_eval.py --weights runs_cctv_vehicle/exp1/weights/best.pt --data data.yaml --eval

    # 새 영상에 대해 탐지 시각화
    python 4_infer_and_eval.py --weights runs_cctv_vehicle/exp1/weights/best.pt --source new_cctv.mp4

    # 새 이미지 폴더에 대해 탐지 시각화
    python 4_infer_and_eval.py --weights runs_cctv_vehicle/exp1/weights/best.pt --source dataset/images/test
"""

import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="학습된 best.pt 경로")
    parser.add_argument("--data", default="data.yaml")
    parser.add_argument("--source", default=None,
                         help="탐지할 영상/이미지 폴더 경로")
    parser.add_argument("--conf", type=float, default=0.25,
                         help="탐지 confidence 임계값")
    parser.add_argument("--eval", action="store_true",
                         help="test 셋으로 정량 평가(mAP 등) 실행")
    args = parser.parse_args()

    model = YOLO(args.weights)

    if args.eval:
        metrics = model.val(data=args.data, split="test")
        print("\n===== Test 셋 평가 결과 =====")
        print(f"mAP50    : {metrics.box.map50:.4f}")
        print(f"mAP50-95 : {metrics.box.map:.4f}")
        print(f"Precision: {metrics.box.mp:.4f}")
        print(f"Recall   : {metrics.box.mr:.4f}")

    if args.source:
        results = model.predict(
            source=args.source,
            conf=args.conf,
            save=True,           # 결과 이미지/영상 저장
            project="runs_cctv_vehicle",
            name="predict",
        )
        print(f"\n탐지 결과 저장 완료 -> runs_cctv_vehicle/predict/")


if __name__ == "__main__":
    main()
