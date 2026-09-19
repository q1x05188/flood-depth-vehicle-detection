
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
