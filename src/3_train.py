
import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data.yaml")
    parser.add_argument("--model", default="yolov8n.pt",
                         help="yolov8n.pt(가장 가벼움) / yolov8s.pt / yolo11n.pt 등")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0",
                         help="GPU 사용시 '0', CPU만 있으면 'cpu'")
    parser.add_argument("--patience", type=int, default=15,
                         help="이만큼 epoch 동안 성능 개선 없으면 조기 종료")
    args = parser.parse_args()

    model = YOLO(args.model)  # 사전학습 가중치에서 시작 (transfer learning, 적은 데이터에 유리)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=args.patience,
        workers=4,
        project="runs_cctv_vehicle",
        name="exp1",
        # 가정용 GPU 부담을 줄이기 위한 옵션들
        cache=False,        # 메모리 부족하면 False, RAM 넉넉하면 True로 속도 향상 가능
        amp=True,           # mixed precision -> VRAM 절약 + 속도 향상
    )

    # 학습 끝나면 val 셋으로 자동 평가된 결과가 runs_cctv_vehicle/exp1 에 저장됨
    # (mAP50, mAP50-95, precision, recall 등 확인 가능)


if __name__ == "__main__":
    main()
