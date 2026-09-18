"""
STEP 5. 홍수(flood) vs 비홍수(normal) 상황에서의 탐지 성능 비교
--------------------------------------------------------------------
2_split_dataset.py 로 만들어진 dataset/images/test_flood, test_normal 폴더를 이용해
같은 모델을 두 그룹에 각각 평가하고 mAP를 나란히 비교합니다.

사용법:
    python 5_eval_by_group.py --weights runs_cctv_vehicle/exp1/weights/best.pt --data_root dataset
"""

import argparse
import tempfile
from pathlib import Path
import yaml
from ultralytics import YOLO


def build_temp_yaml(data_root, group, class_names):
    """해당 그룹의 test 폴더만을 가리키는 임시 data.yaml 생성"""
    cfg = {
        "path": str(Path(data_root).resolve()),
        "train": f"images/test_{group}",   # 사용 안 하지만 필드가 있어야 함
        "val": f"images/test_{group}",
        "test": f"images/test_{group}",
        "names": class_names,
    }
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=f"_{group}.yaml",
                                       delete=False, encoding="utf-8")
    yaml.dump(cfg, tmp, allow_unicode=True)
    tmp.close()
    return tmp.name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data_root", default="dataset",
                         help="images/test_flood, images/test_normal 이 들어있는 최상위 폴더")
    parser.add_argument("--class_names", nargs="+", default=["vehicle"],
                         help="클래스 이름 목록 (data.yaml의 names 순서와 동일하게)")
    args = parser.parse_args()

    class_names = {i: n for i, n in enumerate(args.class_names)}
    model = YOLO(args.weights)

    results = {}
    for group in ["flood", "normal"]:
        test_dir = Path(args.data_root) / "images" / f"test_{group}"
        if not test_dir.exists() or not any(test_dir.iterdir()):
            print(f"[건너뜀] {test_dir} 가 비어있거나 없습니다.")
            continue

        tmp_yaml = build_temp_yaml(args.data_root, group, class_names)
        print(f"\n===== {group.upper()} 그룹 평가 중 =====")
        metrics = model.val(data=tmp_yaml, split="test")

        results[group] = {
            "mAP50": metrics.box.map50,
            "mAP50-95": metrics.box.map,
            "Precision": metrics.box.mp,
            "Recall": metrics.box.mr,
        }

    if not results:
        print("평가할 그룹 데이터가 없습니다. 2_split_dataset.py를 먼저 실행했는지 확인하세요.")
        return

    print("\n\n================ 홍수 vs 비홍수 성능 비교 ================")
    print(f"{'지표':<12}{'Flood':>12}{'Normal':>12}{'차이(Normal-Flood)':>22}")
    for metric in ["mAP50", "mAP50-95", "Precision", "Recall"]:
        f_val = results.get("flood", {}).get(metric)
        n_val = results.get("normal", {}).get(metric)
        if f_val is not None and n_val is not None:
            diff = n_val - f_val
            print(f"{metric:<12}{f_val:>12.4f}{n_val:>12.4f}{diff:>22.4f}")
        else:
            print(f"{metric:<12} (한쪽 그룹 데이터 없음, 비교 불가)")

    print("\n(mAP50 기준으로 Flood가 Normal보다 얼마나 낮은지가 "
          "'침수 상황에서 탐지가 얼마나 어려워지는지'를 보여주는 핵심 지표입니다)")


if __name__ == "__main__":
    main()
