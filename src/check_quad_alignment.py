"""
보조 스크립트: 4분할 경계가 정확히 화면 절반인지 확인용
--------------------------------------------------------------
영상에서 프레임 한 장을 뽑아, 절반 기준으로 빨간 십자선을 그어서 저장합니다.
이 선이 실제 화면 구분선과 딱 맞으면 -> 기본 스크립트(h//2, w//2) 그대로 써도 됨
선이 안 맞고 어긋나 있으면 -> 여백/베젤이 있는 것이므로 좌표를 직접 지정해야 함

사용법:
    python check_quad_alignment.py --video videos/cam01.mp4 --time 5
"""

import cv2
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--time", type=float, default=5.0, help="몇 초 지점 프레임을 볼지")
    parser.add_argument("--out", default="check_alignment.jpg")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(args.time * fps))
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("프레임을 읽지 못했습니다.")
        return

    h, w = frame.shape[:2]
    print(f"영상 해상도: {w} x {h}")
    print(f"절반 기준 좌표: half_w={w//2}, half_h={h//2}")

    # 절반 위치에 빨간 십자선 표시
    vis = frame.copy()
    cv2.line(vis, (w // 2, 0), (w // 2, h), (0, 0, 255), 2)
    cv2.line(vis, (0, h // 2), (w, h // 2), (0, 0, 255), 2)

    cv2.imwrite(args.out, vis)
    print(f"확인용 이미지 저장됨 -> {args.out}")
    print("빨간 선이 실제 4분할 경계선과 딱 맞는지 눈으로 확인하세요.")


if __name__ == "__main__":
    main()
