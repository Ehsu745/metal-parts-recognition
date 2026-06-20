"""
即時辨識 Demo — 金屬零件辨識與計數（YOLOv8）

呼應原始提案「輸送帶上即時辨識」的願景，使用 webcam 串流模擬即時場景。
需先完成 train_yolo.py 的訓練，產出權重檔。

執行：
    python live_demo.py --weights runs/detect/metal_parts_yolov8/weights/best.pt
"""

import argparse
from collections import Counter

import cv2
from ultralytics import YOLO


def main(weights: str, camera_index: int = 0, conf_threshold: float = 0.5):
    model = YOLO(weights)
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError(f"無法開啟攝影機（index={camera_index}）")

    print("按下 'q' 離開")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=conf_threshold, verbose=False)
        result = results[0]

        annotated = result.plot()

        # 計數：依類別統計這個畫面中偵測到的數量
        names = result.names
        counts = Counter(names[int(cls_id)] for cls_id in result.boxes.cls)
        count_text = " | ".join(f"{k}: {v}" for k, v in counts.items()) or "no detections"

        cv2.putText(
            annotated,
            count_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Metal Parts Recognition - Live Demo", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="金屬零件即時辨識 Demo")
    parser.add_argument("--weights", type=str, required=True, help="訓練好的 YOLOv8 權重檔路徑")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.5, dest="conf_threshold")
    args = parser.parse_args()

    main(args.weights, args.camera_index, args.conf_threshold)
