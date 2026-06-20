"""
YOLOv8 物件偵測訓練腳本 — 金屬零件辨識與計數

對應 MATLAB 階段的 metal.m（Faster R-CNN），此處改用 YOLOv8 重新實作
物件偵測 + 計數的目標。

使用前置作業：
1. 蒐集並標註資料集（YOLO 格式：每張圖一個對應 .txt，內容為
   `class_id x_center y_center width height`，座標皆為 0~1 正規化）
2. 將資料集放置於 python/data/yolo_dataset/，結構需符合
   configs/yolo_dataset.yaml 的設定
3. 確認 configs/yolo_dataset.yaml 中的類別與實際資料一致

執行：
    python train_yolo.py
"""

import yaml
from pathlib import Path
from ultralytics import YOLO

CONFIG_DIR = Path(__file__).parent.parent / "configs"


def load_config(name: str) -> dict:
    with open(CONFIG_DIR / name, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    train_cfg = load_config("train_config.yaml")
    dataset_yaml = CONFIG_DIR / "yolo_dataset.yaml"

    if not dataset_yaml.exists():
        raise FileNotFoundError(
            f"找不到資料集設定檔: {dataset_yaml}\n"
            "請先完成資料蒐集與標註，並確認 configs/yolo_dataset.yaml 路徑正確。"
        )

    model = YOLO(train_cfg["model"])

    model.train(
        data=str(dataset_yaml),
        epochs=train_cfg["epochs"],
        batch=train_cfg["batch"],
        imgsz=train_cfg["imgsz"],
        optimizer=train_cfg["optimizer"],
        lr0=train_cfg["lr0"],
        patience=train_cfg["patience"],
        device=train_cfg["device"],
        project="runs/detect",
        name="metal_parts_yolov8",
    )

    # 在驗證集上評估，輸出 mAP / precision / recall
    metrics = model.val()
    print(metrics)


if __name__ == "__main__":
    main()
