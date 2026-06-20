# 資料集

> **更新**：原以為遺失的 MATLAB 階段資料集已在舊電腦上尋回（`toy/` 資料夾），
> 並重新轉換為 YOLO 格式標註，現已納入本 repo。

## 內容

### `yolo_dataset/`（YOLOv8 物件偵測用）

```
yolo_dataset/
├── images/
│   ├── train/    135 張
│   └── test/     15 張
└── labels/
    ├── train/    135 個 .txt（YOLO 格式標註）
    └── test/     15 個 .txt
```

每張圖片對應一個同名 `.txt` 標註檔，內容為：
```
<class_id> <x_center> <y_center> <width> <height>
```
（座標皆為相對影像寬高 768x480 的 0~1 正規化數值）

### 類別對照

| class_id | 類別 |
|---|---|
| 0 | bolt |
| 1 | foundationbolt |
| 2 | gasket |
| 3 | drillingscrew |
| 4 | nut |
| 5 | screw |

## 資料來源與轉換流程

1. 原始圖片與標註由 MATLAB Image Labeler 產生（`groundTruth` 物件，儲存於
   `train_label.mat` / `test_label.mat`），對應 `matlab_legacy/scripts/metal.m`
   的訓練流程。
2. 因 `.mat` 內的 `groundTruth` 為 MATLAB 專屬的 MCOS 序列化物件，無法用
   Python 直接解析，故透過 MATLAB Online 執行轉換腳本，將標註展開匯出為
   CSV（`filename, class, x, y, width, height`）。
3. 再以 Python 腳本將 CSV 轉換為 YOLO 格式（中心點座標、0~1 正規化），
   並以實際疊框視覺化驗證轉換正確性。

## CNN 分類用資料集（`cls_dataset/`）

`train_classifier.py`（GoogLeNet 分類對照組）所需的 ImageFolder 格式資料集
**尚未建立**。原始 MATLAB 階段的 `googlenet_classification.m` 對應的訓練資料
（`new/` 資料夾、`googLeNet_4.mat`）已遺失且類別內容無法確認，故分類對照組
需要另外規劃資料蒐集或從現有 150 張圖重新切分標註。
