# `metal.m` 程式說明

## 方法
Faster R-CNN 物件偵測（base network: **AlexNet**），使用 MATLAB
`trainFasterRCNNObjectDetector`。

## 類別（6 類）
- Bolt（螺栓）
- Foundation bolt（地腳螺栓）
- Gasket（墊片）
- Drilling screw（自攻螺絲）
- Nut（螺帽）
- Screw（螺絲）

## 訓練設定
- Optimizer: Adam
- MaxEpochs: 1（註：原始程式僅設為 1，可能是除錯/測試用的暫定值，
  正式訓練的 epoch 數已不可考）
- MiniBatchSize: 75
- InitialLearnRate: 0.0001

## 評估方式
以 `bboxOverlapRatio` 計算偵測框與標註框（ground truth）的重疊率（IoU），
作為辨識準確度的量化指標。

## 已知缺失資料
- 訓練 / 測試影像資料集（原 ppt 提及約 150 張）
- 標註檔 `train_label.mat`、`test_label.mat`（內含 `gTruth` 物件，
  記錄每張圖中各類別的 bounding box 標註）

以上資料夾結構原為：
```
train/
  train_label.mat
  <各類別子資料夾>/
test/
  test_label.mat
  <各類別子資料夾>/
```

## 與原始提案（ppt）的差異
原提案規劃使用 **GoogLeNet 做影像分類**，但實際留存的這份程式碼採用
**Faster R-CNN 做物件偵測**（可同時定位與計數），推測是方法迭代後的版本。
