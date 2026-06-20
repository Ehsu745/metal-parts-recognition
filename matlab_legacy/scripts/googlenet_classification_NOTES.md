# `googlenet_classification.m` 程式說明

## 方法
GoogLeNet 遷移學習（transfer learning）影像分類，使用 MATLAB
`trainNetwork`，網路結構來自預先儲存的 `googLeNet_4.mat`（**已遺失**，
內含預訓練好的 layer graph `lgraph_1`）。

## 訓練設定
- Optimizer: Adam
- InitialLearnRate: 0.0001
- MaxEpochs: 50
- MiniBatchSize: 1（極小，可能是受限於資料量或顯卡記憶體）
- 訓練/驗證切分：60% / 40%（`splitEachLabel(allData, 0.6, 'randomized')`）
- 執行環境：`multi-gpu`

## 評估方式
- Accuracy（整體準確率）
- Confusion matrix
- Precision / Recall / F1-score（逐類別計算）

## ⚠️ 待確認事項
此程式讀取的資料來源資料夾為 `new`，**目前無法確認其類別是否為金屬零件
（Bolt/Nut/Screw 等）**，也可能屬於同學期另一份練習。檔案來源已遺失，
無法回溯驗證。在 Python 重構階段會重新蒐集明確標註的金屬零件資料集，
不會延用此處的不確定性。

## 已知缺失資料
- `googLeNet_4.mat`（預訓練 layer graph）
- `new/` 資料夾（訓練影像，依 label 分子資料夾）
