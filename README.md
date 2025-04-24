# Mini-ImageNet 動態卷積網絡 - 任務 A

本專案實現了一個能夠處理可變輸入通道的動態卷積模型，以滿足 MILS 作業 I 中的任務 A 要求。我們設計了一個特殊的卷積模組，該模組與空間大小無關，且能處理任意數量的輸入通道（如 RGB、RG、GB、R、G、B 等組合）。

本方案參考了 Wu 等人在 CVPR 2020 發表的「Dynamic Convolution: Attention over Convolution Kernels」論文，但進行了簡化和優化以提高計算效率。

## 專案結構

```
Task_A/
├── models/
│   ├── standard_net.py      # 標準卷積網絡模型(基準)
│   ├── dynamic_net.py       # 動態卷積網絡模型
│   └── dynamic_conv.py      # 動態卷積模組實現
├── utils/
│   ├── dataset.py           # 資料集加載功能(包含通道變換)
│   ├── train.py             # 訓練函數
│   ├── evaluate.py          # 評估函數
│   ├── visualization.py     # 可視化工具
│   └── compare_models.py    # 模型比較工具
├── experiments/
│   ├── standard/            # 標準模型實驗結果
│   ├── dynamic/             # 動態模型實驗結果
│   ├── comparison/          # 模型比較結果
│   └── analysis/            # 分析和可視化結果
├── README.md                # 專案說明
├── main.py                  # 主執行檔
└── test_dynamic_model.py    # 測試和分析腳本
```

### 1. 訓練標準模型（基準模型）

```bash
python main.py --mode train --model standard --batch_size 64 --epochs 20
```

### 2. 訓練動態卷積模型

```bash
python main.py --mode train --model dynamic --batch_size 64 --epochs 20
```

### 3. 評估模型（使用不同通道組合）

```bash
# 評估 RGB 完整通道
python main.py --mode eval --model dynamic --channel_mode rgb --checkpoint experiments/dynamic/best_model.pth
```