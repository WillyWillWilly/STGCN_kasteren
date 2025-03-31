# STGCN for Kasteren Dataset

這是一個基於時空圖卷積網絡（STGCN）的活動預測模型，專門用於處理 Kasteren 數據集。

## 項目結構

```
STGCN_Kasteren/
├── config.py              # 配置文件
├── main.py               # 主程序入口
├── models/               # 模型相關代碼
│   ├── base_model.py     # 基礎模型類
│   ├── layers.py         # 模型層定義
│   ├── trainer.py        # 訓練邏輯
│   └── tester.py         # 測試邏輯
├── data_loader/          # 數據加載相關代碼
│   ├── data_utils.py     # 數據處理工具
│   └── data_processor.py # 數據預處理
├── utils/                # 工具函數
│   ├── logger.py         # 日誌系統
│   ├── math_graph.py     # 圖計算工具
│   └── math_utils.py     # 數學工具
├── dataset/              # 數據集目錄
│   └── base_kasteren-m.csv
├── output/               # 輸出目錄
│   └── checkpoints/      # 模型檢查點
└── logs/                 # 日誌目錄
```

## 環境要求

- Python 3.7+
- TensorFlow 1.15.0
- NumPy >= 1.19.2
- Pandas >= 1.2.0
- SciPy >= 1.6.0
- scikit-learn >= 0.24.0

## 安裝

1. 克隆倉庫：
```bash
git clone [repository_url]
cd STGCN_Kasteren
```

2. 創建虛擬環境：
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. 安裝依賴：
```bash
pip install -r requirements.txt
```

## 使用方法

1. 準備數據：
   - 將 Kasteren 數據集放在 `dataset` 目錄下
   - 確保數據格式符合要求

2. 訓練模型：
```bash
python main.py
```

3. 測試模型：
```bash
python test.py
```

## 配置說明

主要配置參數在 `config.py` 中：

- `DATASET_CONFIG`: 數據集相關配置
- `MODEL_CONFIG`: 模型架構配置
- `TRAIN_CONFIG`: 訓練相關配置
- `LOG_CONFIG`: 日誌配置
- `GPU_CONFIG`: GPU 相關配置

## 數據格式

Kasteren 數據集格式：
- CSV 文件
- 每行包含時間戳和 14 個傳感器的狀態
- 時間間隔為 5 分鐘

## 模型架構

- 時空圖卷積網絡（STGCN）
- 包含空間和時間卷積層
- 使用切比雪夫多項式進行圖卷積

## 性能指標

- MAE (Mean Absolute Error)
- RMSE (Root Mean Square Error)
- MAPE (Mean Absolute Percentage Error)

## 注意事項

1. GPU 使用：
   - 確保 CUDA 和 cuDNN 已正確安裝
   - 可以通過 `config.py` 中的 `GPU_CONFIG` 調整 GPU 設置

2. 內存使用：
   - 注意調整 batch_size 以適應可用內存
   - 可以通過 `TRAIN_CONFIG` 調整相關參數

3. 模型保存：
   - 模型檢查點保存在 `output/checkpoints` 目錄
   - 可以通過 `CHECKPOINT_CONFIG` 調整保存策略

## 貢獻指南

1. Fork 項目
2. 創建特性分支
3. 提交更改
4. 推送到分支
5. 創建 Pull Request

## 許可證

MIT License 