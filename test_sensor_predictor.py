import pandas as pd
from data_loader.data_processor import DataProcessor
from sensor_predictor import SensorPredictor

def main():
    # 載入資料
    processor = DataProcessor('dataset/base_kasteren-m.csv')
    df = processor.load_data()
    
    # 初始化預測器
    predictor = SensorPredictor(sequence_length=10)
    
    # 準備訓練資料
    X, y = predictor.prepare_data(df)
    
    # 訓練模型
    print("開始訓練模型...")
    history = predictor.train(X, y, epochs=50, batch_size=32)
    
    # 測試預測
    print("\n測試預測功能...")
    # 使用最後10個感測器作為輸入序列
    last_sequence = df['sensor'].map(predictor.sensor_mapping).values[-10:].tolist()
    
    # 預測下一個可能出現的感測器
    predictions = predictor.predict_next_sensors(last_sequence, top_k=3)
    
    print("\n預測結果:")
    for sensor, probability in predictions:
        print(f"感測器: {sensor}, 機率: {probability:.4f}")

if __name__ == "__main__":
    main() 