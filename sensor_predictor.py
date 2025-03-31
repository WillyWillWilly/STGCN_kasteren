import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

class SensorPredictor:
    def __init__(self, sequence_length=10):
        """初始化感測器預測器
        
        Args:
            sequence_length (int): 用於預測的歷史序列長度
        """
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = StandardScaler()
        self.sensor_mapping = {}
        self.reverse_sensor_mapping = {}
        
    def prepare_data(self, df):
        """準備訓練資料
        
        Args:
            df (pd.DataFrame): 包含感測器資料的DataFrame
        """
        # 建立感測器映射
        unique_sensors = sorted(df['sensor'].unique())
        self.sensor_mapping = {sensor: idx for idx, sensor in enumerate(unique_sensors)}
        self.reverse_sensor_mapping = {idx: sensor for sensor, idx in self.sensor_mapping.items()}
        
        # 將感測器資料轉換為數值
        sensor_data = df['sensor'].map(self.sensor_mapping).values
        
        # 標準化資料
        sensor_data_scaled = self.scaler.fit_transform(sensor_data.reshape(-1, 1))
        
        # 建立序列資料
        X, y = [], []
        for i in range(len(sensor_data_scaled) - self.sequence_length):
            X.append(sensor_data_scaled[i:(i + self.sequence_length)])
            y.append(sensor_data_scaled[i + self.sequence_length])
            
        return np.array(X), np.array(y)
    
    def build_model(self):
        """建立LSTM模型"""
        self.model = Sequential([
            LSTM(64, input_shape=(self.sequence_length, 1), return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
    
    def train(self, X, y, epochs=50, batch_size=32):
        """訓練模型
        
        Args:
            X (np.array): 輸入序列
            y (np.array): 目標值
            epochs (int): 訓練輪數
            batch_size (int): 批次大小
        """
        if self.model is None:
            self.build_model()
            
        # 分割訓練和驗證資料
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 訓練模型
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            verbose=1
        )
        
        return history
    
    def predict_next_sensors(self, current_sequence, top_k=3):
        """預測下一個可能出現的感測器
        
        Args:
            current_sequence (list): 目前的感測器序列
            top_k (int): 返回前k個最可能的感測器
            
        Returns:
            list: 預測的感測器列表及其機率
        """
        if len(current_sequence) < self.sequence_length:
            raise ValueError(f"序列長度必須至少為 {self.sequence_length}")
            
        # 轉換序列為模型輸入格式
        sequence = current_sequence[-self.sequence_length:]
        sequence_scaled = self.scaler.transform(np.array(sequence).reshape(-1, 1))
        sequence_reshaped = sequence_scaled.reshape(1, self.sequence_length, 1)
        
        # 預測
        prediction_scaled = self.model.predict(sequence_reshaped)
        prediction = self.scaler.inverse_transform(prediction_scaled)[0][0]
        
        # 計算所有感測器的預測分數
        all_sensors = np.array(list(self.sensor_mapping.values()))
        scores = np.exp(-np.abs(all_sensors - prediction))
        scores = scores / np.sum(scores)
        
        # 獲取前k個最可能的感測器
        top_indices = np.argsort(scores)[-top_k:][::-1]
        predictions = []
        
        for idx in top_indices:
            sensor = self.reverse_sensor_mapping[idx]
            probability = scores[idx]
            predictions.append((sensor, probability))
            
        return predictions 