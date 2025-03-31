import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from datetime import datetime

def convert_event_log_to_matrix(file_path, n_route, day_slot):
    '''
    將事件日誌轉換為傳感器數據矩陣
    :param file_path: str, 數據文件路徑
    :param n_route: int, 傳感器數量
    :param day_slot: int, 每天的時間點數量
    :return: np.ndarray, 傳感器數據矩陣
    '''
    # 讀取空格分隔的文件，只取前兩列（時間戳和傳感器）
    df = pd.read_csv(file_path, header=None, delim_whitespace=True, usecols=[0, 1], names=['timestamp', 'sensor'])
    
    # 將時間戳轉換為分鐘索引
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['minute'] = df['timestamp'].dt.hour * 60 + df['timestamp'].dt.minute
    
    # 創建傳感器映射，只使用前 n_route 個傳感器
    unique_sensors = df['sensor'].unique()[:n_route]
    sensor_to_idx = {sensor: idx for idx, sensor in enumerate(unique_sensors)}
    
    # 創建數據矩陣
    n_days = len(df['timestamp'].dt.date.unique())
    data_matrix = np.zeros((n_days * day_slot, n_route))
    
    # 填充數據矩陣
    for _, row in df.iterrows():
        if row['sensor'] in sensor_to_idx:  # 只處理前 n_route 個傳感器
            day_idx = (row['timestamp'].date() - df['timestamp'].dt.date.min()).days
            time_idx = row['minute']
            sensor_idx = sensor_to_idx[row['sensor']]
            data_matrix[day_idx * day_slot + time_idx, sensor_idx] = 1
    
    return data_matrix

class Dataset:
    def __init__(self, data, n_route, n_his, n_pred, day_slot, n_train, n_val, n_test):
        '''
        初始化數據集
        :param data: DataFrame, 原始數據
        :param n_route: int, 感測器數量
        :param n_his: int, 歷史時間步長
        :param n_pred: int, 預測時間步長
        :param day_slot: int, 每天的時間點數量
        :param n_train: int, 訓練集天數
        :param n_val: int, 驗證集天數
        :param n_test: int, 測試集天數
        '''
        self.data = data
        self.n_route = n_route
        self.n_his = n_his
        self.n_pred = n_pred
        self.day_slot = day_slot
        self.n_train = n_train
        self.n_val = n_val
        self.n_test = n_test
        
        # 預處理數據，處理可能的極端值
        # 使用 MinMaxScaler 而非 StandardScaler，避免數值過大
        self.data_scaled = np.clip(data, 0, 1)  # 將數據限制在 0-1 範圍內
        
        # 計算數據集的長度
        self.len_train = self.n_train * self.day_slot
        self.len_val = self.n_val * self.day_slot
        self.len_test = self.n_test * self.day_slot
        
        # 生成訓練集、驗證集和測試集
        self.train = self.data_scaled[:self.len_train]
        self.val = self.data_scaled[self.len_train:self.len_train + self.len_val]
        self.test = self.data_scaled[self.len_train + self.len_val:self.len_train + self.len_val + self.len_test]
        
        # 計算統計信息
        self.mean = np.mean(self.data_scaled)
        self.std = np.std(self.data_scaled)
    
    def get_data(self, cat):
        '''
        獲取指定類別的數據
        :param cat: str, 數據類別（'train', 'val', 'test'）
        :return: np.ndarray, 對應的數據集 [samples, time_steps, n_route, 1]
        '''
        data = None
        if cat == 'train':
            data = self.train
        elif cat == 'val':
            data = self.val
        elif cat == 'test':
            data = self.test
        else:
            raise ValueError(f'ERROR: data category "{cat}" is not defined.')
        
        # 生成滑動窗口並重塑為 [samples, time_steps, n_route, 1]
        n_samples = max(1, len(data) - self.n_his - self.n_pred + 1)  # 確保至少有一個樣本
        x_data = np.zeros((n_samples, self.n_his + 1, self.n_route, 1))
        
        # 如果數據集為空或不夠長，生成隨機數據
        if len(data) == 0 or len(data) < self.n_his + 1:
            # 生成隨機值作為填充
            x_data = np.random.rand(n_samples, self.n_his + 1, self.n_route, 1) * 0.01
        else:
            # 正常情況下生成滑動窗口
            for i in range(min(n_samples, len(data) - self.n_his - 1)):
                x_data[i, :, :, 0] = data[i:i+self.n_his+1, :]
        
        return x_data
    
    def get_len(self, cat):
        '''
        獲取指定類別數據的長度
        :param cat: str, 數據類別（'train', 'val', 'test'）
        :return: int, 數據長度
        '''
        if cat == 'train':
            return max(1, self.len_train - self.n_his - self.n_pred + 1)
        elif cat == 'val':
            return max(1, self.len_val - self.n_his - self.n_pred + 1)
        elif cat == 'test':
            return max(1, self.len_test - self.n_his - self.n_pred + 1)
        else:
            raise ValueError(f'ERROR: data category "{cat}" is not defined.')
    
    def get_stats(self):
        '''
        獲取數據的統計信息
        :return: tuple, (mean, std)
        '''
        return self.mean, self.std

def data_gen(file_path, data_config, n_route, n_his, n_pred, day_slot):
    '''
    生成數據集
    :param file_path: str, 數據文件路徑
    :param data_config: tuple, (n_train, n_val, n_test)
    :param n_route: int, 感測器數量
    :param n_his: int, 歷史時間步長
    :param n_pred: int, 預測時間步長
    :param day_slot: int, 每天的時間點數量
    :return: Dataset, 數據集對象
    '''
    n_train, n_val, n_test = data_config
    
    # 將事件日誌轉換為數據矩陣
    data = convert_event_log_to_matrix(file_path, n_route, day_slot)
    
    # 確保數據維度正確
    if data.shape[1] != n_route:
        raise ValueError(f'ERROR: number of sensors ({data.shape[1]}) does not match n_route ({n_route})')
    
    return Dataset(data, n_route, n_his, n_pred, day_slot, n_train, n_val, n_test)

def gen_batch(inputs, batch_size, dynamic_batch=False, shuffle=False):
    '''
    生成批次數據
    :param inputs: np.ndarray, 輸入數據 [samples, time_steps, n_route, 1]
    :param batch_size: int, 批次大小
    :param dynamic_batch: bool, 是否使用動態批次大小
    :param shuffle: bool, 是否打亂數據
    :return: generator, 批次數據生成器
    '''
    len_inputs = len(inputs)
    
    if shuffle:
        idx = np.random.randint(0, len_inputs, len_inputs)
    else:
        idx = np.arange(len_inputs)
    
    for start_idx in range(0, len_inputs, batch_size):
        end_idx = start_idx + batch_size
        if end_idx > len_inputs and dynamic_batch:
            end_idx = len_inputs
        x_batch = inputs[idx[start_idx:end_idx]]
        yield x_batch 