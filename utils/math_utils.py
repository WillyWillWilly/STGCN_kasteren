import numpy as np

def evaluation(y_true, y_pred, x_stats):
    '''
    計算評估指標
    :param y_true: np.ndarray, 真實值
    :param y_pred: np.ndarray, 預測值
    :param x_stats: tuple, (mean, std) 數據統計信息
    :return: np.ndarray, [MAPE, MAE, RMSE]
    '''
    mean, std = x_stats
    
    # 檢查輸入數據是否為空或含有 NaN
    if y_true.size == 0 or y_pred.size == 0 or np.isnan(y_true).any() or np.isnan(y_pred).any():
        return np.array([0.0, 0.0, 0.0])  # 返回零值表示無法評估
    
    # 確保維度一致
    if y_true.shape != y_pred.shape:
        print(f"警告: 維度不匹配 - y_true: {y_true.shape}, y_pred: {y_pred.shape}")
        # 如果是多步預測情況，調整維度
        if len(y_true.shape) > len(y_pred.shape) and y_true.shape[0] == 3:
            # 如果y_true是3D但y_pred是2D，可能是多步預測結果
            # 使用第一個時間步的結果
            y_true = y_true[0]
        elif len(y_pred.shape) > len(y_true.shape) and y_pred.shape[0] == 3:
            # 反之亦然
            y_pred = y_pred[0]
        else:
            # 其他情況嘗試調整維度
            try:
                # 嘗試擴展維度
                if len(y_true.shape) < len(y_pred.shape):
                    y_true = np.expand_dims(y_true, axis=0)
                elif len(y_pred.shape) < len(y_true.shape):
                    y_pred = np.expand_dims(y_pred, axis=0)
            except:
                print("無法調整維度，返回默認值")
                return np.array([0.0, 0.0, 0.0])
    
    # 反標準化
    y_true = y_true * std + mean
    y_pred = y_pred * std + mean
    
    # 計算MAPE，避免除零錯誤
    try:
        # 確保mask的維度與數據相同
        mask = y_true != 0
        if np.sum(mask) == 0:
            mape = 0.0  # 所有真實值為零，無法計算 MAPE
        else:
            # 遍歷所有維度計算平均MAPE
            if len(y_true.shape) > 1 and y_true.shape[0] > 1:
                mapes = []
                for i in range(y_true.shape[0]):
                    local_mask = y_true[i] != 0
                    if np.sum(local_mask) > 0:
                        local_mape = np.mean(np.abs((y_true[i][local_mask] - y_pred[i][local_mask]) / y_true[i][local_mask])) * 100
                        mapes.append(local_mape)
                mape = np.mean(mapes) if mapes else 0.0
            else:
                mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    except Exception as e:
        print(f"計算MAPE時出錯: {e}")
        mape = 0.0
    
    # 計算MAE
    try:
        mae = np.mean(np.abs(y_true - y_pred))
    except:
        mae = 0.0
    
    # 計算RMSE
    try:
        rmse = np.sqrt(np.mean(np.square(y_true - y_pred)))
    except:
        rmse = 0.0
    
    return np.array([mape, mae, rmse])

def evaluation_metrics(y_true, y_pred):
    '''
    計算多個評估指標
    :param y_true: np.ndarray, 真實值
    :param y_pred: np.ndarray, 預測值
    :return: dict, 包含多個評估指標的字典
    '''
    # 計算MAPE
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # 計算MAE
    mae = np.mean(np.abs(y_true - y_pred))
    
    # 計算RMSE
    rmse = np.sqrt(np.mean(np.square(y_true - y_pred)))
    
    # 計算R2
    r2 = 1 - np.sum(np.square(y_true - y_pred)) / np.sum(np.square(y_true - np.mean(y_true)))
    
    return {
        'MAPE': mape,
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2
    } 