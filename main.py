import os
# 啟用GPU支持 - 如果有多個GPU，可以選擇特定的GPU，例如"0,1"表示使用前兩個GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from os.path import join as pjoin

import tensorflow as tf
# 禁用 Eager Execution
tf.compat.v1.disable_eager_execution()

# 優化TensorFlow配置以支持GPU
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # 動態分配GPU內存
config.gpu_options.per_process_gpu_memory_fraction = 0.8  # 限制GPU內存使用比例
config.log_device_placement = False  # 不記錄設備放置日誌
config.allow_soft_placement = True  # 如果操作無法在GPU上運行，自動轉移到CPU

# 添加XLA JIT編譯選項，可以提高TensorFlow操作性能
config.graph_options.optimizer_options.global_jit_level = tf.compat.v1.OptimizerOptions.ON_1

# 設置為默認配置
tf.compat.v1.keras.backend.clear_session()
tf.compat.v1.disable_eager_execution()

# 檢查GPU是否可用
try:
    gpu_available = tf.test.is_gpu_available()
    print("GPU可用性:", gpu_available)
    if gpu_available:
        gpu_devices = tf.config.list_physical_devices('GPU')
        print("GPU設備:", gpu_devices)
except:
    print("無法檢測GPU可用性")

# 設置操作日誌層級，減少不必要的日誌輸出
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=全部日誌, 1=INFO, 2=WARNING, 3=ERROR

from utils.math_graph import *
from data_loader.data_utils import *
from models.trainer import model_train
from models.tester import model_test

import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--n_route', type=int, default=14)  # 修改為 Kasteren 數據集的傳感器數量
parser.add_argument('--n_his', type=int, default=6)     # 減少歷史時間步長，適應小數據集
parser.add_argument('--n_pred', type=int, default=3)    # 減少預測時間步長，避免維度問題
parser.add_argument('--batch_size', type=int, default=10) # 減小批次大小
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--save', type=int, default=10)
parser.add_argument('--ks', type=int, default=3)
parser.add_argument('--kt', type=int, default=2,
                    help='time kernel size of spatial-temporal convolution')
parser.add_argument('--lr', type=float, default=5e-4)  # 適當增加學習率以加速收斂
parser.add_argument('--opt', type=str, default='RMSProp')
parser.add_argument('--graph', type=str, default='default')
parser.add_argument('--inf_mode', type=str, default='merge')
parser.add_argument('--day_slot', type=int, default=288)  # 每天的時間點數量
parser.add_argument('--n_train', type=int, default=34)    # 訓練集天數
parser.add_argument('--n_val', type=int, default=5)       # 驗證集天數
parser.add_argument('--n_test', type=int, default=5)      # 測試集天數
parser.add_argument('--use_gpu', type=bool, default=True)  # 是否使用GPU
parser.add_argument('--gpu_id', type=str, default='0')    # 使用的GPU ID

args = parser.parse_args()

# 更新CUDA_VISIBLE_DEVICES環境變量
if args.use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    print(f"使用GPU ID: {args.gpu_id}")
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 禁用GPU
    print("使用CPU進行計算")

print(f'Training configs: {args}')

n, n_his, n_pred = args.n_route, args.n_his, args.n_pred
Ks, Kt = args.ks, args.kt
# blocks: 通道配置
blocks = [[1, n, 16], [16, n, 32]]  # 確保中間通道數與 n_route 一致

# 為 Kasteren 數據集創建一個簡單的權重矩陣
W = np.ones((n, n))  # 創建一個全1矩陣作為初始權重
np.fill_diagonal(W, 0)  # 將對角線設為0
W = W / (n-1)  # 歸一化

# 計算圖核
L = scaled_laplacian(W)
Lk = cheb_poly_approx(L, Ks, n)
tf.compat.v1.add_to_collection(name='graph_kernel', value=tf.cast(tf.constant(Lk), tf.float32))

# 數據預處理
data_file = 'base_kasteren-m.csv'
n_train, n_val, n_test = args.n_train, args.n_val, args.n_test
Kesteren = data_gen(pjoin('./dataset', data_file), (n_train, n_val, n_test), n, n_his, n_pred, args.day_slot)
print(f'>> Loading dataset with Mean: {Kesteren.mean:.2f}, STD: {Kesteren.std:.2f}')

# 計算模型大小和參數數量的粗略估計
def estimate_model_size():
    # 簡單估計模型參數數量 (根據blocks和層配置)
    total_params = 0
    for i, channels in enumerate(blocks):
        c_si, c_t, c_oo = channels
        # 時間卷積層參數 (輸入)
        total_params += Kt * 1 * c_si * c_t * 2  # GLU
        # 空間卷積層參數
        total_params += c_t * c_t
        # 時間卷積層參數 (輸出)
        total_params += Kt * 1 * c_t * c_oo
    # 輸出層參數 - 修正變量名稱
    n_route = args.n_route
    last_channel = blocks[-1][2]  # 獲取最後一個塊的輸出通道數
    total_params += 2 * n_his * last_channel * last_channel
    total_params += last_channel * 1
    
    # 每個參數假設是float32 (4字節)
    model_size_mb = total_params * 4 / (1024 * 1024)
    return total_params, model_size_mb

# 估計模型大小
try:
    total_params, model_size_mb = estimate_model_size()
    print(f"估計模型參數: {total_params:,}")
    print(f"估計模型大小: {model_size_mb:.2f} MB")
except:
    print("無法估計模型大小")

if __name__ == '__main__':
    # 監控GPU內存使用
    if args.use_gpu and tf.test.is_gpu_available():
        print("訓練前GPU內存使用:")
        try:
            # 使用更兼容的方法獲取GPU信息
            import subprocess
            import re
            try:
                nvidia_info = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits']).decode('utf-8')
                for i, line in enumerate(nvidia_info.strip().split('\n')):
                    mem_used, mem_total = map(int, line.split(','))
                    print(f"  GPU #{i}: 已用 {mem_used} MB / 總計 {mem_total} MB ({mem_used/mem_total*100:.1f}%)")
            except:
                print("  無法獲取GPU內存信息，需要安裝nvidia-smi")
        except:
            print("  無法獲取GPU內存信息")
    
    # 開始訓練
    print("開始模型訓練...")
    model_train(Kesteren, blocks, args)
    
    # 進行測試
    print("開始模型測試...")
    model_test(Kesteren, Kesteren.get_len('test'), n_his, n_pred, args.inf_mode)
    
    # 再次監控GPU內存，查看使用情況
    if args.use_gpu and tf.test.is_gpu_available():
        print("測試後GPU內存使用:")
        try:
            # 使用更兼容的方法獲取GPU信息
            import subprocess
            import re
            try:
                nvidia_info = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits']).decode('utf-8')
                for i, line in enumerate(nvidia_info.strip().split('\n')):
                    mem_used, mem_total = map(int, line.split(','))
                    print(f"  GPU #{i}: 已用 {mem_used} MB / 總計 {mem_total} MB ({mem_used/mem_total*100:.1f}%)")
            except:
                print("  無法獲取GPU內存信息，需要安裝nvidia-smi")
        except:
            print("  無法獲取GPU內存信息") 