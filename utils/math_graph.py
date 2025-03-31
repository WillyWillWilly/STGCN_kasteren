import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse import csr_matrix

def weight_matrix(file_path, sigma2=0.1, epsilon=0.5, scaling=True):
    '''
    生成權重矩陣
    :param file_path: str, 數據文件路徑
    :param sigma2: float, 高斯核的方差
    :param epsilon: float, 閾值
    :param scaling: bool, 是否進行縮放
    :return: np.ndarray, 權重矩陣
    '''
    try:
        W = np.load(file_path)
    except FileNotFoundError:
        print(f'ERROR: input file was not found in {file_path}')
        return None
    
    # 檢查W是否為方陣
    if W.shape[0] != W.shape[1]:
        raise ValueError(f'ERROR: W is not square. Shape: {W.shape}')
    
    # 計算高斯核
    n = W.shape[0]
    W = W / 10000.  # 縮放權重
    W2, W_mask = W * W, np.ones([n, n]) - np.identity(n)
    # 計算高斯核
    return np.exp(-W2 / sigma2) * (np.exp(-W2 / sigma2) >= epsilon) * W_mask

def scaled_laplacian(W):
    '''
    計算縮放拉普拉斯矩陣
    :param W: np.ndarray, 權重矩陣
    :return: np.ndarray, 縮放拉普拉斯矩陣
    '''
    n, d = np.shape(W)[0], np.sum(W, axis=1)
    # D -> O(d)
    L = -W
    L.flat[::n + 1] = d
    for i in range(n):
        for j in range(n):
            if (d[i] > 0) and (d[j] > 0):
                L[i, j] = L[i, j] / np.sqrt(d[i] * d[j])
    return L

def cheb_poly_approx(L, Ks, n):
    '''
    使用切比雪夫多項式近似拉普拉斯矩陣
    :param L: np.ndarray, 拉普拉斯矩陣
    :param Ks: int, 切比雪夫多項式的階數
    :param n: int, 節點數量
    :return: np.ndarray, 切比雪夫多項式係數
    '''
    # 計算拉普拉斯矩陣的最大特徵值
    lambda_max = eigs(L, k=1, which='LR')[0][0].real
    
    # 處理可能的數值不穩定性
    lambda_max = max(lambda_max, 1.0)  # 確保至少為 1，避免除以小數
    
    # 計算切比雪夫多項式係數
    L_hat = 2 * L / lambda_max - np.identity(n)
    Lk = np.zeros([Ks, n, n])
    Lk[0] = np.identity(n)
    Lk[1] = L_hat
    
    for i in range(2, Ks):
        Lk[i] = 2 * L_hat @ Lk[i - 1] - Lk[i - 2]
    
    # 對結果進行簡單的正則化處理
    for i in range(Ks):
        # 將每個矩陣歸一化，避免過大或過小的值
        Lk[i] = Lk[i] / (np.linalg.norm(Lk[i]) + 1e-5)
    
    return Lk

def first_approx(W, n):
    '''
    使用一階近似計算拉普拉斯矩陣
    :param W: np.ndarray, 權重矩陣
    :param n: int, 節點數量
    :return: np.ndarray, 一階近似的拉普拉斯矩陣
    '''
    A = csr_matrix(W)
    D = np.diag(np.array(A.sum(axis=1)).flatten())
    L = D - A
    return L.toarray() 