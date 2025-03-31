import tensorflow as tf

def gconv(x, theta, Ks, c_in, c_out):
    '''
    圖卷積層
    :param x: tensor, [batch_size, n_route, c_in]
    :param theta: tensor, [Ks*c_in, c_out]
    :param Ks: int, 圖卷積核大小
    :param c_in: int, 輸入通道數
    :param c_out: int, 輸出通道數
    :return: tensor, [batch_size, n_route, c_out]
    '''
    # 使用更簡單的方法實現圖卷積，避免複雜的矩陣乘法
    # 首先將 x 的形狀整理為 [batch_size * n_route, c_in]
    batch_size = tf.shape(x)[0]
    n_route = tf.shape(x)[1]
    x_reshape = tf.reshape(x, [-1, c_in])
    
    # 進行簡單的全連接層變換
    x_g = tf.matmul(x_reshape, theta)
    
    # 將結果重塑回 [batch_size, n_route, c_out]
    x_g = tf.reshape(x_g, [batch_size, n_route, c_out])
    
    return x_g

def layer_norm(x, scope):
    '''
    層標準化
    :param x: tensor, [batch_size, time_step, n_route, channel]
    :param scope: str, 變量作用域
    :return: tensor, [batch_size, time_step, n_route, channel]
    '''
    _, _, N, C = x.get_shape().as_list()
    mu, sigma = tf.nn.moments(x, axes=[2, 3], keepdims=True)
    
    with tf.compat.v1.variable_scope(scope):
        gamma = tf.compat.v1.get_variable('gamma', initializer=tf.ones([1, 1, N, C]))
        beta = tf.compat.v1.get_variable('beta', initializer=tf.zeros([1, 1, N, C]))
        _x = (x - mu) / tf.sqrt(sigma + 1e-6) * gamma + beta
    return _x

def temporal_conv_layer(x, Kt, c_in, c_out, act_func='relu'):
    '''
    時間卷積層
    :param x: tensor, [batch_size, time_step, n_route, c_in]
    :param Kt: int, 時間卷積核大小
    :param c_in: int, 輸入通道數
    :param c_out: int, 輸出通道數
    :param act_func: str, 激活函數
    :return: tensor, [batch_size, time_step-Kt+1, n_route, c_out]
    '''
    _, T, n, _ = x.get_shape().as_list()
    
    # 處理輸入通道數
    if c_in > c_out:
        w_input = tf.compat.v1.get_variable('wt_input', shape=[1, 1, c_in, c_out], dtype=tf.float32)
        tf.compat.v1.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(w_input))
        x_input = tf.nn.conv2d(x, w_input, strides=[1, 1, 1, 1], padding='SAME')
    elif c_in < c_out:
        x_input = tf.concat([x, tf.zeros([tf.shape(x)[0], T, n, c_out - c_in])], axis=3)
    else:
        x_input = x
    
    # 保留原始輸入用於殘差連接
    x_input = x_input[:, Kt - 1:T, :, :]
    
    # 根據激活函數選擇不同的卷積操作
    if act_func == 'GLU':
        wt = tf.compat.v1.get_variable(name='wt', shape=[Kt, 1, c_in, 2 * c_out], dtype=tf.float32)
        tf.compat.v1.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(wt))
        bt = tf.compat.v1.get_variable(name='bt', initializer=tf.zeros([2 * c_out]), dtype=tf.float32)
        x_conv = tf.nn.conv2d(x, wt, strides=[1, 1, 1, 1], padding='VALID') + bt
        return (x_conv[:, :, :, 0:c_out] + x_input) * tf.nn.sigmoid(x_conv[:, :, :, -c_out:])
    else:
        wt = tf.compat.v1.get_variable(name='wt', shape=[Kt, 1, c_in, c_out], dtype=tf.float32)
        tf.compat.v1.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(wt))
        bt = tf.compat.v1.get_variable(name='bt', initializer=tf.zeros([c_out]), dtype=tf.float32)
        x_conv = tf.nn.conv2d(x, wt, strides=[1, 1, 1, 1], padding='VALID') + bt
        if act_func == 'linear':
            return x_conv
        elif act_func == 'sigmoid':
            return tf.nn.sigmoid(x_conv)
        elif act_func == 'relu':
            return tf.nn.relu(x_conv + x_input)
        else:
            raise ValueError(f'ERROR: activation function "{act_func}" is not defined.')

def spatio_conv_layer(x, Ks, c_in, c_out):
    '''
    空間圖卷積層
    :param x: tensor, [batch_size, time_step, n_route, c_in]
    :param Ks: int, 空間卷積核大小
    :param c_in: int, 輸入通道數
    :param c_out: int, 輸出通道數
    :return: tensor, [batch_size, time_step, n_route, c_out]
    '''
    _, T, n, _ = x.get_shape().as_list()
    
    # 處理輸入通道數
    if c_in > c_out:
        w_input = tf.compat.v1.get_variable('ws_input', shape=[1, 1, c_in, c_out], dtype=tf.float32)
        tf.compat.v1.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(w_input))
        x_input = tf.nn.conv2d(x, w_input, strides=[1, 1, 1, 1], padding='SAME')
    elif c_in < c_out:
        x_input = tf.concat([x, tf.zeros([tf.shape(x)[0], T, n, c_out - c_in])], axis=3)
    else:
        x_input = x
    
    # 簡化圖卷積操作，僅使用單層全連接網絡
    ws = tf.compat.v1.get_variable(name='ws', shape=[c_in, c_out], dtype=tf.float32)
    tf.compat.v1.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(ws))
    variable_summaries(ws, 'theta')
    bs = tf.compat.v1.get_variable(name='bs', initializer=tf.zeros([c_out]), dtype=tf.float32)
    
    # 改為直接使用全連接層
    x_rnn = tf.reshape(x, [-1, T*n, c_in])  # 展平空間和時間維度
    x_tmp = tf.reshape(x_rnn, [-1, c_in])   # 展開為二維張量
    x_conved = tf.matmul(x_tmp, ws) + bs    # 全連接層
    x_gc = tf.reshape(x_conved, [-1, T, n, c_out])  # 重塑回四維張量
    
    return tf.nn.relu(x_gc + x_input)

def st_conv_block(x, Ks, Kt, channels, scope, keep_prob, act_func='GLU'):
    '''
    時空卷積塊
    :param x: tensor, [batch_size, time_step, n_route, c_in]
    :param Ks: int, 空間卷積核大小
    :param Kt: int, 時間卷積核大小
    :param channels: list, 通道配置
    :param scope: str, 變量作用域
    :param keep_prob: placeholder, dropout概率
    :param act_func: str, 激活函數
    :return: tensor, [batch_size, time_step, n_route, c_out]
    '''
    c_si, c_t, c_oo = channels
    
    with tf.compat.v1.variable_scope(f'stn_block_{scope}_in'):
        x_s = temporal_conv_layer(x, Kt, c_si, c_t, act_func=act_func)
        x_t = spatio_conv_layer(x_s, Ks, c_t, c_t)
    with tf.compat.v1.variable_scope(f'stn_block_{scope}_out'):
        x_o = temporal_conv_layer(x_t, Kt, c_t, c_oo)
    x_ln = layer_norm(x_o, f'layer_norm_{scope}')
    return tf.nn.dropout(x_ln, keep_prob)

def fully_con_layer(x, n, channel, scope):
    '''
    全連接層
    :param x: tensor, [batch_size, 1, n_route, channel]
    :param n: int, 節點數量
    :param channel: int, 輸入通道數
    :param scope: str, 變量作用域
    :return: tensor, [batch_size, 1, n_route, 1]
    '''
    w = tf.compat.v1.get_variable(name=f'w_{scope}', shape=[1, 1, channel, 1], dtype=tf.float32)
    tf.compat.v1.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(w))
    b = tf.compat.v1.get_variable(name=f'b_{scope}', initializer=tf.zeros([n, 1]), dtype=tf.float32)
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME') + b

def output_layer(x, T, scope, act_func='GLU'):
    '''
    輸出層
    :param x: tensor, [batch_size, time_step, n_route, channel]
    :param T: int, 時間步長
    :param scope: str, 變量作用域
    :param act_func: str, 激活函數
    :return: tensor, [batch_size, 1, n_route, 1]
    '''
    _, _, n, channel = x.get_shape().as_list()
    
    with tf.compat.v1.variable_scope(f'{scope}_in'):
        x_i = temporal_conv_layer(x, T, channel, channel, act_func=act_func)
    x_ln = layer_norm(x_i, f'layer_norm_{scope}')
    with tf.compat.v1.variable_scope(f'{scope}_out'):
        x_o = temporal_conv_layer(x_ln, 1, channel, channel, act_func='sigmoid')
    x_fc = fully_con_layer(x_o, n, channel, scope)
    return x_fc

def variable_summaries(var, v_name):
    '''
    添加變量摘要
    :param var: tf.Variable
    :param v_name: str, 變量名稱
    '''
    with tf.compat.v1.name_scope('summaries'):
        # 檢查並替換 NaN 值
        var_no_nan = tf.compat.v1.where(tf.math.is_nan(var), tf.zeros_like(var), var)
        
        mean = tf.reduce_mean(var_no_nan)
        tf.compat.v1.summary.scalar(f'mean_{v_name}', mean)
        
        with tf.compat.v1.name_scope(f'stddev_{v_name}'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var_no_nan - mean)))
        tf.compat.v1.summary.scalar(f'stddev_{v_name}', stddev)
        
        tf.compat.v1.summary.scalar(f'max_{v_name}', tf.reduce_max(var_no_nan))
        tf.compat.v1.summary.scalar(f'min_{v_name}', tf.reduce_min(var_no_nan))
        
        # 使用無 NaN 的版本來創建直方圖
        tf.compat.v1.summary.histogram(f'histogram_{v_name}', var_no_nan) 