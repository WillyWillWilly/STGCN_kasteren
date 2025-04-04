from data_loader.data_utils import gen_batch
from models.tester import model_inference
from models.base_model import build_model, model_save
from os.path import join as pjoin
import os
import shutil

import tensorflow as tf
import numpy as np
import time

def model_train(inputs, blocks, args, sum_path='./output/tensorboard'):
    '''
    訓練模型
    :param inputs: Dataset實例, 訓練數據
    :param blocks: list, 通道配置
    :param args: argparse實例, 訓練參數
    :param sum_path: str, tensorboard日誌路徑
    '''
    n, n_his, n_pred = args.n_route, args.n_his, args.n_pred
    Ks, Kt = args.ks, args.kt
    batch_size, epoch, inf_mode, opt = args.batch_size, args.epoch, args.inf_mode, args.opt
    
    # 定義模型輸入
    x = tf.compat.v1.placeholder(tf.float32, [None, n_his + 1, n, 1], name='data_input')
    keep_prob = tf.compat.v1.placeholder(tf.float32, name='keep_prob')
    
    # 定義模型損失
    train_loss, pred = build_model(x, n_his, Ks, Kt, blocks, keep_prob)
    tf.compat.v1.summary.scalar('train_loss', train_loss)
    copy_loss = tf.compat.v1.add_n(tf.compat.v1.get_collection('copy_loss'))
    tf.compat.v1.summary.scalar('copy_loss', copy_loss)
    
    # 學習率設置
    global_steps = tf.compat.v1.Variable(0, trainable=False)
    len_train = inputs.get_len('train')
    if len_train % batch_size == 0:
        epoch_step = len_train / batch_size
    else:
        epoch_step = int(len_train / batch_size) + 1
    
    # 學習率衰減
    lr = tf.compat.v1.train.exponential_decay(args.lr, global_steps, decay_steps=5 * epoch_step, decay_rate=0.7, staircase=True)
    tf.compat.v1.summary.scalar('learning_rate', lr)
    step_op = tf.compat.v1.assign_add(global_steps, 1)
    
    with tf.compat.v1.control_dependencies([step_op]):
        if opt == 'RMSProp':
            train_op = tf.compat.v1.train.RMSPropOptimizer(lr).minimize(train_loss)
        elif opt == 'ADAM':
            train_op = tf.compat.v1.train.AdamOptimizer(lr).minimize(train_loss)
        else:
            raise ValueError(f'ERROR: optimizer "{opt}" is not defined.')
    
    merged = tf.compat.v1.summary.merge_all()
    
    with tf.compat.v1.Session() as sess:
        writer = tf.compat.v1.summary.FileWriter(pjoin(sum_path, 'train'), sess.graph)
        sess.run(tf.compat.v1.global_variables_initializer())
        
        if inf_mode == 'sep':
            # 單步預測模式
            step_idx = n_pred - 1
            tmp_idx = [step_idx]
            min_val = min_va_val = np.array([4e1, 1e5, 1e5])
        elif inf_mode == 'merge':
            # 多步預測模式
            step_idx = tmp_idx = np.arange(3, n_pred + 1, 3) - 1
            min_val = min_va_val = np.array([4e1, 1e5, 1e5] * len(step_idx))
        else:
            raise ValueError(f'ERROR: test mode "{inf_mode}" is not defined.')
        
        # 記錄最佳驗證性能
        best_val_mape = float('inf')
        best_epoch = 0
        
        for i in range(epoch):
            start_time = time.time()
            for j, x_batch in enumerate(gen_batch(inputs.get_data('train'), batch_size, dynamic_batch=True, shuffle=True)):
                summary, _ = sess.run([merged, train_op],
                                    feed_dict={x: x_batch[:, 0:n_his + 1, :, :], keep_prob: 1.0})
                writer.add_summary(summary, i * epoch_step + j)
                
                if j % 50 == 0:
                    loss_value = sess.run([train_loss, copy_loss],
                                        feed_dict={x: x_batch[:, 0:n_his + 1, :, :], keep_prob: 1.0})
                    print(f'Epoch {i:2d}, Step {j:3d}: [{loss_value[0]:.3f}, {loss_value[1]:.3f}]')
            
            print(f'Epoch {i:2d} Training Time {time.time() - start_time:.3f}s')
            
            start_time = time.time()
            min_va_val, min_val = model_inference(sess, pred, inputs, batch_size, n_his, n_pred, step_idx, min_va_val, min_val)
            
            for ix in tmp_idx:
                va, te = min_va_val[ix - 2:ix + 1], min_val[ix - 2:ix + 1]
                print(f'Time Step {ix + 1}: '
                      f'MAPE {va[0]:7.3%}, {te[0]:7.3%}; '
                      f'MAE  {va[1]:4.3f}, {te[1]:4.3f}; '
                      f'RMSE {va[2]:6.3f}, {te[2]:6.3f}.')
            print(f'Epoch {i:2d} Inference Time {time.time() - start_time:.3f}s')
            
            # 檢查是否是最佳驗證性能
            current_val_mape = min_va_val[0]  # 使用第一個時間步的MAPE作為指標
            if current_val_mape < best_val_mape:
                best_val_mape = current_val_mape
                best_epoch = i
                
                # 刪除舊的最佳模型
                model_dir = './output/models'
                for file in os.listdir(model_dir):
                    if file.startswith('STGCN-best'):
                        os.remove(os.path.join(model_dir, file))
                
                # 保存新的最佳模型
                model_save(sess, global_steps, 'STGCN-best')
                print(f'<< 保存最佳模型 (Epoch {i}, MAPE: {best_val_mape:.3%})')
        
        writer.close()
    print(f'Training model finished! Best model saved at epoch {best_epoch} with validation MAPE: {best_val_mape:.3%}') 