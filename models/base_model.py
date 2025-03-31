from models.layers import *
from os.path import join as pjoin
import tensorflow as tf

def build_model(inputs, n_his, Ks, Kt, blocks, keep_prob):
    '''
    構建基礎模型
    :param inputs: placeholder, 輸入數據
    :param n_his: int, 歷史時間步長
    :param Ks: int, 空間卷積核大小
    :param Kt: int, 時間卷積核大小
    :param blocks: list, 通道配置
    :param keep_prob: placeholder, dropout概率
    :return: tuple, (train_loss, single_pred)
    '''
    x = inputs[:, 0:n_his, :, :]
    
    # 計算輸出層的時間卷積核大小
    Ko = n_his
    
    # 時空卷積塊
    for i, channels in enumerate(blocks):
        x = st_conv_block(x, Ks, Kt, channels, i, keep_prob, act_func='GLU')
        Ko -= 2 * (Kt - 1)
    
    # 輸出層
    if Ko > 1:
        y = output_layer(x, Ko, 'output_layer')
    else:
        raise ValueError(f'ERROR: kernel size Ko must be greater than 1, but received "{Ko}".')
    
    # 計算損失
    tf.compat.v1.add_to_collection(name='copy_loss',
                        value=tf.nn.l2_loss(inputs[:, n_his - 1:n_his, :, :] - inputs[:, n_his:n_his + 1, :, :]))
    train_loss = tf.nn.l2_loss(y - inputs[:, n_his:n_his + 1, :, :])
    single_pred = y[:, 0, :, :]
    tf.compat.v1.add_to_collection(name='y_pred', value=single_pred)
    
    return train_loss, single_pred

def model_save(sess, global_steps, model_name, save_path='./output/models/'):
    '''
    保存模型
    :param sess: tf.Session
    :param global_steps: tensor, 全局步數
    :param model_name: str, 模型名稱
    :param save_path: str, 保存路徑
    '''
    saver = tf.compat.v1.train.Saver(max_to_keep=3)
    prefix_path = saver.save(sess, pjoin(save_path, model_name), global_step=global_steps)
    print(f'<< Saving model to {prefix_path} ...') 