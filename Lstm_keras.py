import numpy as np
import pandas as pd
from keras.layers import LSTM, Dense
from keras.models import Sequential

from numpy import array, hstack


from sklearn.metrics import mean_absolute_error as mae
import tensorflow as tf
from tensorflow.python.keras.backend import set_session

data = pd.read_csv("data/1.csv")
data1 = data[:1000]
n_steps = 20
n_features = 1


# 找出所有为nan的索引
class Seq_count():
    def __init__(self, seq, count):
        self.seq = seq
        self.count = count

    def get_seq(self):
        return self.seq

    def get_count(self):
        return self.count


def gpu_train_init():
    sess_config = tf.compat.v1.ConfigProto(log_device_placement=True, allow_soft_placement=True)
    sess_config.gpu_options.per_process_gpu_memory_fraction = 0.8  # 最多使用70%GPU内存
    sess_config.gpu_options.allow_growth = True  # 初始化时不全部占满GPU显存, 按需分配
    sess = tf.compat.v1.Session(config=sess_config)
    set_session(sess)


# 对数据进行预处理（进行分段）
def deal_data(data):
    index = data[data["X"].isna()].index
    start_list = []  # 存储每个非空序列的起始位置
    end_list = []  # 存储每个非空序列的结束位置
    start_list.append(0)

    end_list.append(index[0])
    for i in range(1, len(index)):
        if index[i - 1] + 1 != index[i]:
            start_list.append(index[i - 1] + 1)
            end_list.append(index[i])
    start_list.append(index[-1] + 1)
    end_list.append(len(data))
    # 创建类数组
    seq_count_list = []
    # 查找每个非空序列，以及其后的缺失长度（存到Seq_count对象中）
    for i in range(len(start_list) - 1):
        seq = data["X"][start_list[i]:end_list[i]]
        count = start_list[i + 1] - end_list[i]
        seq_count_list.append(Seq_count(seq, count))
    return seq_count_list


# 创建模型(GPU内存允许的情况下增大batch将会增快训练速度)
def create_model():
    model = Sequential()
    model.add(LSTM(2, activation='relu', input_shape=(n_steps, n_features)))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer="adam", metrics=['acc'])
    return model


# 构造一元监督学习型数据
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # 获取待预测数据的位置
        end_ix = i + n_steps
        # 如果待预测数据超过序列长度，构造完成
        if end_ix >= len(sequence):
            break
        # 分别汇总 输入 和 输出 数据集
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


# 递归预测
def loopLstm(model, seq, count):
    for i in range(0, count):
        print("======开始构建第 ", i, " 个LSTM,序列长度为：", len(seq))
        X, Y = split_sequence(seq, n_steps)
        X = X.reshape((X.shape[0], X.shape[1], n_features))
        model.fit(X, Y, epochs=2, verbose=0)
        # 最后n_steps个数组为实验样本
        sample = np.array(seq[len(seq) - n_steps:len(seq)])
        # reshape(样本数量,步长，特征数)
        sample = sample.reshape((1, n_steps, n_features))
        pre = model.predict(sample, verbose=0)
        print("======第 ", len(seq) + 1, " 个元素的预测结果为：", pre)
        seq = np.append(seq, pre)
    return seq


def test():
    # 预测981-1000的元素(测试)
    t = np.array(data[0:980])
    pred = loopLstm(model, t, 20)
    pre = pred[980:1000].reshape(20, 1)
    real = np.array(data[980:1000])
    print(mae(pre, real))



if __name__ == '__main__':
    # 使用gpu
    gpu_train_init()
    # 创建一个model

    model = create_model()
    seq_count = deal_data(data)
    pre_seq = np.array([])
    for i in seq_count:
        seq = np.append(pre_seq, i.seq)
        pre_seq = loopLstm(model, seq, i.count)
        print("seq", seq.shape)
        print("pre_seq", pre_seq.shape)
