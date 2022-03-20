import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from torch.utils.data import DataLoader, TensorDataset
import logging
import time as t

data = pd.read_csv("data/15.csv")
batch_size = 2
n_features = 1
device = torch.device("cpu")
log_path = "logging/log15.txt"
input_steps = 20


# 保存日志文件
def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


# 找出所有为nan的索引
class Seq_count():
    def __init__(self, seq, count):
        self.seq = seq
        self.count = count


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
def loopLstm(seq, count):
    for i in range(0, count):
        sample = np.array(seq[len(seq) - input_steps:len(seq)])
        temp_seq = seq[len(seq) - 500: len(seq) ]
        X, Y = split_sequence(temp_seq, input_steps)
        # 梯度提升回归树
        model_gbdt = GradientBoostingRegressor(n_estimators=245,
                                               learning_rate=0.01,
                                               max_depth=3)
        model_gbdt.fit(X, Y)
        pre = model_gbdt.predict(np.array(sample).reshape(1, -1))
        logger.info('---------------------第{}个预测结果：{}'.format(len(seq) + 1, pre))
        seq = np.append(seq, pre)
    return seq


if __name__ == '__main__':
    seq_count = deal_data(data)
    pre_seq = np.array([])
    begin_time = t.time()
    # 日志文件初始化
    logger = get_logger(log_path)
    logger.info('start training!')
    for i in seq_count:
        seq = np.append(pre_seq, i.seq)
        pre_seq = loopLstm(seq, i.count)
    end_time = t.time()
    print("运行时间为: {:.3f} 小时".format((end_time - begin_time) / 3600))
    logger.info('finish training!')
    pd_data = pd.DataFrame(pre_seq, columns=['X'])
    pd_data.to_csv('result/1.csv',index=False)
