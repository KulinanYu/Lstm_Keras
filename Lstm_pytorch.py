import logging

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import time as t
from torch.nn import Module, GRU, LSTM, Linear
from torch.utils.data import DataLoader, TensorDataset

data = pd.read_csv("data/9.csv")
batch_size = 2
n_features = 1
device = torch.device("cuda")
log_path = "logging/log9.txt"

# 9.csv特有参数(20试一下)
n_steps = 10
LR = 0.01


# 1.csv四个早停策略
def strategy_stop(loss_list):
    lg = len(loss_list)
    strategy1 = loss_list[lg - 1] >= loss_list[lg - 2] >= loss_list[lg - 3]
    strategy2 = loss_list[lg - 1] >= 2 * loss_list[lg - 2] or loss_list[lg - 1] >= loss_list[lg - 3]
    strategy3 = loss_list[lg - 1] <= 1
    # 策略四配合动态调整学习率一块使用(即从第二轮开始loss下降超过十倍)
    strategy4 = (lg >= 5 and loss_list[lg - 1] * 10 <= loss_list[lg - 2])
    # 连续三轮loss在1%徘徊的（适用于大型loss）
    strategy5 = loss_list[lg - 1] * 1.01 >= loss_list[lg - 2] and loss_list[lg - 2] * 1.01 >= loss_list[lg - 3]
    if strategy1 or strategy2 or strategy3 or strategy4 or strategy5:
        return True
    else:
        return False


def dy_change_lr(loss_list, lr):
    lg = len(loss_list)
    # lr过大(喂新数据也可以到时候loss变大2倍+)
    if loss_list[lg - 1] >= 2 * loss_list[lg - 2] or loss_list[lg - 1] >= loss_list[lg - 2] >= loss_list[lg - 3]:
        lr = lr / 1.2
        return lr
    # lr过小（9.csv专用：0.2
    if 1.2 * loss_list[lg - 1] >= loss_list[lg - 2] >= 0.85 * loss_list[lg - 3] and loss_list[lg - 1] <= loss_list[
        lg - 2]:
        lr = lr * 1.2
        return lr
    return lr


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


class Lstm(Module):
    def __init__(self):
        super(Lstm, self).__init__()
        self.lstm = GRU(input_size=n_steps, hidden_size=100,
                        num_layers=2, dropout=0.1, batch_first=True)
        self.linear = Linear(in_features=100, out_features=1)

    def forward(self, x):
        x = x.view(len(x), 1, -1)  # 维度变为 [20,1,5]
        lstm_out, _ = self.lstm(x)
        linear_out = self.linear(lstm_out)
        return linear_out


# 找出所有为nan的索引
class Seq_count():
    def __init__(self, seq, count):
        self.seq = seq
        self.count = count

    def get_seq(self):
        return self.seq

    def get_count(self):
        return self.count


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
    LR = 0.005 * (((seq[len(seq)-1-count:len(seq)-1]**2)**0.5).mean()/2)  # 当有新数据均值进入则重新定义学习率
    for i in range(0, count):
        model = Lstm().to(device)  # 如果是GPU训练， .to(device) 会把模型/数据复制到GPU显存中

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = torch.nn.MSELoss()  # 这两句是定义优化器和loss
        loss_func = nn.MSELoss()
        temp_seq = seq[len(seq) - 801: len(seq) - 1]
        X, Y = split_sequence(temp_seq, n_steps)
        X = X.reshape((X.shape[0], X.shape[1], n_features))
        Y = Y.reshape((Y.shape[0], 1, n_features))
        X = torch.from_numpy(X).float()
        Y = torch.from_numpy(Y).float()
        # DataLoader可自动生成可训练的batch数据
        train_loader = DataLoader(TensorDataset(X, Y), batch_size=batch_size)

        loss_list = [1e10, 1e10]
        # 迭代50轮
        for j in range(0, 10):
            model.train()
            for index, data in enumerate(train_loader):
                x, y = data
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad()
                pre_y = model(x)  # 这个执行的是def forward()
                loss = loss_func(pre_y, y)
                loss.backward()
                optimizer.step()
            loss_list.append(loss.item())
            logger.info('Epoch:{}, Loss:{:.5f}'.format(j + 1, loss_list[j + 2]))
            if strategy_stop(loss_list):
                break
        # 下一次lstm的lr调整
        LR = dy_change_lr(loss_list, LR)
        logger.info('---------------------Learning_rate:{:.6f}'.format(LR))
        # 预测过程
        sample = np.array(seq[len(seq) - n_steps:len(seq)])
        sample = sample.reshape((1, n_steps, n_features))
        sample = torch.from_numpy(sample).float().to(device)
        pre = model(sample)
        logger.info('---------------------第{}个预测结果：{}'.format(len(seq)+1, pre))
        pre = pre.cuda().data.cpu().numpy()
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
    pd_data.to_csv('result/9.csv',index=False)
