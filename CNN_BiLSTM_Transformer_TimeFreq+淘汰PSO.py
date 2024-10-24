# CNN在空间方向上卷积, Transformer，时频结合

import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import time
import torch.fft as fft

from datetime import datetime
from torch.optim import lr_scheduler as lr_scheduler
from torch import optim as optim
from scipy.stats import pearsonr
from mpl_toolkits.mplot3d import Axes3D


#  加载数据并制作数据集

def make_dataset(filepath, xcol, ycol, seq_len=64, train_ratio=0.8, toshuffle=False, pred_mid=False):
    """
    从文件中读取数据，并根据参数生成数据集
    参数说明：
        filepath: 数据文件路径
        xcol: [x_col_from, x_col_to]，训练特征的列范围
        ycol: [y_col_from, y_col_to]，结果特征的列范围
        seq_len: 输入数据的序列长度（行数），即使用seq_len行计算一行特征
        train_ratio: 训练集在整个数据集中的比例
        toshuffle: 是否打乱数据集
        pred_mid: 如果使用biLSTM，此参数应设为True，否则为False
    """
    # 读取CSV文件
    df = pd.read_csv(filepath)
    in_size_raw = len(df.iloc[1, :])  # 获取数据集的列数
    if in_size_raw <= xcol[1] or in_size_raw <= xcol[0] or in_size_raw <= ycol[1] or in_size_raw <= ycol[0]:
        raise Exception("Invalid parameter col ranges: no so many cols in the data file")  # 检查列范围是否合法

    # 特征数量
    amount_of_features = xcol[1] - xcol[0] + 1
    out_size = ycol[1] - ycol[0] + 1
    nrows = len(df)  # 数据集的行数
    data = df.values  # 将数据转换为numpy数组
    x_dataset = []  # 存储训练特征的列表
    y_dataset = []  # 存储结果特征的列表

    # 如果预测中间时间点，则设置row_pred为序列长度的一半，否则为序列长度
    if pred_mid:
        row_pred = seq_len // 2
    else:
        row_pred = seq_len

    # 分离数据到x和y
    for nrow in range(nrows - seq_len):
        if not pred_mid:
            x_dataset.append(data[nrow: nrow + seq_len, xcol[0]:(xcol[1] + 1)])  # 添加seq_len行训练特征
            y_dataset.append(data[nrow + seq_len, ycol[0]:(ycol[1] + 1)])  # 添加预测目标行
        else:
            x_dataset.append(np.vstack((data[nrow: nrow + row_pred, xcol[0]:(xcol[1] + 1)],
                                        data[nrow + row_pred + 1: nrow + seq_len + 1,
                                        xcol[0]:(xcol[1] + 1)])))  # 添加训练特征
            y_dataset.append(data[nrow + row_pred, ycol[0]:(ycol[1] + 1)])  # 添加中间行预测目标

    # 转换为numpy数组
    x_dataset = np.array(x_dataset)
    y_dataset = np.array(y_dataset)

    # 划分训练集和测试集
    num_records = x_dataset.shape[0]  # 总记录数
    row_train = int(round(train_ratio * num_records))  # 训练集行数
    if toshuffle:
        # 如果需要打乱数据集
        shuffled_idx = np.arange(num_records)
        np.random.shuffle(shuffled_idx)  # 打乱索引

        x_train = np.array([x_dataset[idx, :] for idx in shuffled_idx[:row_train]])  # 获取训练特征
        y_train = np.array([y_dataset[idx, :] for idx in shuffled_idx[:row_train]])  # 获取训练目标
        x_test = np.array([x_dataset[idx, :] for idx in shuffled_idx[row_train:]])  # 获取测试特征
        y_test = np.array([y_dataset[idx, :] for idx in shuffled_idx[row_train:]])  # 获取测试目标
    else:
        # 不打乱数据集
        x_train = x_dataset[:row_train, :]
        y_train = y_dataset[:row_train, :]
        x_test = x_dataset[row_train:, :]
        y_test = y_dataset[row_train:, :]

    # 打印训练集的形状
    print(f"Shape of x_train: {x_train.shape}")
    print(f"Shape of y_train: {y_train.shape}")

    # 将数据重塑为 n行*m特征
    X_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], amount_of_features)).astype(np.float32)
    X_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], amount_of_features)).astype(np.float32)
    Y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1])).astype(np.float32)
    Y_test = np.reshape(y_test, (y_test.shape[0], y_test.shape[1])).astype(np.float32)

    return X_train, Y_train, X_test, Y_test  # 返回训练和测试数据集


#  定义损失函数

#  平均相对误差损失函数

class MSRELoss(nn.Module):
    """
    Mean SquareRoot Relative Error
    """

    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        return torch.sqrt(torch.mean(((y_pred - y_true) / (torch.fmax(torch.abs(y_true), torch.Tensor([1])))) ** 2))


# 自定义R2评价函数

def my_R2_score(x_pred, x_real):
    """
    self-defined r2_score,
    if elements in x_real are all equal, return square-root of mean (squre of relative error);
    else return normal r2_score
    """
    real_mean = torch.mean(x_real)
    ss_tot = torch.sum((x_real - real_mean) ** 2)
    if ss_tot >= 1e-10:
        ss_res = torch.sum((x_real - x_pred) ** 2)
        r2 = 1 - ss_res / ss_tot
    else:
        dif = x_pred - x_real
        den = torch.fmax(torch.abs(x_real), torch.abs(x_pred))
        den[den <= 0.001] = 1
        r2 = 1 - torch.sqrt(torch.mean((dif / den) ** 2))
    r2 = 0.5 if r2 < 0 else r2
    return r2


# 定义模型

# FFT神经网络层

# 定义一个简单的频域变换模块，这里使用离散傅里叶变换（DFT）
class FFTLayer(nn.Module):
    def __init__(self, n_features=None):
        super(FFTLayer, self).__init__()
        self.n_features = n_features

    def forward(self, x):
        # x 的形状是 (batch_size, seq_len, n_features)
        # 对每个特征进行DFT变换
        dft = fft.fft(x, n=self.n_features, dim=1, norm="forward")  # 1表示一维变换
        return dft


# 模型 CNN在空间方向上卷积, Transformer，时频结合

class CNN_BiLSTM_Transformer_TimeFreq(nn.Module):
    """
    class for time series data restore using CNN+BiLSTM+Transformer
    Parameters:
    input_dim: No. of features or input channels
    cnn_filters: No. of CNN output channels
    cnn_kernel_size: kernel size for CNN
    lstm_hidden_dim: output dim for lstm, for bilstm, real output size is lstm_hidden_dim*2
    transformer_dim: dim of transformer output
    transformer_heads: heads for transformer
    """

    def __init__(self, input_dim,
                 cnn_filters=64,
                 lstm_hidden_dim=128,
                 transformer_dim=256, transformer_heads=8,
                 output_dim=None,
                 cnn_kernel_size=3,
                 pool_size=3,
                 dropout=0):
        super(CNN_BiLSTM_Transformer_TimeFreq, self).__init__()

        self.dft = FFTLayer(input_dim)

        # CNN层
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=cnn_filters,
                      kernel_size=[cnn_kernel_size, 3], padding=[cnn_kernel_size // 2, 0]),
            #nn.BatchNorm2d(cnn_filters),  # 加入Batch Normalization
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=[pool_size, 1], stride=1, padding=[pool_size // 2, 0])
        )

        # BiLSTM层
        self.bilstm = nn.LSTM(cnn_filters, lstm_hidden_dim, bidirectional=True,
                              batch_first=True, num_layers=2,
                              dropout=dropout if dropout > 0 else 0,
                              device=device)
        #self.bilstm_bn = nn.BatchNorm1d(lstm_hidden_dim * 2)  # 加入Batch Normalization
        # Transformer层
        encoder_layer = nn.TransformerEncoderLayer(d_model=lstm_hidden_dim * 2, nhead=transformer_heads,
                                                   dim_feedforward=transformer_dim, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # 全连接层
        self.fc1 = nn.Linear(lstm_hidden_dim * 2, transformer_dim)
        self.fc2 = nn.Linear(transformer_dim, output_dim if output_dim is not None else input_dim)

    def forward(self, x):
        # print(f"Shape of x:{x.shape}")
        # x 是时域数据，形状是 (batch_size, seq_len, num_features)
        dft_x = self.dft(x)  # 频域变换
        # 将频域数据的实部和虚部与时域数据连接
        x_combined = torch.cat((x.unsqueeze(-1), torch.stack((dft_x.real, dft_x.imag), dim=-1)), dim=-1)
        # CNN层
        x = self.cnn(x_combined).squeeze(-1)  # (B, T, C) -> (B, C, T) for CNN
        # print(f"x shape after cnn:{x.shape}")
        x = x.permute(0, 2, 1)  # (B, C, T) -> (B, T, C) for LSTM
        # print(f"x shape after permute:{x.shape}")

        # BiLSTM层
        x, _ = self.bilstm(x)
        # print(f"x shape after bilstm:{x.shape}")

        # Transformer层
        x = self.transformer(x)
        # print(f"x shape after transformer:{x.shape}")

        # 全连接层
        x = F.relu(self.fc1(x))
        # print(f"x shape after fc1:{x.shape}")
        #x = x.mean(dim=-1)
        #print(f"x shape after mean:{x.shape}")
        x = self.fc2(x).squeeze(2)
        # print(f"x shape after fc:{x.shape}")

        return x


#  模型-对时频Transformer空间维度卷积训练函数

def train_spatio_transformer_timefreq(x_train, y_train, cnn_filters=64, lstm_hidden_dim=128,
                                      transformer_dim=256, transformer_heads=8,
                                      dropout=0.3, cnn_kernel_size=3, pool_size=2,
                                      batch_size=32, num_epochs=100):
    global prev_loss, time_loss_nochange
    prev_loss = 0
    time_loss_nochange = 0
    # print(type(x_train))
    # 将训练数据转为为pytorch的Tensor
    x_train = torch.from_numpy(x_train).to(device)
    y_train = torch.from_numpy(y_train).to(device)

    # 定义模型和优化器
    in_size = x_train.shape[1]
    model = CNN_BiLSTM_Transformer_TimeFreq(in_size, cnn_filters, lstm_hidden_dim,
                                            transformer_dim, transformer_heads, 1,  #in_size,
                                            cnn_kernel_size, pool_size, dropout).to(device)
    # print(f"in_size:{in_size}, cnn_filters{cnn_filters}, lstm_hidden_dim:{lstm_hidden_dim}, transformer_dim:{transformer_dim}, pool_size:{pool_size}")

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    lr_sched = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9, last_epoch=-1)
    criterion = nn.HuberLoss()

    losses_train = []

    # 开始训练

    losses_epoch = np.zeros((num_epochs, 1), dtype=float)
    start_time = time.time()
    start_time1 = start_time

    for epoch in range(num_epochs):
        loss_epoch_sum = 0
        cnt_epoch = 0
        for i in range(0, len(x_train), batch_size):
            batch_data = x_train[i: i + batch_size]
            real_y = y_train[i: (i + batch_size)]
            optimizer.zero_grad()
            pred_y = model(batch_data)
            # print(pred_y.shape, real_y.shape)
            loss = criterion(pred_y, real_y)
            loss.backward()
            optimizer.step()

            losses_train.append(loss.item())
            loss_epoch_sum += loss.item()
            cnt_epoch += 1

        lr_sched.step()

        curlossepoch = loss_epoch_sum / cnt_epoch if cnt_epoch > 0 else 0  # loss.item()
        losses_epoch[epoch] = curlossepoch
        end_time = time.time()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {curlossepoch:.8f}, Time: {(end_time - start_time) * 1000.0}ms,"
              f" Train count: {train_count}")
        start_time = end_time

        cur_loss_abs = abs(prev_loss - curlossepoch) / max(prev_loss, curlossepoch)
        if cur_loss_abs <= thr_loss_stop:
            time_loss_nochange = time_loss_nochange + 1
            if time_loss_nochange > time_to_stop:
                break
        else:
            time_loss_nochange = 0
        prev_loss = curlossepoch
    print("Training finished!")

    return model, losses_train, losses_epoch[:epoch + 1]


# 训练损失显示函数

def show_train_loss(losses):
    plt.figure(figsize=(10, 5), layout="tight")  # 设置图像大小
    plt.title("训练误差")  # 图像标题（中文）
    plt.semilogy(losses)  # 使用semilogy绘制损失值（对数刻度）
    plt.xlabel("epochs")  # X轴标签
    plt.ylabel("Loss")  # Y轴标签

    # 创建保存路径（如果不存在）
    save_dir = 'data/figure/'
    os.makedirs(save_dir, exist_ok=True)

    # 生成带有时间戳的文件名或使用指定的文件名

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'train_loss_{timestamp}.png'

    # 保存图像到指定路径

    plt.savefig(os.path.join(save_dir, filename))

    # 关闭当前图像，防止显示在屏幕上
    plt.close()


#  模型测试函数

def test(model, x_test, y_test, show_data=False):
    # 确保模型和数据都在同一设备上
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 将测试数据转为pytorch的Tensor
    x_test = torch.from_numpy(x_test).to(device)
    y_test = torch.from_numpy(y_test).to(device)

    criterion = nn.HuberLoss()

    losses_test = np.zeros((y_test.shape[0], 3))  # 增加一列用于存储相关系数
    y_preds = torch.zeros(y_test.shape).to(device)

    # 开始测试
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():  # 禁用梯度计算
        for idx, (x_real, y_real) in enumerate(zip(x_test, y_test)):
            y_pred = model(x_real.unsqueeze(0))
            y_preds[idx, :] = y_pred

            losses_test[idx, 0] = criterion(y_pred, y_real.unsqueeze(0)).item()

            y_real_np = y_real.cpu().numpy()
            y_pred_np = y_pred.squeeze(0).cpu().numpy()

            r2_score = my_R2_score(y_pred.squeeze(0), y_real)
            losses_test[idx, 1] = r2_score

            # 计算相关系数
            y_real_df = pd.Series(y_real_np).replace([np.inf, -np.inf], np.nan).fillna(0)
            y_pred_df = pd.Series(y_pred_np).replace([np.inf, -np.inf], np.nan).fillna(0)

            # 检查是否为常量序列
            if y_real_df.std() == 0 or y_pred_df.std() == 0:
                correlation = 0  # 如果是常量序列，设置相关系数为0
            else:
                correlation, _ = pearsonr(y_real_df, y_pred_df)  # 使用pearsonr计算相关系数

            losses_test[idx, 2] = correlation

    if show_data:
        plt.figure(figsize=(10, 10), layout="tight")
        plt.suptitle("Real vs. Predicted", fontsize=14)

        nrows = y_test.shape[1]
        for nidx in range(nrows):
            ax1 = plt.subplot(nrows, 1, nidx + 1)
            # 设置标题及其字号
            ax1.set_title(titles[nidx], fontsize=14)
            # 设置坐标轴刻度标签字号
            ax1.tick_params(axis='both', which='major', labelsize=14)
            # 设置X轴和Y轴标签及字号
            ax1.set_xlabel("samples", fontsize=14)
            ax1.set_ylabel("value", fontsize=14)

            ax1.plot(y_test[:, nidx].cpu().numpy(), 'b-', lw=2, label="Real")
            ax1.plot(y_preds[:, nidx].cpu().numpy(), 'r--', lw=2, label="Predicted")

        # 添加统一的图例
        handles, labels = ax1.get_legend_handles_labels()
        ax1.legend(handles, labels, loc='upper center', ncol=2, bbox_to_anchor=(0.5, -0.5), fontsize=14)

        plt.show()

    return losses_test, y_test, y_preds


def test_freq(model, x_test, y_test, sampling_rate=128, show_data=False):
    # 确保模型和数据都在同一设备上
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 将测试数据转为pytorch的Tensor
    x_test = torch.from_numpy(x_test).to(device)
    y_test = torch.from_numpy(y_test).to(device)
    y_preds = torch.zeros(y_test.shape).to(device)

    # 开始测试
    for idx, (x_real, y_real) in enumerate(zip(x_test, y_test)):
        y_pred = model(x_real.unsqueeze(0))
        y_preds[idx, :] = y_pred.detach()  # 使用 detach 分离计算图

    if show_data:
        plt.figure(figsize=(10, 10), layout="tight")
        plt.suptitle("Frequency Domain: Real vs. Predicted", fontsize=18)

        nrows = y_test.shape[1]
        for nidx in range(nrows):
            ax1 = plt.subplot(nrows, 1, nidx + 1)

            # 设置每个子图的标题为指定的字符串
            ax1.set_title(titles[nidx], fontsize=14)

            # 设置坐标轴刻度标签字号
            ax1.tick_params(axis='both', which='major', labelsize=14)

            # 设置X轴和Y轴标签及字号
            ax1.set_xlabel("Frequency (Hz)", fontsize=14)
            ax1.set_ylabel("Magnitude", fontsize=14)

            # 真实数据和预测数据的频域转换
            real_freq_data = torch.fft.fft(y_test[:, nidx])
            pred_freq_data = torch.fft.fft(y_preds[:, nidx])

            # 计算频率轴
            freq = torch.fft.fftfreq(y_test.shape[0], d=1 / sampling_rate)

            # 计算幅值
            real_freq_magnitude = torch.abs(real_freq_data)
            pred_freq_magnitude = torch.abs(pred_freq_data)

            # 只保留正频率部分
            pos_mask = freq >= 0
            freq_pos = freq[pos_mask]
            real_freq_magnitude_pos = real_freq_magnitude[pos_mask]
            pred_freq_magnitude_pos = pred_freq_magnitude[pos_mask]

            ax1.plot(freq_pos.cpu().numpy(), real_freq_magnitude_pos.cpu().numpy(), 'b-', lw=2, label="Real")
            ax1.plot(freq_pos.cpu().numpy(), pred_freq_magnitude_pos.cpu().numpy(), 'r--', lw=2, label="Predicted")

            ax1.set_yscale('log')
            # 添加统一的图例
        handles, labels = ax1.get_legend_handles_labels()
        ax1.legend(handles, labels, loc='upper center', ncol=2, bbox_to_anchor=(0.5, -0.5), fontsize=14)

        plt.show()

    return y_test, y_preds


#  测试损失显示函数


def show_test_loss(losses):
    # print(losses)
    print(f"{labels}: {np.mean(losses, 0)}")
    plt.figure(figsize=(10, 8), layout="tight")
    plt.suptitle("Discriminator Loss During test")
    num_losses = losses.shape[1]
    for idx in range(num_losses):
        ax = plt.subplot(num_losses, 1, idx + 1)
        # 设置每个子图的标题为指定的字符串
        ax.set_title(labels[idx], fontsize=14)

        # 设置坐标轴刻度标签字号
        ax.tick_params(axis='both', which='major', labelsize=14)

        # 设置X轴和Y轴标签及字号
        ax.set_xlabel("samples", fontsize=14)

        ax.semilogy(losses[:, idx])

    # plt.legend()
    plt.show()


# 约束检查函数
def check_constraints(lstm_hidden_dim, transformer_dim, transformer_heads):
    if transformer_dim % transformer_heads != 0:
        return False
    return True


# 适应度函数

def initialize_particles(num_particles, bounds):
    particles = np.zeros((num_particles, len(bounds)))
    num_dims = len(bounds)

    for i in range(num_dims):
        min_bound, max_bound = bounds[i]
        particles[:, i] = np.linspace(min_bound, max_bound, num_particles)

    return particles


def evaluate_model(params, iteration):
    global train_count, effective_particles, num_particles, num_iterations, g_best_score
    train_count += 1
    effective_particles = np.sum(is_active)
    print(f"Iteration {iteration + 1}/{num_iterations}, "
          f"Effective particles: {effective_particles}/{num_particles}, "
          f"Global best score: {g_best_score:.4f}, ", f"Train count: {train_count}")

    cnn_filters, lstm_hidden_dim, transformer_heads = params
    cnn_filters = int(cnn_filters)
    lstm_hidden_dim = int(lstm_hidden_dim)
    transformer_heads = int(transformer_heads)
    transformer_dim = 2 * lstm_hidden_dim

    if not check_constraints(lstm_hidden_dim, transformer_dim, transformer_heads):
        return float('inf')  # 违反约束，返回一个很大的值

    # 重新训练模型并计算损失
    model, _, _ = train_spatio_transformer_timefreq(x_train, y_train, cnn_filters=cnn_filters,
                                                    lstm_hidden_dim=lstm_hidden_dim,
                                                    transformer_dim=transformer_dim,
                                                    transformer_heads=transformer_heads,
                                                    dropout=0.8, cnn_kernel_size=3, pool_size=3, batch_size=1024,
                                                    num_epochs=500)  # 可以减少训练轮数，加快评估速度
    losses_test, _, _ = test(model, x_test, y_test)
    avg_loss = np.mean(losses_test[:, 0])  # 使用HuberLoss作为评估标准，改为R2或CoRR要改正负号

    return avg_loss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 配置 matplotlib 后端
plt.switch_backend('TkAgg')  # 'Agg' 后端适用于无GUI环境，如服务器；如在桌面环境，可以使用 'TkAgg', 'Qt5Agg' 等
# 使用默认样式
plt.style.use('default')

# 支持中文
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams['image.cmap'] = 'jet'  # 图示色标
labels = ["HuberLoss", "R2", "CORR"]
titles = ["site002", "site004", "site006", "site009"]
# 迭代控制参数
# 用来控制迭代的停止

prev_loss = 1e10  # previous loss
thr_loss_stop = 0.001  # threshold to judge whether the loss changes or not
time_loss_nochange = 0  # how many times that loss change is less than thr_loss_stop
time_to_stop = 10  # if time_loss_nochange >= time_to_stop, stop the iteration
train_count = 0

# 制作数据集

# Actual_5_Min_merged.csv：某一地区多个电场光电数据，5分钟一个数据
x_train, y_train, x_test, y_test = make_dataset("data/Ex.csv", [0, 3],
                                                [0, 3], seq_len=512, train_ratio=0.8, toshuffle=False, pred_mid=True)

# 定义参数范围
lstm_hidden_dim_min = 1
lstm_hidden_dim_max = 512
transformer_heads_min = 1
transformer_heads_max = 64
cnn_filters_min = 64
cnn_filters_max = 512

# 初始化粒子群
num_particles = 10  # 粒子数量
num_iterations = 30  # 迭代次数，即粒子群在搜索空间中移动的总次数
particles = np.zeros((num_particles, 3), dtype=np.float64)
is_active = np.ones(num_particles, dtype=bool)
# 初始化存储粒子的列表
particles = []

# 在 cnn_filters 上生成均匀分布的粒子
cnn_filters_values = np.linspace(cnn_filters_min, cnn_filters_max, num_particles, dtype=np.float64).astype(int)

# 生成粒子并确保其他维度满足约束条件
for cnn_filters in cnn_filters_values:
    while len(particles) < num_particles:
        lstm_hidden_dim = np.random.randint(lstm_hidden_dim_min, lstm_hidden_dim_max + 1)
        transformer_heads = np.random.randint(transformer_heads_min, transformer_heads_max + 1)
        transformer_dim = 2 * lstm_hidden_dim

        # 检查两个约束条件
        if transformer_heads <= transformer_dim and transformer_dim % transformer_heads == 0:
            particles.append([float(cnn_filters), float(lstm_hidden_dim), float(transformer_heads)])
            break

# 打印生成的粒子
particles = np.array(particles)

# 打印生成的粒子，去掉小数点


# 打印生成的粒子，使用逗号分隔
particles_str = np.array2string(np.floor(particles).astype(int), separator=', ')
print(particles_str)

print(f"Total particles generated: {len(particles)}")

# PSO 参数
w_max = 0.9  # 初始惯性权重
w_min = 0.4  # 最小惯性权重
c1 = 1.5  # 自我认知系数
c2 = 1.5  # 社会认知系数

# 记录每个粒子的移动路径
cnn_filters_history = [[] for _ in range(num_particles)]
lstm_hidden_dim_history = [[] for _ in range(num_particles)]
transformer_heads_history = [[] for _ in range(num_particles)]

# 初始化淘汰机制相关变量
effective_particle_history = []  # 记录每次迭代后的有效粒子数量
no_improvement_counts = np.zeros(num_particles, dtype=int)
max_no_improvement = 10  # 允许的最大无改进次数

# 初始化速度和最佳位置
velocities = np.random.uniform(-1, 1, (num_particles, 3))
p_best_positions = particles.copy()
p_best_scores = np.full(num_particles, float('inf'))  # 使用inf初始化

g_best_score = float('inf')
# 计算初始适应度
for i in range(num_particles):
    p_best_scores[i] = evaluate_model(particles[i], iteration=0)

# 初始化全局最佳

g_best_position = particles[np.argmin(p_best_scores)]
g_best_score = np.min(p_best_scores)

# 主循环中的代码
for iteration in range(num_iterations):
    # 线性衰减惯性权重
    w = w_max - ((w_max - w_min) / num_iterations) * iteration

    for i in range(num_particles):
        if not is_active[i]:  # 如果粒子已经被淘汰，跳过该粒子
            continue
        # 记录粒子当前位置
        cnn_filters_history[i].append(particles[i][0])
        lstm_hidden_dim_history[i].append(particles[i][1])
        transformer_heads_history[i].append(particles[i][2])

        # 更新速度
        velocities[i] = (
                w * velocities[i]
                + c1 * np.random.rand() * (p_best_positions[i] - particles[i])
                + c2 * np.random.rand() * (g_best_position - particles[i])
        )
        # 更新位置
        particles[i] += velocities[i]
        particles[i] = np.clip(particles[i],
                               [cnn_filters_min, lstm_hidden_dim_min, transformer_heads_min],
                               [cnn_filters_max, lstm_hidden_dim_max, transformer_heads_max])

        # 重新检查约束，确保粒子的参数满足模型要求
        cnn_filters, lstm_hidden_dim, transformer_heads = particles[i].astype(int)
        transformer_dim = 2 * lstm_hidden_dim
        while not check_constraints(lstm_hidden_dim, transformer_dim, transformer_heads):
            transformer_heads = np.random.randint(transformer_heads_min, transformer_heads_max + 1)
        particles[i] = [cnn_filters, lstm_hidden_dim, transformer_heads]

        # 计算适应度
        score = evaluate_model(particles[i], iteration)

        if score < p_best_scores[i]:
            p_best_positions[i] = particles[i]
            p_best_scores[i] = score
            no_improvement_counts[i] = 0
        else:
            no_improvement_counts[i] += 1

        if score < g_best_score:
            g_best_position = particles[i]
            g_best_score = score

        # 处理粒子淘汰
        if no_improvement_counts[i] >= max_no_improvement:
            print(f"Particle {i} is eliminated due to no improvement after {no_improvement_counts[i]} iterations.")
            # 重新初始化被淘汰的粒子位置
            is_active[i] = False  # 标记该粒子为无效
            no_improvement_counts[i] = 0

    # 计算有效粒子数
    effective_particles = np.sum(is_active)
    print(f"Iteration {iteration + 1}/{num_iterations}, "
          f"Effective particles: {effective_particles}/{num_particles}, "
          f"Global best score: {g_best_score:.4f}")

    # 记录有效粒子数
    effective_particle_history.append(effective_particles)

# 最优结果
best_cnn_filters, best_lstm_hidden_dim, best_transformer_heads = g_best_position.astype(int)
best_transformer_dim = 2 * best_lstm_hidden_dim

print(
    f"Best CNN Filters: {best_cnn_filters}, Best LSTM Hidden Dim: {best_lstm_hidden_dim}, "
    f"Best Transformer Heads: {best_transformer_heads}, Best Transformer Dim: {best_transformer_dim}")

# 绘制有效粒子数随迭代次数的变化图
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_iterations+1), effective_particle_history, marker='o', color='b')
plt.title("Number of Effective Particles Over Iterations")
plt.xlabel("Iteration")
plt.ylabel("Number of Effective Particles")
plt.grid(True)
plt.show()

# 绘制三维图像
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

for i in range(num_particles):
    ax.plot(cnn_filters_history[i], lstm_hidden_dim_history[i], transformer_heads_history[i],
            marker='o', label=f'Particle {i + 1}')

ax.set_xlabel('CNN Filters')
ax.set_ylabel('LSTM Hidden Dim')
ax.set_zlabel('Transformer Heads')
ax.set_title('Particle Movement in 3D Search Space')
ax.legend()
plt.show()

# 模型训练
# Transformer Time-Frequency
# 训练


prev_loss = 0
time_loss_nochange = 0

# 使用优化后的cnn_filters重新训练模型

model_trained_tr_tf, _, losses_epoch_tr_tf = train_spatio_transformer_timefreq(
    x_train, y_train,
    cnn_filters=int(best_cnn_filters),
    lstm_hidden_dim=int(best_lstm_hidden_dim),
    transformer_dim=int(best_transformer_dim),
    transformer_heads=int(best_transformer_heads),
    dropout=0.8,
    cnn_kernel_size=3,
    pool_size=3,
    batch_size=1024,
    num_epochs=500
)
# 显示训练损失曲线

show_train_loss(losses_epoch_tr_tf)

# 模型测试


# CNN+LSTM+Transformer Time-Frequency

# 训练集

print("testing, please wait...")
losses_train_tr_tf, _, _ = test(model_trained_tr_tf, x_train[128::], y_train[128::], show_data=True)

show_test_loss(losses_train_tr_tf)

test_freq(model_trained_tr_tf, x_train, y_train, sampling_rate=128, show_data=True)

# 测试集

print("testing, please wait...")
losses_test_tr_tf, _, _ = test(model_trained_tr_tf, x_test[128::], y_test[128::], show_data=True)

show_test_loss(losses_test_tr_tf)

test_freq(model_trained_tr_tf, x_test, y_test, sampling_rate=128, show_data=True)
