# -*- coding: utf-8 -*-
"""
DPNet —— 双通道融合深度攻击分类网络
=====================================================================
架构：
    输入: 流量特征序列 X ∈ R^{B x T x F}（B 批大小，T 时间步，F 特征数）

    通道 1 (空间通道, CNN)：将序列展平为一维“信号”，3 层
        Conv1d + BatchNorm + ReLU + MaxPooling，提取局部/空间特征；
    通道 2 (时序通道, BiLSTM)：3 层双向 LSTM（隐藏层 128），
        提取前向 + 后向的长程时序依赖；

    融合：两通道 Flatten 后拼接 -> 2 层 Dense -> Softmax 多分类
        （BENIGN / DoS / DDoS / PortScan / BruteForce / FDIA ...）

需要 PyTorch。
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:
    raise ImportError(
        "DPNet 需要 PyTorch，请先安装（建议阿里云镜像）：\n"
        "pip install torch -i https://mirrors.aliyun.com/pypi/simple/"
    ) from exc

from .preprocessing import get_device


# =====================================================================
# 模型定义
# =====================================================================
class DPNet(nn.Module):
    r"""
    Dual-Channel Parallel Convolutional-Recurrent Network (DPNet)。

    -----------------------------------------------------------------
    通道 1：CNN（空间/局部形态特征）
    -----------------------------------------------------------------
    将 (B,T,F) 展平成单通道一维信号 (B, 1, L), L = T*F。
    每一层卷积执行：
        Y = phi( W (*) X + b )          # (*) 一维互相关，phi=ReLU
        Y = MaxPool(Y, kernel=2)        # 长度减半
    三层通道数 32 -> 64 -> 128，卷积核 k=3，padding=1 保持长度；
    末端 AdaptiveAvgPool1d(pool_len) 固定输出长度，再 Flatten：
        z_cnn ∈ R^{B x (128 * pool_len)}

    -----------------------------------------------------------------
    通道 2：3 层 BiLSTM（时序依赖特征）
    -----------------------------------------------------------------
    单层 LSTM 的门控方程（x_t 输入，h_{t-1} 隐状态，C 记忆细胞）：
        f_t = sigmoid(W_f [h_{t-1}, x_t] + b_f)       # 遗忘门
        i_t = sigmoid(W_i [h_{t-1}, x_t] + b_i)       # 输入门
        g_t = tanh(W_C [h_{t-1}, x_t] + b_C)          # 候选记忆
        C_t = f_t * C_{t-1} + i_t * g_t               # 细胞更新
        o_t = sigmoid(W_o [h_{t-1}, x_t] + b_o)       # 输出门
        h_t = o_t * tanh(C_t)
    双向结构在每个时间步同时维护正向 h_t^f 与反向 h_t^b：
        h_t = [overrightarrow h_t ; overleftarrow h_t]
    3 层堆叠（层间 dropout），hidden=128，取最后一层的终态：
        z_rnn = [ h_T^{f,layer3} ; h_1^{b,layer3} ] ∈ R^{B x 256}

    -----------------------------------------------------------------
    融合与输出
    -----------------------------------------------------------------
        z = [z_cnn ; z_rnn]
        h1 = ReLU( W_1 z + b_1 )          # Dense 第 1 层
        p  = softmax( W_2 h1 + b_2 )      # Dense 第 2 层 + Softmax
        p_c = exp(a_c) / sum_{c'} exp(a_{c'})   # 多类概率
    """

    def __init__(self, n_features, seq_len, num_classes,
                 cnn_channels=(32, 64, 128),
                 cnn_kernel_size=3, pool_len=8,
                 lstm_hidden=128, lstm_layers=3,
                 dense_units=256, dropout=0.3):
        """
        Parameters
        ----------
        n_features : F，每个时间步的特征数
        seq_len    : T，时间窗长度
        num_classes: 攻击类别数（含 BENIGN）
        pool_len   : CNN 末端自适应池化长度（使模型对 T 鲁棒）
        lstm_hidden: BiLSTM 单向隐藏维数（双向输出 2*hidden）
        """
        super().__init__()
        self.n_features = n_features
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.lstm_hidden = lstm_hidden

        # ================= 通道 1：CNN =================
        c1, c2, c3 = cnn_channels
        pad = cnn_kernel_size // 2
        # Conv1d 输入通道=1；MaxPool1d(2) 每次长度减半
        self.conv1 = nn.Conv1d(1, c1, cnn_kernel_size, padding=pad)
        self.bn1 = nn.BatchNorm1d(c1)
        self.conv2 = nn.Conv1d(c1, c2, cnn_kernel_size, padding=pad)
        self.bn2 = nn.BatchNorm1d(c2)
        self.conv3 = nn.Conv1d(c2, c3, cnn_kernel_size, padding=pad)
        self.bn3 = nn.BatchNorm1d(c3)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.adapt_pool = nn.AdaptiveAvgPool1d(pool_len)   # -> (B,c3,pool_len)
        self.cnn_out_dim = c3 * pool_len

        # ================= 通道 2：3 层 BiLSTM =================
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.rnn_out_dim = 2 * lstm_hidden

        # ================= 融合 + 2 层 Dense =================
        fused_dim = self.cnn_out_dim + self.rnn_out_dim
        self.fc1 = nn.Linear(fused_dim, dense_units)
        self.fc2 = nn.Linear(dense_units, num_classes)
        self.dropout = nn.Dropout(dropout)

    # -----------------------------------------------------------------
    # 前向传播（逐步注释张量形状）
    # -----------------------------------------------------------------
    def forward(self, x):
        """
        Parameters
        ----------
        x : Tensor (B, T, F)

        Returns
        -------
        probs : (B, num_classes)，各类别概率，行和为 1
        """
        B = x.size(0)

        # ============ 通道 1：CNN 分支 ============
        # (B,T,F) -> (B, T*F) -> (B, 1, T*F)：视作单通道一维信号
        x_cnn = x.reshape(B, 1, -1)

        # Conv block 1: (B,1,L) -> (B,32,L/2)
        x_cnn = self.pool(F.relu(self.bn1(self.conv1(x_cnn))))
        # Conv block 2: (B,32,L/2) -> (B,64,L/4)
        x_cnn = self.pool(F.relu(self.bn2(self.conv2(x_cnn))))
        # Conv block 3: (B,64,L/4) -> (B,128,L/8)
        x_cnn = self.pool(F.relu(self.bn3(self.conv3(x_cnn))))
        # 自适应池化到固定长度 -> (B,128,pool_len) -> Flatten
        x_cnn = self.adapt_pool(x_cnn)
        z_cnn = x_cnn.reshape(B, -1)                       # (B,1024)

        # ============ 通道 2：BiLSTM 分支 ============
        # out: (B,T,2H) 全部时间步隐状态
        # h_n: (2*num_layers, B, H) 各层终态
        out, (h_n, c_n) = self.lstm(x)
        # 取最后一层：正向终态 h_n[2*(L-1)] 与反向终态 h_n[2*(L-1)+1]
        h_forward = h_n[2 * (self.lstm.num_layers - 1)]      # (B,H)
        h_backward = h_n[2 * (self.lstm.num_layers - 1) + 1]  # (B,H)
        z_rnn = torch.cat([h_forward, h_backward], dim=1)   # (B,256)

        # ============ 融合 ============
        z = torch.cat([z_cnn, z_rnn], dim=1)               # (B,1024+256)

        # ============ 2 层 Dense + Softmax ============
        h1 = F.relu(self.fc1(z))                            # (B,dense_units)
        h1 = self.dropout(h1)
        logits = self.fc2(h1)                               # (B,C)
        probs = F.softmax(logits, dim=1)                    # (B,C)
        return probs


# =====================================================================
# 训练 / 评估辅助函数
# =====================================================================
def train_epoch(model, loader, optimizer, criterion=None, device=None):
    """
    训练一个 epoch。

    Parameters
    ----------
    loader    : DataLoader，产出 (X, y)，X 形状 (B,T,F)
    criterion : 损失函数，默认多分类交叉熵（注意：模型输出为 softmax
                概率，故使用 NLLLoss(log(probs), y)）
    Returns
    -------
    mean_loss, accuracy
    """
    device = device or get_device()
    model.to(device).train()
    criterion = criterion or nn.NLLLoss()
    total_loss, correct, total = 0.0, 0, 0

    for X, y in loader:
        X = X.float().to(device)
        y = y.long().to(device)

        probs = model(X)
        loss = criterion(torch.log(probs + 1e-12), y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)
        correct += (probs.argmax(dim=1) == y).sum().item()
        total += X.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device=None):
    """在验证/测试集上评估，返回 (accuracy, y_true, y_pred, y_prob)。"""
    device = device or get_device()
    model.to(device).eval()
    ys, preds, probs_all = [], [], []
    for X, y in loader:
        X = X.float().to(device)
        p = model(X).cpu().numpy()
        probs_all.append(p)
        preds.append(p.argmax(axis=1))
        ys.append(np.asarray(y))
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(preds)
    y_prob = np.vstack(probs_all)
    acc = float((y_true == y_pred).mean())
    return acc, y_true, y_pred, y_prob
