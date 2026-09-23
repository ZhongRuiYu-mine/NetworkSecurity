# -*- coding: utf-8 -*-
"""
数据预处理模块
=====================================================================
1. Standardizer      : 标准化（仅用训练集统计量，杜绝数据泄漏）
2. ADASYN            : 自适应合成少数类过采样（处理 CIC-IDS2017 等
                       攻击/正常样本极度不平衡问题）
3. DenoisingAutoencoder : 降噪自编码器（PyTorch），用于
                       特征降维 + 去噪；重构误差亦可作为无监督异常分数

依赖说明：numpy 为独立硬依赖；torch 为可选依赖（仅自编码器需要），
未安装 torch 时 ADASYN/Standardizer 仍可正常使用。
"""

import numpy as np
from scipy.spatial import cKDTree

EPS = 1e-8


def get_device(prefer_gpu=True, gpu_mem_fraction=0.8):
    """
    选择计算设备。

    - 有 CUDA 时限制单进程显存占用比例（默认 80%），避免占满显存
      影响计算机正常使用（RTX 5060 Ti / Blackwell）。
    - 无 torch 或无 GPU 时退回 CPU。
    """
    try:
        import torch
    except ImportError:
        return "cpu"
    if prefer_gpu and torch.cuda.is_available():
        try:
            torch.cuda.set_per_process_memory_fraction(gpu_mem_fraction, 0)
        except Exception:
            pass  # 某些驱动版本不支持该接口，忽略即可
        return torch.device("cuda")
    return torch.device("cpu")


def _require_torch():
    try:
        import torch
        import torch.nn as nn
        return torch, nn
    except ImportError as exc:
        raise ImportError(
            "该功能需要 PyTorch，请先安装（建议使用阿里云镜像）。\n"
            "pip install torch -i https://mirrors.aliyun.com/pypi/simple/"
        ) from exc


# =====================================================================
# 标准化（必须在训练/测试切分之后 fit，避免统计量泄漏）
# =====================================================================
class Standardizer:
    r"""
    Z-Score 标准化：

        x' = (x - mu) / sigma,    mu, sigma 仅由训练集估计。

    对常量列（sigma≈0）置 sigma=1，避免除零。
    """

    def __init__(self):
        self.mu_ = None
        self.sigma_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mu_ = X.mean(axis=0)
        self.sigma_ = X.std(axis=0, ddof=0)
        self.sigma_[self.sigma_ < EPS] = 1.0
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=float) - self.mu_) / self.sigma_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


# =====================================================================
# ADASYN —— Adaptive Synthetic Sampling
# =====================================================================
class ADASYN:
    r"""
    ADASYN 自适应合成采样（He et al., 2009）。

    与 SMOTE 对所有少数类样本“一视同仁”不同，ADASYN 让越靠近
    决策边界（周围多数类邻居比例越高）的少数类样本生成越多合成样本，
    从而自适应地强化难分区域。

    算法步骤：
    1) 记少数类样本数 m_s，多数类样本数 m_l，需合成的总量
           G = (m_l - m_s) * beta,    beta ∈ [0,1]
       beta=1 表示采样后两类完全平衡。

    2) 对每个少数类样本 x_i，在全体数据中找 K 个近邻，计算其中
       多数类占比（越靠近边界越高）：
           r_i = Delta_i / K,    Delta_i = 近邻中多数类个数

    3) 归一化为生成权重：
           r_hat_i = r_i / sum_j r_j

    4) 每个少数类样本需生成的合成样本数：
           g_i = round(r_hat_i * G)

    5) 从 x_i 的少数类 K 近邻中随机抽取 x_zi，线性插值合成：
           s = x_i + lambda * (x_zi - x_i),   lambda ~ U(0,1)
    """

    def __init__(self, beta=1.0, k_neighbors=5, random_state=None):
        """
        Parameters
        ----------
        beta        : 平衡目标，1.0 完全平衡，<1 部分平衡
        k_neighbors : 近邻数 K
        """
        self.beta = beta
        self.k = k_neighbors
        self.rng = np.random.default_rng(random_state)

    def fit_resample(self, X, y):
        """
        Parameters
        ----------
        X : (N, d) 特征
        y : (N,)   标签（二分类；少数类自动识别为样本数最少的类）

        Returns
        -------
        X_new, y_new : 平衡后的特征与标签
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).reshape(-1)

        classes, counts = np.unique(y, return_counts=True)
        if classes.size < 2:
            return X, y
        minority_cls = classes[np.argmin(counts)]
        majority_cls = classes[np.argmax(counts)]
        X_min = X[y == minority_cls]
        X_maj = X[y == majority_cls]
        m_s, m_l = X_min.shape[0], X_maj.shape[0]

        # 目标合成总量
        G = int(round((m_l - m_s) * self.beta))
        if G <= 0:
            return X, y

        K = min(self.k, len(X) - 1)
        # --- 步骤 2：在全体数据上查 K 近邻，统计多数类占比 r_i ---
        tree_all = cKDTree(X)
        tree_min = cKDTree(X_min)
        # k=K+1：第 1 个是自身
        _, idx_all = tree_all.query(X_min, k=K + 1)
        idx_all = np.atleast_2d(idx_all)[:, 1:]

        is_majority = (y[idx_all] == majority_cls)
        r = is_majority.sum(axis=1) / K

        # 所有少数样本都被多数类包围程度极低（如完全可分）时退化为均匀加权
        if r.sum() < EPS:
            r = np.ones(m_s)
        r_hat = r / r.sum()
        g = np.floor(r_hat * G).astype(int)
        # 舍入误差导致的数量缺口，补给权重最大的样本
        deficit = G - g.sum()
        if deficit > 0:
            order = np.argsort(-r_hat)
            g[order[:deficit]] += 1

        # --- 步骤 5：在少数类内部查近邻并插值合成 ---
        K_min = min(self.k, m_s - 1)
        synthetic = []
        if K_min >= 1:
            _, idx_min = tree_min.query(X_min, k=K_min + 1)
            idx_min = np.atleast_2d(idx_min)[:, 1:]
            for i in range(m_s):
                if g[i] == 0:
                    continue
                chosen = self.rng.choice(idx_min[i], size=g[i], replace=True)
                lam = self.rng.random((g[i], 1))
                syn = X_min[i] + lam * (X_min[chosen] - X_min[i])
                synthetic.append(syn)

        if not synthetic:
            return X, y
        X_syn = np.vstack(synthetic)
        y_syn = np.full(X_syn.shape[0], minority_cls, dtype=y.dtype)
        return np.vstack([X, X_syn]), np.concatenate([y, y_syn])


# =====================================================================
# 降噪自编码器（PyTorch）
# =====================================================================
class _AENet:
    """延迟构建的内部网络（仅在实例化自编码器时才要求 torch）。"""


class DenoisingAutoencoder:
    r"""
    全连接降噪自编码器。

    编码（Enc）：
        h = sigma_e( W_e x_tilde + b_e ),   x_tilde = x dropout/加噪后的损坏输入
        z = W_b h + b_b                       # bottleneck 潜在表示 z

    解码（Dec）：
        x_hat = sigma_d( W_d z + b_d )

    训练目标（最小化重构 MSE）：
        L(theta) = (1/N) sum_i || x_i - Dec(Enc(x_tilde_i)) ||_2^2

    用途：
    - encode()   : 输出低维潜在特征，供后续 CSEKM/DPNet 使用；
    - reconstruction_error() : 作为无监督异常分数
      （攻击样本偏离正常流形，重构误差更大）。

    【关键口径】为避免“把攻击也学会重构”，fit 时应只传入正常样本
    （X_train_normal）；标准化统计量也只由正常训练集估计。
    """

    def __init__(self, input_dim, latent_dim=8, hidden_dims=(32, 16),
                 noise_factor=0.1, lr=1e-3, epochs=100,
                 batch_size=128, weight_decay=1e-5,
                 device=None, random_state=42):
        """
        Parameters
        ----------
        input_dim   : 输入特征维度
        latent_dim  : bottleneck 维度（降维后的特征数）
        hidden_dims : 编码器隐藏层宽度（解码器对称镜像）
        noise_factor: 输入高斯噪声强度（降噪 AE 的损坏比例）
        """
        torch, nn = _require_torch()
        self.torch = torch
        self.nn = nn
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.noise_factor = noise_factor
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.device = device if device is not None else get_device()

        torch.manual_seed(random_state)
        np.random.seed(random_state)

        # ---- 构建编码器 ----
        enc_layers = []
        prev = input_dim
        for h in hidden_dims:
            enc_layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        enc_layers += [nn.Linear(prev, latent_dim)]
        self.encoder = nn.Sequential(*enc_layers)

        # ---- 构建解码器（对称） ----
        dec_layers = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            dec_layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        dec_layers += [nn.Linear(prev, input_dim)]
        self.decoder = nn.Sequential(*dec_layers)

        self._net = nn.Module()
        self._net.encoder = self.encoder
        self._net.decoder = self.decoder
        self._net.to(self.device)

        self.optimizer = torch.optim.Adam(self._net.parameters(),
                                          lr=lr, weight_decay=weight_decay)
        self.loss_fn = nn.MSELoss()
        self.scaler_ = Standardizer()
        self.loss_history_ = []

    def _to_tensor(self, X):
        return self.torch.as_tensor(np.asarray(X, dtype=np.float32),
                                    device=self.device)

    def fit(self, X, verbose=False):
        """
        训练自编码器。

        重要：请只传入正常样本 X_normal，且 X 应为切分后的训练集
        （内部会基于该数据 fit 标准化器，避免泄漏验证/测试信息）。
        """
        X = np.asarray(X, dtype=float)
        Xs = self.scaler_.fit_transform(X).astype(np.float32)
        n = Xs.shape[0]

        self._net.train()
        for ep in range(self.epochs):
            perm = np.random.permutation(n)
            epoch_loss = 0.0
            for s in range(0, n, self.batch_size):
                idx = perm[s:s + self.batch_size]
                xb = self._to_tensor(Xs[idx])
                # 降噪损坏：x_tilde = x + noise_factor * N(0,1)
                x_tilde = xb + self.noise_factor * self.torch.randn_like(xb)

                z = self.encoder(x_tilde)
                x_hat = self.decoder(z)
                loss = self.loss_fn(x_hat, xb)   # 目标仍是干净 x

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * xb.size(0)

            epoch_loss /= n
            self.loss_history_.append(epoch_loss)
            if verbose and (ep % max(1, self.epochs // 10) == 0):
                print(f"[AE] epoch {ep:4d}  recon MSE = {epoch_loss:.6f}")
        return self

    def encode(self, X):
        """输出降维后的潜在特征 z（numpy）。"""
        Xs = self.scaler_.transform(X).astype(np.float32)
        self._net.eval()
        with self.torch.no_grad():
            return self.encoder(self._to_tensor(Xs)).cpu().numpy()

    def reconstruct(self, X):
        """返回重构数据（已映射回原始量纲）。"""
        Xs = self.scaler_.transform(X).astype(np.float32)
        self._net.eval()
        with self.torch.no_grad():
            rec = self.decoder(self.encoder(self._to_tensor(Xs)))
        return rec.cpu().numpy() * self.scaler_.sigma_ + self.scaler_.mu_

    def reconstruction_error(self, X):
        """
        逐样本重构误差（无监督异常分数）：

            score_i = || x_i - x_hat_i ||_2^2
        """
        X = np.asarray(X, dtype=float)
        rec = self.reconstruct(X)
        return ((X - rec) ** 2).sum(axis=1)
