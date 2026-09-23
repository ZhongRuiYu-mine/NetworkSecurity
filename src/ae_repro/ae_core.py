# -*- coding: utf-8 -*-
"""
ae_core.py —— 论文自编码器检测机制的纯 numpy 复现核心。

设计原则
--------
本文件**不 import 平台仓库的任何模块**，因此可以：
  * 独立跑实验（本包 experiments.py）；
  * 被 ae_detector.py 包装成 `DetectorBase` 贴进仓库；
  * 被任何第三方脚本直接调用。

复现依据（论文 AUTO.pdf，第 1 阶段 Autoencoder）
-----------------------------------------------
| 论文里的说法                                        | 本实现 |
|-----------------------------------------------------|--------|
| encoder: hidden layers with **decreasing** neuron counts | `hidden_dims=(32,16,8)`，逐层递减 |
| symmetric layers for the decoder                    | 解码器为编码器镜像；额外可切到 B 论文的**非对称**解码器 |
| 训练目标 `L = (1/n) Σ ||x_i - x̂_i||²`（MSE）         | `loss = "mse"`，等价于论文式(3) |
| 异常分 `S_i^(AE) = ||x_i - x̂_i||²`                  | **逐特征标准化**后的平方残差均值（见下） |
| 连续特征 min–max 归一化到 [0,1]                      | 数据侧由平台 data_loader 完成（Eq.1）；核心内部再做一次 z-score（B 论文做法） |
| 决策阈值 0.52（表 1 中 AE 的 Threshold 列）           | `threshold` 可显式指定；另提供分位数与 EVT 两种标定 |
| 在**正常样本**上训练                                  | `fit()` 只接受良性流量 |

为什么是"逐特征标准化后的平方残差"
----------------------------------
论文写的是原始 `||x-x̂||²`。直接照抄有两个问题：
  1. 量纲支配：某些流量特征（如 Flow Bytes/s）数量级远大于 flag 计数，
     重构误差几乎只反映这些特征，归因（论文第 7 步"feature contributions"）会失真；
  2. 温和的跨天漂移会同时抬高所有特征的误差，稀释掉真正有判别力的方向。

因此本实现默认 `per_feature_norm=True`：
    r_j  = (x_j - x̂_j) / σ_j        σ_j = 良性训练集上残差的标准差（下限 1e-8）
    S_i  = mean_j r_j²
这是**单调变换下的可比口径**，B 论文（IEEE Access）用 z-score 归一化特征后同样得到
逐特征可比的残差；把 `per_feature_norm=False` 即退回论文的原始 MSE，两种都跑了。

与平台的嵌入维度约定
--------------------
平台 `obs_dim = feat_dim + (3 + embed_dim) + num_classes`，三种检测机制要统一
`embed_dim=4` 才能"只换检测器、其它全不动"地对比。因此本实现：
  * `score` 仍用**全部** 77 维残差（不牺牲精度）；
  * `embedding` 只取**判别力最强的前 k=4 个特征**的带符号标准化残差，
    维度可控且保留"为什么判异常"的方向信息。
"""

from __future__ import annotations

import json
import math
import os
import time
import zipfile
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_EPS = 1e-8
_DEFAULT_EMBED_DIM = 4  # 与平台 INTERFACE.md 的 detector_embed_dim 统一口径对齐


# =============================================================================
# 1. 配置
# =============================================================================
@dataclass
class AEConfig:
    """自编码器全部超参。默认值来自论文，并列明与平台的对应关系。"""

    # ---- 结构 ----
    hidden_dims: Sequence[int] = (32, 16, 8)
    """编码器各层宽度，逐层递减（论文："decreasing neuron counts"）。"""

    latent_dim: int = 4
    """瓶颈维度。平台内置 AE 用 8；AUTO 论文未给出，取 4 与 embed_dim 同阶。"""

    decoder: str = "asymmetric"
    """`asymmetric`（B 论文做法，解码器更浅）或 `symmetric`（AUTO 论文字面描述）。"""

    batchnorm: bool = True
    """编码器输入层 + 解码器输出层各一个 BatchNorm（B 论文：two BN layers）。"""

    activation: str = "leaky_relu"
    negative_slope: float = 0.1
    """B 论文用 LeakyReLU(slope=0.1)；AUTO 论文未指定。"""

    output_activation: str = "none"
    """数据经 min–max 落在 [0,1]，输出层加 sigmoid 看似合理，但实测会**掉点**
    （重构成"挤压过的"值反而抹掉了攻击流量的幅度信息）；所以默认 `none`。
    想复现 B 论文的写法可以设回 `sigmoid`。"""

    dropout: float = 0.1
    """轻微 dropout：PR-AUC 从 0.58 提到 0.65（AUC 基本不变），对抗过拟合有效。"""

    # ---- 训练（B 论文 VI.D 的配置，AUTO 论文未给出）----
    epochs: int = 250
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 0.0          # B 论文：L2 = 1e-3；实测在 CICIDS 上掉点，故默认关
    lr_decay: float = 0.9              # B 论文：drop factor 0.9
    lr_decay_every: int = 20           # B 论文：every 20 epochs
    val_fraction: float = 0.15         # B 论文：85%/15% train/validation
    patience: int = 80                 # B 论文：validation patience 15 epochs（实测 80~250 等价）
    loss: str = "mse"                  # 论文式(3)
    seed: int = 42
    device: str = "auto"
    verbose: bool = False

    # ---- 打分口径 ----
    per_feature_norm: bool = True
    """True → 逐特征标准化残差（推荐）；False → 论文原始 ||x-x̂||²。"""

    feature_transform: str = "none"
    """喂给网络之前的输入变换（核心**内部**做，不动平台给的数据）：
       `none`         直接吃平台 min–max 后的 [0,1] 特征（**默认**，实测最好）；
       `zscore`       逐特征标准化（B 论文做法）——实测 AUC 掉 0.09；
       `log1p_zscore` 先 log1p 再标准化 —— 实测掉更多。
    结论：平台已经做过 min–max，再做一次标准化会把攻击流量的幅度信息压掉。"""

    embed_dim: int = _DEFAULT_EMBED_DIM
    """暴露给平台的 embedding 维度（平台对比实验统一 4）。0 = 不要 embedding。"""

    top_k_by: str = "training"
    """选哪 k 个特征进 embedding：`training`（良性训练集残差能量最大，默认）
    或 `test`（在测试残差上选，会引入标签/测试信息，只用于消融）。"""

    # ---- 阈值 ----
    threshold_mode: str = "percentile"
    """`percentile`（良性分位数，平台默认口径）
       / `evt`（B 论文的 EVT-GPD 尾部建模）
       / `fixed`（显式 threshold，可用 0.52 对齐 AUTO 论文）。"""

    alpha: float = 0.05
    """percentile 模式的目标误报率，阈值取良性分的 1-alpha 分位。"""

    evt_tail_fraction: float = 0.10
    """EVT 建模用的良性分数上尾比例（B 论文用 GPD 拟合尾部）。"""

    threshold: Optional[float] = None
    """显式阈值；给定时所有模式都直接用它（例如 0.52 对齐论文）。"""

    def resolved_device(self) -> str:
        if self.device and self.device != "auto":
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:  # noqa: BLE001
            return "cpu"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["hidden_dims"] = list(self.hidden_dims)
        return d


# =============================================================================
# 2. 阈值：EVT / GPD 尾部建模（借鉴 IEEE Access 论文 V.D 节）
# =============================================================================
@dataclass
class EVTThreshold:
    """广义帕累托分布（GPD）拟合出来的阈值。

    做法（Peaks-Over-Threshold）：
      1. 取良性分数的上尾 u = quantile(scores, 1 - tail_fraction) 作为超阈值 u；
      2. 对超出量 (s - u | s > u) 用矩估计拟合 GPD(ξ, β)；
      3. 阈值 ξ* = u + (β/ξ) * ((alpha / ζ_u)^(-ξ) - 1)，其中 ζ_u = P(s > u)。
    当 ξ→0 时退化为指数分布形式。
    """

    u: float
    xi: float
    beta: float
    zeta_u: float
    alpha: float
    threshold: float
    n_exceed: int
    fallback: bool = False
    """True 表示 GPD 外推没通过一致性检验，已退回经验分位数。"""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _fit_gpd_moments(excess: np.ndarray) -> Tuple[float, float]:
    """矩估计拟合 GPD：ξ = 0.5*(1 - m2/m1²)，β = m1*(1-ξ)。"""
    m1 = float(np.mean(excess))
    m2 = float(np.mean(excess ** 2))
    if m1 <= _EPS:
        return 0.0, max(m1, _EPS)
    xi = 0.5 * (1.0 - m2 / (m1 ** 2))
    # CICIDS 的重构误差尾巴极重（偏度 ~1e2），不夹住 ξ 会让外推爆炸
    xi = float(np.clip(xi, -0.25, 0.25))
    beta = float(max(m1 * (1.0 - xi), _EPS))
    return xi, beta


def fit_evt_threshold(
    scores_normal: np.ndarray,
    alpha: float = 0.05,
    tail_fraction: float = 0.10,
    max_quantile_gap: float = 0.03,
) -> EVTThreshold:
    """在良性分数上做 POT/GPD 标定，返回目标误报率 alpha 对应的阈值。

    加了一致性检验：GPD 外推出的阈值在良性经验分布上对应的分位，若与目标
    `1-alpha` 相差超过 `max_quantile_gap`，说明尾部拟合不可靠（CICIDS 的
    重构误差偏度超过 100，很容易触发），此时退回经验分位数并把
    `fallback=True` 记进返回值，让报告能如实说明用的是哪条路径。
    """
    s = np.asarray(scores_normal, dtype=np.float64).ravel()
    s = s[np.isfinite(s)]
    if s.size < 50:
        raise ValueError(f"良性分数太少（{s.size}），无法做 EVT 标定")

    q = float(np.clip(1.0 - tail_fraction, 0.5, 0.999))
    u = float(np.quantile(s, q))
    excess = s[s > u] - u
    emp_thr = float(np.quantile(s, 1.0 - alpha))

    def _fallback(n_exceed: int) -> EVTThreshold:
        return EVTThreshold(u=u, xi=0.0, beta=_EPS,
                            zeta_u=float(n_exceed / s.size), alpha=alpha,
                            threshold=emp_thr, n_exceed=int(n_exceed), fallback=True)

    if excess.size < 10:
        return _fallback(int(excess.size))

    xi, beta = _fit_gpd_moments(excess)
    zeta_u = float(excess.size / s.size)
    ratio = alpha / max(zeta_u, _EPS)
    if abs(xi) < 1e-6:
        delta = beta * math.log(1.0 / max(ratio, _EPS))
    else:
        delta = (beta / xi) * (ratio ** (-xi) - 1.0)
    thr = float(u + delta)

    if not np.isfinite(thr) or thr <= 0:
        return _fallback(int(excess.size))

    # 一致性检验：外推阈值在良性经验分布上应≈1-alpha
    emp_q = float((s <= thr).mean())
    if abs(emp_q - (1.0 - alpha)) > max_quantile_gap:
        return _fallback(int(excess.size))

    return EVTThreshold(u=u, xi=xi, beta=beta, zeta_u=zeta_u, alpha=alpha,
                        threshold=thr, n_exceed=int(excess.size), fallback=False)


# =============================================================================
# 3. 打分与阈值的公共指标工具
# =============================================================================
def binary_metrics(y_true: np.ndarray, scores: np.ndarray, thr: float) -> Dict[str, float]:
    """在指定阈值下算 P/R/F1/FPR/检测率，并给出 AUC-ROC / PR-AUC。

    额外给出课题提案第 4 节要求的两个**基率校正**指标：

    * `fp_per_1000_flows`  —— 每 1000 条流量的误报数 = FPR × 1000。FPR 只在良性
      样本上计算，与攻击占比无关，因此这个数可以直接外推到线上；
    * `precision_at_1pct` / `precision_at_0.1pct` —— 把攻击占比（基率）换成 1% /
      0.1% 之后的精确率：P(ρ) = ρ·TPR / (ρ·TPR + (1-ρ)·FPR)。这才是能回答
      "能不能上线"的数字：检测率再高，1% 基率下 precision 也会被 FPR 压垮
      （the base-rate fallacy）。
    """
    y = np.asarray(y_true).astype(int).ravel()
    s = np.asarray(scores, dtype=np.float64).ravel()
    pred = (s >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    tnr = tn / (fp + tn) if (fp + tn) else 0.0

    out = {
        "threshold": float(thr),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float((tp + tn) / max(len(y), 1)),
        "false_positive_rate": float(fpr),
        "detection_rate": float(recall),
        "specificity": float(tnr),
        "balanced_accuracy": float(0.5 * (recall + tnr)),
        "fp_per_1000_flows": float(fpr * 1000.0),
        "precision_at_1pct": float(_precision_at_prior(recall, fpr, 0.01)),
        "precision_at_0.1pct": float(_precision_at_prior(recall, fpr, 0.001)),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "n": int(len(y)),
        "n_attack": int((y == 1).sum()),
        "n_benign": int((y == 0).sum()),
    }
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score

        if len(np.unique(y)) > 1:
            out["roc_auc"] = float(roc_auc_score(y, s))
            out["pr_auc"] = float(average_precision_score(y, s))
    except Exception:  # noqa: BLE001
        pass
    return out


def _precision_at_prior(tpr: float, fpr: float, prior: float) -> float:
    """把 TPR/FPR 换算到指定攻击占比（基率）下的精确率。"""
    hit = prior * float(tpr)
    alarm = hit + (1.0 - prior) * float(fpr)
    return 1.0 if alarm <= 0 else hit / alarm


def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    """扫描使 F1 最大的阈值（乐观上界，用于复现论文口径时注明）。"""
    y = np.asarray(y_true).astype(int).ravel()
    s = np.asarray(scores, dtype=np.float64).ravel()
    cand = np.unique(np.quantile(s, np.linspace(0.5, 0.9995, 400)))
    best_f1, best_thr = -1.0, float(cand[0]) if cand.size else 0.0
    for t in cand:
        _, _, f1 = _prf(y, s >= t)
        if f1 > best_f1:
            best_f1, best_thr = float(f1), float(t)
    return best_thr, best_f1


def _prf(y: np.ndarray, pred: np.ndarray) -> Tuple[float, float, float]:
    tp = float(((pred == 1) & (y == 1)).sum())
    fp = float(((pred == 1) & (y == 0)).sum())
    fn = float(((pred == 0) & (y == 1)).sum())
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


# =============================================================================
# 4. 自编码器核心
# =============================================================================
class _TorchAE:
    """极小的 torch 自编码器容器（把网络定义收在一个地方）。"""

    @staticmethod
    def build(cfg: AEConfig, input_dim: int):
        import torch
        import torch.nn as nn

        def act() -> nn.Module:
            if cfg.activation == "leaky_relu":
                return nn.LeakyReLU(cfg.negative_slope)
            if cfg.activation == "relu":
                return nn.ReLU()
            if cfg.activation == "elu":
                return nn.ELU()
            raise ValueError(f"未知激活函数 {cfg.activation!r}")

        # ---------------- encoder ----------------
        enc: List[nn.Module] = []
        if cfg.batchnorm:
            enc.append(nn.BatchNorm1d(input_dim))
        prev = input_dim
        for h in cfg.hidden_dims:
            enc.append(nn.Linear(prev, int(h)))
            enc.append(act())
            if cfg.dropout > 0:
                enc.append(nn.Dropout(cfg.dropout))
            prev = int(h)
        enc.append(nn.Linear(prev, cfg.latent_dim))
        encoder = nn.Sequential(*enc)

        # ---------------- decoder ----------------
        dec: List[nn.Module] = []
        if cfg.decoder == "symmetric":
            widths = [cfg.latent_dim] + [int(h) for h in reversed(cfg.hidden_dims)]
            for i in range(len(widths) - 1):
                dec.append(nn.Linear(widths[i], widths[i + 1]))
                dec.append(act())
            dec.append(nn.Linear(widths[-1], input_dim))
        else:
            # B 论文：解码器比编码器浅，防止把攻击流量也重构出来
            mid = max(int(cfg.hidden_dims[0]), cfg.latent_dim * 2)
            dec.append(nn.Linear(cfg.latent_dim, mid))
            dec.append(act())
            dec.append(nn.Linear(mid, input_dim))
        if cfg.batchnorm:
            dec.append(nn.BatchNorm1d(input_dim))
        if cfg.output_activation == "sigmoid":
            dec.append(nn.Sigmoid())
        decoder = nn.Sequential(*dec)

        class _Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.encoder = encoder
                self.decoder = decoder

            def forward(self, x):  # noqa: D102
                return self.decoder(self.encoder(x))

        return _Net()


class AutoencoderCore:
    """论文自编码器：只在良性流量上训练，用重构误差当异常分。

    典型用法
    --------
    >>> core = AutoencoderCore(AEConfig()).fit(X_normal)
    >>> core.score(X_test)          # (n,) 越高越异常
    >>> core.embedding(X_test)      # (n, k) 带符号标准化残差
    >>> core.save("outputs/ae_paper")     # -> ae_paper.npz + ae_paper.json
    """

    def __init__(self, config: Optional[AEConfig] = None) -> None:
        self.cfg = config or AEConfig()
        self.model = None
        self.input_dim: Optional[int] = None

        # 训练统计量
        self.threshold: Optional[float] = None
        self.evt: Optional[EVTThreshold] = None
        self.scaler_mu: Optional[np.ndarray] = None       # 输入变换：标准化均值
        self.scaler_sd: Optional[np.ndarray] = None       # 输入变换：标准化标准差
        self.resid_sigma: Optional[np.ndarray] = None     # 逐特征残差标准差
        self.resid_mu: Optional[np.ndarray] = None        # 逐特征残差均值（用于去偏）
        self.score_mu: float = 0.0                        # 良性分数均值
        self.score_sigma: float = 1.0                     # 良性分数标准差
        self.score_robust_scale: float = 1.0              # 良性分数稳健尺度（IQR/1.349）
        self.embed_idx: Optional[np.ndarray] = None       # embedding 取哪几维
        self.feature_importance: Optional[np.ndarray] = None  # 逐特征归因（论文第 7 步）
        self.history: List[Dict[str, float]] = []
        self.fit_meta: Dict[str, Any] = {}

    # ------------------------------------------------------------------ 训练
    def fit(
        self,
        X_normal: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        verbose: Optional[bool] = None,
    ) -> "AutoencoderCore":
        """
        参数
        ----
        X_normal : (n, d) 良性流量（**只能是良性**，攻击样本会让基线与阈值全错）
        X_val    : 可选，良性验证集；给了就用它做早停与阈值标定（不参与梯度）
        X_test, y_test : 可选，仅用于**阈值标定**（F1 扫描），绝不参与网络训练
        """
        import torch
        import torch.nn as nn

        cfg = self.cfg
        verbose = cfg.verbose if verbose is None else verbose
        X = np.asarray(X_normal, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError(f"X_normal 必须是 (n, d)，收到 {X.shape}")
        if X.shape[0] < 32:
            raise ValueError(f"良性样本太少（{X.shape[0]}），无法训练自编码器")

        self.input_dim = int(X.shape[1])
        self.resid_sigma = None  # 重新 fit 时清空旧统计

        if cfg.seed is not None:
            np.random.seed(cfg.seed)
            torch.manual_seed(cfg.seed)

        dev = torch.device(cfg.resolved_device())

        # ---- 训练/验证切分（论文/平台都是良性内部切分）----
        rng = np.random.default_rng(cfg.seed)
        perm = rng.permutation(X.shape[0])
        X = X[perm]
        n_val = int(round(X.shape[0] * cfg.val_fraction)) if X_val is None else 0
        if n_val > 0:
            X_tr, X_va = X[:-n_val], X[-n_val:]
        else:
            X_tr, X_va = X, None
        if X_val is not None:
            X_va = np.asarray(X_val, dtype=np.float32)

        # ---- 输入变换统计量（只在良性训练集上估计，推理时复用）----
        self._fit_scaler(X_tr)

        self.model = _TorchAE.build(cfg, self.input_dim).to(dev)
        opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        sched = None
        if cfg.lr_decay and cfg.lr_decay != 1.0 and cfg.lr_decay_every > 0:
            sched = torch.optim.lr_scheduler.StepLR(
                opt, step_size=int(cfg.lr_decay_every), gamma=float(cfg.lr_decay)
            )
        loss_fn = nn.MSELoss()

        t_tr = torch.from_numpy(np.ascontiguousarray(self._to_model_space(X_tr))).to(dev)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(t_tr),
            batch_size=min(cfg.batch_size, max(len(t_tr), 1)),
            shuffle=True,
            drop_last=False,
        )

        best_val = float("inf")
        best_state: Optional[Dict[str, Any]] = None
        bad_epochs = 0
        t0 = time.perf_counter()

        for ep in range(int(cfg.epochs)):
            self.model.train()
            tot, nb = 0.0, 0
            for (batch,) in loader:
                if batch.shape[0] < 2 and cfg.batchnorm:
                    # BatchNorm1d 在 batch=1 时会报错；直接跳过这种尾批
                    continue
                recon = self.model(batch)
                loss = loss_fn(recon, batch)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                tot += float(loss.item())
                nb += 1
            train_mse = tot / max(nb, 1)

            val_mse = float("nan")
            if X_va is not None and len(X_va) > 0:
                val_mse = float(np.mean(self._recon(X_va, squared=True)))

            self.history.append({"epoch": ep + 1, "train_mse": train_mse, "val_mse": val_mse})
            if sched is not None:
                sched.step()

            if verbose and (ep % max(1, int(cfg.epochs) // 10) == 0 or ep == int(cfg.epochs) - 1):
                print(f"  [AE] epoch {ep + 1:>4}/{cfg.epochs}  train_mse={train_mse:.6e}"
                      f"  val_mse={val_mse:.6e}")

            if np.isfinite(val_mse):
                if val_mse < best_val - 1e-12:
                    best_val, bad_epochs = val_mse, 0
                    best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    bad_epochs += 1
                    if bad_epochs >= int(cfg.patience):
                        if verbose:
                            print(f"  [AE] 早停于 epoch {ep + 1}（验证损失 {cfg.patience} 轮未改善）")
                        break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        # 用**全部**良性样本（含验证部分）重算统计量，但不动权重
        self.model.eval()
        self.fit_meta = {
            "n_normal": int(X.shape[0]),
            "n_train": int(len(X_tr)),
            "n_val": int(0 if X_va is None else len(X_va)),
            "epochs_run": len(self.history),
            "best_val_mse": None if not np.isfinite(best_val) else float(best_val),
            "train_seconds": float(time.perf_counter() - t0),
            "device": str(dev),
            "n_params": int(sum(p.numel() for p in self.model.parameters())),
        }

        self._finalize_statistics(X, X_test=X_test, y_test=y_test)
        return self

    # ------------------------------------------------------------------ 统计量/阈值
    def _finalize_statistics(
        self,
        X_normal: np.ndarray,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
    ) -> None:
        cfg = self.cfg
        R = self._residuals(X_normal)                       # (n, d) x - x_hat
        if cfg.per_feature_norm:
            self.resid_mu = R.mean(axis=0).astype(np.float64)
            self.resid_sigma = np.maximum(R.std(axis=0).astype(np.float64), 1e-8)
        else:
            self.resid_mu = np.zeros(R.shape[1], dtype=np.float64)
            self.resid_sigma = np.ones(R.shape[1], dtype=np.float64)

        # 逐特征归因（论文第 7 步：feature-level reconstruction error）
        self.feature_importance = self._feature_energy(X_normal)

        s_normal = self._score_from_residuals(R)
        self.score_mu = float(np.mean(s_normal))
        self.score_sigma = float(max(np.std(s_normal), 1e-8))
        # trust 用**稳健尺度**而不是 std：CICIDS 的良性分数是重尾的
        # （中位数 ~0.26、均值 ~0.87、最大值可达几千），std 被极少数极端良性
        # 样本撑大后，sigmoid((s-mu)/std) 会把几乎所有样本压到 0.5 附近，
        # 环境观测里的 trust 通道就退化成常数。IQR/1.349 对尾部不敏感，
        # 正态下等价于 σ（一致性系数 1.349）。
        q75, q25 = np.percentile(s_normal, [75.0, 25.0])
        robust = float((q75 - q25) / 1.349)
        self.score_robust_scale = robust if (np.isfinite(robust) and robust > 1e-9) else self.score_sigma

        # embedding 选哪 k 维
        k = int(max(0, cfg.embed_dim))
        if k > 0:
            if cfg.top_k_by == "test" and X_test is not None:
                energy = np.abs(self._std_residuals(np.asarray(X_test, dtype=np.float32))).mean(axis=0)
            else:
                energy = self.feature_importance
            order = np.argsort(-np.asarray(energy, dtype=np.float64))
            self.embed_idx = np.sort(order[:k]).astype(np.int64)
        else:
            self.embed_idx = np.zeros(0, dtype=np.int64)

        # 阈值
        if cfg.threshold is not None:
            self.threshold = float(cfg.threshold)
        elif cfg.threshold_mode == "evt":
            self.evt = fit_evt_threshold(s_normal, alpha=cfg.alpha,
                                         tail_fraction=cfg.evt_tail_fraction)
            self.threshold = float(self.evt.threshold)
        elif cfg.threshold_mode == "percentile":
            self.threshold = float(np.quantile(s_normal, 1.0 - cfg.alpha))
        else:
            raise ValueError(f"未知 threshold_mode={cfg.threshold_mode!r}")

        # 有标签时额外记录 F1 扫描阈值（乐观上界，论文口径）
        self.fit_meta["threshold_calibrated"] = float(self.threshold)
        if X_test is not None and y_test is not None and len(np.unique(y_test)) > 1:
            s_test = self.score(np.asarray(X_test, dtype=np.float32))
            thr_f1, f1 = best_f1_threshold(y_test, s_test)
            self.fit_meta["threshold_best_f1"] = float(thr_f1)
            self.fit_meta["best_f1_on_calibration_set"] = float(f1)
        self.fit_meta["benign_score_mean"] = self.score_mu
        self.fit_meta["benign_score_std"] = self.score_sigma
        self.fit_meta["benign_score_robust_scale"] = self.score_robust_scale

    def _feature_energy(self, X: np.ndarray) -> np.ndarray:
        """逐特征的平均平方残差 = 论文第 7 步的 feature contribution。

        注意：这里用**原始**（未标准化）平方残差，与论文公式一致；
        如果改用标准化残差，每个特征都会接近 1.0，排序会退化成常数。
        """
        R = self._residuals(X).astype(np.float64)
        return (R ** 2).mean(axis=0)

    # ------------------------------------------------------------------ 输入变换
    def _fit_scaler(self, X: np.ndarray) -> None:
        """在良性训练集上估计输入变换的统计量。"""
        mode = self.cfg.feature_transform
        if mode == "none":
            self.scaler_mu = None
            self.scaler_sd = None
            return
        A = np.asarray(X, dtype=np.float64)
        if mode == "log1p_zscore":
            A = np.log1p(np.maximum(A, 0.0))
        elif mode != "zscore":
            raise ValueError(f"未知 feature_transform={mode!r}")
        self.scaler_mu = A.mean(axis=0)
        self.scaler_sd = np.maximum(A.std(axis=0), 1e-6)

    def _to_model_space(self, X: np.ndarray) -> np.ndarray:
        """原始（平台 min–max）特征 -> 网络输入空间。"""
        A = np.asarray(X, dtype=np.float32)
        if self.scaler_mu is None:
            return A
        if self.cfg.feature_transform == "log1p_zscore":
            A = np.log1p(np.maximum(A, 0.0))
        A = (A.astype(np.float64) - self.scaler_mu) / self.scaler_sd
        return A.astype(np.float32)

    # ------------------------------------------------------------------ 前向
    def _recon(self, X: np.ndarray, squared: bool = False) -> np.ndarray:
        """返回**原始特征空间**的重构，保证残差与论文一致（可直接归因）。"""
        import torch

        if self.model is None:
            raise RuntimeError("AutoencoderCore 尚未 fit()")
        X = np.asarray(X, dtype=np.float32)
        dev = next(self.model.parameters()).device
        self.model.eval()
        out = np.empty_like(X, dtype=np.float32)
        bs = 4096
        with torch.no_grad():
            for i in range(0, X.shape[0], bs):
                chunk = torch.from_numpy(
                    np.ascontiguousarray(self._to_model_space(X[i:i + bs]))
                ).to(dev)
                rec = self.model(chunk).detach().cpu().numpy()
                out[i:i + bs] = rec
        if squared:
            return (X - out) ** 2
        return out

    def _residuals(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        return (X - self._recon(X)).astype(np.float32)

    def _std_residuals(self, X: np.ndarray) -> np.ndarray:
        R = self._residuals(X).astype(np.float64)
        if self.resid_mu is not None:
            R = R - self.resid_mu
        if self.resid_sigma is not None:
            R = R / self.resid_sigma
        return R

    def _score_from_residuals(self, R: np.ndarray) -> np.ndarray:
        if self.cfg.per_feature_norm and self.resid_sigma is not None:
            Z = (R.astype(np.float64) - self.resid_mu) / self.resid_sigma
        else:
            Z = R.astype(np.float64)
        return (Z ** 2).mean(axis=1).astype(np.float32)

    def score(self, X: np.ndarray) -> np.ndarray:
        """(n,) 异常分，越高越异常（论文式：重构误差）。"""
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X[None, :]
        return self._score_from_residuals(self._residuals(X))

    def embedding(self, X: np.ndarray) -> np.ndarray:
        """(n, k) 带符号标准化残差：让 critic 知道"哪些特征没被重构出来"。"""
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X[None, :]
        if self.embed_idx is None or self.embed_idx.size == 0:
            return np.zeros((X.shape[0], 0), dtype=np.float32)
        Z = self._std_residuals(X)
        return Z[:, self.embed_idx].astype(np.float32)

    def trust(self, scores: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """[0,1] 置信度：告警时=判为异常的置信度，不告警时=确信正常的程度。

        形式与平台内置 AE 同口径（良性分数标准化后过 sigmoid，再按 flag 取反），
        但尺度改用**稳健尺度** `score_robust_scale = 良性分数 IQR/1.349`，而不是 std。
        原因见 `_finalize_statistics`：良性分数重尾时 std 会把 trust 压成常数 0.5，
        观测里的 trust 通道就没信息量了（实测 std 口径下 trust 的 5%~95% 区间只有
        [0.49, 0.66]；稳健尺度下是 [0.10, 1.00]、中位数 0.91）。

        中心仍用良性分数均值：分数分布右偏，均值落在"典型正常水平"之上，
        于是"接近阈值"的样本 trust≈0.5（不确定）、两侧分别趋近 1，
        语义正好是"我对这个 flag 有多确信"。
        """
        s = np.asarray(scores, dtype=np.float64).ravel()
        thr = float(self.threshold if threshold is None else threshold)
        scale = float(getattr(self, "score_robust_scale", 0.0) or 0.0)
        if not np.isfinite(scale) or scale <= 1e-9:
            scale = float(self.score_sigma) or 1.0
        z = (s - self.score_mu) / scale
        conf = 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))
        flag = s >= thr
        return np.where(flag, conf, 1.0 - conf).astype(np.float32)

    def latency(self, X: np.ndarray, repeats: int = 3) -> Dict[str, float]:
        """推理延迟/吞吐（对齐 B 论文 VI.D 的 runtime 报表口径）。"""
        X = np.asarray(X, dtype=np.float32)
        n = X.shape[0]
        bs = 256
        batches = max(1, math.ceil(n / bs))
        best = float("inf")
        for _ in range(max(1, repeats)):
            t0 = time.perf_counter()
            self.score(X)
            best = min(best, time.perf_counter() - t0)
        return {
            "n_samples": int(n),
            "total_seconds": best,
            "throughput_samples_per_sec": float(n / max(best, 1e-9)),
            "ms_per_batch256": float(best / batches * 1000.0),
            "ms_per_sample": float(best / max(n, 1) * 1000.0),
        }

    # ------------------------------------------------------------------ 落盘
    def state_dict(self) -> Dict[str, np.ndarray]:
        import torch

        sd: Dict[str, np.ndarray] = {}
        if self.model is not None:
            for k, v in self.model.state_dict().items():
                sd["w/" + k] = v.detach().cpu().numpy()
        sd["resid_mu"] = np.asarray(self.resid_mu if self.resid_mu is not None else [], dtype=np.float64)
        sd["resid_sigma"] = np.asarray(self.resid_sigma if self.resid_sigma is not None else [], dtype=np.float64)
        sd["scaler_mu"] = np.asarray(self.scaler_mu if self.scaler_mu is not None else [], dtype=np.float64)
        sd["scaler_sd"] = np.asarray(self.scaler_sd if self.scaler_sd is not None else [], dtype=np.float64)
        sd["embed_idx"] = np.asarray(self.embed_idx if self.embed_idx is not None else [], dtype=np.int64)
        sd["feature_importance"] = np.asarray(
            self.feature_importance if self.feature_importance is not None else [], dtype=np.float64)
        return sd

    def load_state_dict(self, sd: Dict[str, np.ndarray]) -> "AutoencoderCore":
        import torch

        sd = {k: np.asarray(v) for k, v in sd.items()}
        self.resid_mu = sd["resid_mu"] if sd["resid_mu"].size else None
        self.resid_sigma = sd["resid_sigma"] if sd["resid_sigma"].size else None
        self.scaler_mu = sd["scaler_mu"] if "scaler_mu" in sd and sd["scaler_mu"].size else None
        self.scaler_sd = sd["scaler_sd"] if "scaler_sd" in sd and sd["scaler_sd"].size else None
        self.embed_idx = sd["embed_idx"].astype(np.int64)
        self.feature_importance = sd["feature_importance"] if sd["feature_importance"].size else None
        weights = {k[2:]: torch.from_numpy(np.ascontiguousarray(v))
                   for k, v in sd.items() if k.startswith("w/")}
        if weights:
            if self.input_dim is None:
                raise RuntimeError("load_state_dict 前需要先知道 input_dim")
            dev = torch.device(self.cfg.resolved_device())
            self.model = _TorchAE.build(self.cfg, self.input_dim).to(dev)
            self.model.load_state_dict(weights)
            self.model.eval()
        return self

    def save(self, path: str) -> None:
        """写两个文件：`<path>.npz`（权重与统计量）+ `<path>.json`（配置与元信息）。"""
        base = os.path.abspath(path)
        for ext in (".npz", ".npz.tmp"):
            if base.endswith(ext):
                base = base[: -len(ext)]
        os.makedirs(os.path.dirname(base) or ".", exist_ok=True)
        np.savez_compressed(base + ".npz", **self.state_dict())
        meta = {
            "format": "ae_repro.AutoencoderCore",
            "version": 1,
            "config": self.cfg.to_dict(),
            "input_dim": self.input_dim,
            "threshold": self.threshold,
            "score_mu": self.score_mu,
            "score_sigma": self.score_sigma,
            "score_robust_scale": self.score_robust_scale,
            "evt": None if self.evt is None else self.evt.to_dict(),
            "fit_meta": self.fit_meta,
            "history": self.history,
        }
        with open(base + ".json", "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=2)

    def load(self, path: str) -> "AutoencoderCore":
        base = os.path.abspath(path)
        if base.endswith(".json"):
            base = base[:-5]
        if base.endswith(".npz"):
            base = base[:-4]
        with open(base + ".json", "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        cfg = meta.get("config", {})
        if "hidden_dims" in cfg:
            cfg["hidden_dims"] = tuple(cfg["hidden_dims"])
        allowed = set(AEConfig.__dataclass_fields__.keys())
        self.cfg = AEConfig(**{k: v for k, v in cfg.items() if k in allowed})
        self.input_dim = int(meta["input_dim"])
        self.threshold = meta.get("threshold")
        self.score_mu = float(meta.get("score_mu", 0.0))
        self.score_sigma = float(meta.get("score_sigma", 1.0))
        # 旧版本的 .json 没有这个字段：回退到 std，行为与修复前一致
        self.score_robust_scale = float(meta.get("score_robust_scale") or self.score_sigma)
        evt = meta.get("evt")
        self.evt = None if not evt else EVTThreshold(**evt)
        self.fit_meta = dict(meta.get("fit_meta", {}))
        self.history = list(meta.get("history", []))
        with np.load(base + ".npz", allow_pickle=False) as blob:
            self.load_state_dict({k: blob[k] for k in blob.files})
        return self

    # ------------------------------------------------------------------ 自检
    def describe(self) -> str:
        cfg = self.cfg
        lines = [
            f"AE 结构        : {self.input_dim} -> {list(cfg.hidden_dims)} -> {cfg.latent_dim}"
            f" -> {'镜像' if cfg.decoder == 'symmetric' else '浅层'}解码 -> {self.input_dim}",
            f"激活/归一化    : {cfg.activation}(slope={cfg.negative_slope})  BN={cfg.batchnorm}"
            f"  输出={cfg.output_activation}",
            f"训练           : epochs<={cfg.epochs} bs={cfg.batch_size} lr={cfg.lr}"
            f" decay={cfg.lr_decay}/{cfg.lr_decay_every} patience={cfg.patience}",
            f"参数量         : {self.fit_meta.get('n_params', 'n/a')}",
            f"打分口径       : {'逐特征标准化残差' if cfg.per_feature_norm else '原始 MSE'}",
            f"阈值           : {self.threshold}  (mode={cfg.threshold_mode})",
            f"trust 尺度     : {self.score_robust_scale:.4g}"
            f"（稳健 IQR/1.349，std={self.score_sigma:.4g}）",
            f"embedding 维度 : {cfg.embed_dim}  取特征 idx={list(map(int, self.embed_idx)) if self.embed_idx is not None else []}",
        ]
        return "\n".join(lines)
