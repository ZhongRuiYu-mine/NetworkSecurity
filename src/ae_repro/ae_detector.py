# -*- coding: utf-8 -*-
"""
ae_detector.py —— ★ 贴进平台仓库的检测器文件（复现版自编码器）

用法（二选一）
--------------
① **拷贝进仓库**（推荐，仓库内零改动即可被 `make_detector` / CLI 认识）：

       cp ae_detector.py  <repo>/agentenvs/detectors/ae_paper_detector.py
       # 然后在 <repo>/agentenvs/detectors/__init__.py 里加一行：
       #   from .ae_paper_detector import AePaperDetector
       # 之后就能：python train.py --detector ae_paper --embed-dim 4

② **不改仓库**，直接传对象：

       from ae_repro.ae_detector import AePaperDetector
       env = make_env(ds, detector=AePaperDetector(embed_dim=4))

契约实现情况（对照 INTERFACE.md 第 8 章检查清单）
------------------------------------------------
[x] 继承 `DetectorBase`，实现 `fit()` 与 `score_batch()`，`fit()` 返回 self
[x] 只用良性流量 fit；`X_test/y_test` 仅用于阈值标定，绝不进网络训练
[x] `score` 方向 = 越高越异常（重构误差，天然同向）
[x] `score/flag/trust` 形状 (n,)，`embedding` (n,k)，全 `np.isfinite`
[x] `flag = score >= self.threshold`，`threshold` 存成属性
[x] `trust` = 对当前 flag 的置信度（告警时为置信度，否则为确信正常的程度）
[x] `embed_dim` 类属性可配（平台对比实验统一 4）
[x] `reset()` 无状态 no-op
[x] `save()/load()` 已实现（.npz 权重 + .json 元信息，不依赖 torch.save）
"""

from __future__ import annotations

import os
from typing import Any, Optional, Sequence

import numpy as np

from .ae_core import AEConfig, AutoencoderCore

# -------- 契约导入：贴进仓库时走仓库的 base；独立运行时退回自带 shim --------
try:  # pragma: no cover - 取决于运行位置
    from .base import DetectorBase, DetectorOutput, register_detector
except Exception:  # noqa: BLE001
    try:
        from agentenvs.detectors.base import (  # type: ignore
            DetectorBase,
            DetectorOutput,
            register_detector,
        )
    except Exception:  # noqa: BLE001
        from ae_repro._contract import (  # type: ignore
            DetectorBase,
            DetectorOutput,
            register_detector,
        )


@register_detector("ae_paper")
@register_detector("ae_cicids")
class AePaperDetector(DetectorBase):
    """论文复现版自编码器检测器（AUTO.pdf 第 1 阶段 + IEEE Access 的 EVT 阈值）。

    参数
    ----
    hidden_dims   : 编码器各层宽度，默认 (32,16,8)，逐层递减（论文原文）
    latent_dim    : 瓶颈维度，默认 4
    epochs        : 最大训练轮数，默认 200（早停控制实际轮数）
    batch_size    : 默认 128
    lr            : Adam 初始学习率，默认 1e-3（论文）
    weight_decay  : L2 正则，默认 0.0（论文用 1e-3，CICIDS 上偏欠拟合）
    alpha         : percentile 模式的目标误报率，默认 0.05 -> 阈值取良性 95% 分位
    threshold     : 显式阈值（例如 0.52 对齐论文 Table 1 的 AE 行）
    threshold_mode: 'percentile' | 'evt' | 'fixed'
    decoder       : 'asymmetric'（默认）| 'symmetric'
    batchnorm     : 是否用 BatchNorm1d，默认 True
    per_feature_norm : 是否用逐特征标准化残差，默认 True（见 ae_core 的说明）
    embed_dim     : embedding 维度，默认 4（与平台三方对比口径一致）
    device        : 'auto' | 'cpu' | 'cuda'
    seed          : 随机种子，默认 42
    """

    embed_dim: int = 4
    name: str = "ae_paper"

    def __init__(
        self,
        hidden_dims: Sequence[int] = (32, 16, 8),
        latent_dim: int = 4,
        epochs: int = 250,
        batch_size: int = 128,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        lr_decay: float = 0.9,
        lr_decay_every: int = 20,
        val_fraction: float = 0.15,
        patience: int = 80,
        alpha: float = 0.05,
        threshold: Optional[float] = None,
        threshold_mode: str = "percentile",
        evt_tail_fraction: float = 0.10,
        decoder: str = "asymmetric",
        batchnorm: bool = True,
        activation: str = "leaky_relu",
        negative_slope: float = 0.1,
        output_activation: str = "none",
        dropout: float = 0.1,
        feature_transform: str = "none",
        per_feature_norm: bool = True,
        embed_dim: int = 4,
        top_k_by: str = "training",
        device: str = "auto",
        seed: int = 42,
        verbose: bool = False,
        **_: Any,
    ) -> None:
        self.cfg = AEConfig(
            hidden_dims=tuple(int(h) for h in hidden_dims),
            latent_dim=int(latent_dim),
            epochs=int(epochs),
            batch_size=int(batch_size),
            lr=float(lr),
            weight_decay=float(weight_decay),
            lr_decay=float(lr_decay),
            lr_decay_every=int(lr_decay_every),
            val_fraction=float(val_fraction),
            patience=int(patience),
            alpha=float(alpha),
            threshold=None if threshold is None else float(threshold),
            threshold_mode=str(threshold_mode),
            evt_tail_fraction=float(evt_tail_fraction),
            decoder=str(decoder),
            batchnorm=bool(batchnorm),
            activation=str(activation),
            negative_slope=float(negative_slope),
            output_activation=str(output_activation),
            dropout=float(dropout),
            feature_transform=str(feature_transform),
            per_feature_norm=bool(per_feature_norm),
            embed_dim=int(embed_dim),
            top_k_by=str(top_k_by),
            device=str(device),
            seed=int(seed),
            verbose=bool(verbose),
        )
        self.core = AutoencoderCore(self.cfg)
        self.embed_dim = int(embed_dim)
        self.input_dim: Optional[int] = None

    # ------------------------------------------------------------------ 属性
    @property
    def threshold(self) -> Optional[float]:
        return self.core.threshold

    @threshold.setter
    def threshold(self, value: Optional[float]) -> None:
        self.core.threshold = value

    @property
    def device(self) -> str:
        """当前实际使用的计算设备（加载后跟随权重所在设备）。"""
        if self.core.model is not None:
            try:
                return str(next(self.core.model.parameters()).device)
            except StopIteration:  # pragma: no cover
                pass
        return self.cfg.resolved_device()

    @property
    def feature_names_(self) -> Optional[np.ndarray]:
        """归因用的特征名（首次 `fit(..., feature_names=[...])` 时记录）。"""
        return getattr(self, "_feature_names", None)

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        X_normal: np.ndarray,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        feature_names: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> "AePaperDetector":
        """在良性流量上训练自编码器并标定阈值。返回 self（平台要求）。"""
        X_normal = np.asarray(X_normal, dtype=np.float32)
        if X_normal.ndim != 2:
            raise ValueError(f"X_normal 必须是 (n, d)，收到 {X_normal.shape}")
        self.input_dim = int(X_normal.shape[1])
        if feature_names is not None:
            self._feature_names = np.asarray(list(feature_names), dtype=object)
        elif not hasattr(self, "_feature_names") and kwargs.get("feature_names") is None:
            self._feature_names = None

        self.core.fit(
            X_normal,
            X_test=None if X_test is None else np.asarray(X_test, dtype=np.float32),
            y_test=None if y_test is None else np.asarray(y_test),
        )
        return self

    # ------------------------------------------------------------------ 打分
    def score_batch(self, X: np.ndarray) -> DetectorOutput:
        """对 (n, d) 打分，返回契约对象 DetectorOutput。"""
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X[None, :]
        if X.shape[1] != self.input_dim:
            raise ValueError(
                f"特征维度不匹配：检测器按 {self.input_dim} 维训练，收到 {X.shape[1]} 维。"
                f"换特征子集后必须重新 fit()。"
            )

        score = self.core.score(X).astype(np.float32)
        thr = float(self.threshold if self.threshold is not None else np.inf)
        flag = (score >= thr).astype(np.float32)
        trust = self.core.trust(score, threshold=thr)

        emb = None
        if self.embed_dim > 0:
            emb = self.core.embedding(X).astype(np.float32)
            if emb.shape[1] < self.embed_dim:  # k > 可用特征数时补零
                emb = np.concatenate(
                    [emb, np.zeros((emb.shape[0], self.embed_dim - emb.shape[1]), np.float32)],
                    axis=1,
                )

        score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
        trust = np.nan_to_num(trust, nan=0.5, posinf=1.0, neginf=0.0)

        extra = {
            "threshold": thr,
            "score_mean_benign": self.core.score_mu,
            "score_std_benign": self.core.score_sigma,
            "score_robust_scale_benign": self.core.score_robust_scale,
            "embed_feature_idx": None if self.core.embed_idx is None
            else [int(i) for i in self.core.embed_idx],
        }
        if self.feature_names_ is not None and self.core.embed_idx is not None:
            fn = self.feature_names_
            extra["embed_feature_names"] = [
                str(fn[i]) for i in self.core.embed_idx if 0 <= int(i) < len(fn)
            ]
        return DetectorOutput(score=score, flag=flag, trust=trust, embedding=emb, extra=extra)

    def reset(self) -> None:
        """无时序状态，no-op（平台在 episode 边界会调用）。"""
        return None

    # ------------------------------------------------------------------ 归因
    def feature_attribution(self, top_k: int = 15) -> list:
        """论文第 7 步：返回 (特征名, 平均平方重构误差) 的 Top-k 列表。"""
        fi = self.core.feature_importance
        if fi is None:
            raise RuntimeError("尚未 fit()")
        order = np.argsort(-np.asarray(fi, dtype=np.float64))[: int(top_k)]
        names = self.feature_names_
        return [
            (
                str(names[i]) if names is not None and i < len(names) else f"f{i}",
                float(fi[i]),
            )
            for i in order
        ]

    # ------------------------------------------------------------------ 落盘
    def save(self, path: str) -> None:
        """`<path>.npz` 权重 + `<path>.json` 元信息（内部即 core.save）。"""
        self.core.save(path)
        if self.feature_names_ is not None:
            import json

            base = os.path.abspath(path)
            if base.endswith(".npz"):
                base = base[:-4]
            side = base + ".features.json"
            with open(side, "w", encoding="utf-8") as fh:
                json.dump([str(x) for x in self.feature_names_], fh, ensure_ascii=False)

    def load(self, path: str) -> "AePaperDetector":
        """返回 self（平台要求）。"""
        self.core.load(path)
        self.embed_dim = int(self.core.cfg.embed_dim)
        self.input_dim = int(self.core.input_dim or 0)
        base = os.path.abspath(path)
        if base.endswith(".npz"):
            base = base[:-4]
        side = base + ".features.json"
        if os.path.exists(side):
            import json

            with open(side, "r", encoding="utf-8") as fh:
                self._feature_names = np.asarray(json.load(fh), dtype=object)
        return self

    # ------------------------------------------------------------------ 摘要
    def describe(self) -> str:
        return self.core.describe()

    def extra_report(self) -> dict:
        """给报告用的额外信息（阈值口径、逐特征归因、训练历史结尾）。"""
        return {
            "threshold": self.core.threshold,
            "score_mean_benign": self.core.score_mu,
            "score_std_benign": self.core.score_sigma,
            "score_robust_scale_benign": self.core.score_robust_scale,
            "evt": None if self.core.evt is None else self.core.evt.to_dict(),
            "embed_feature_idx": None if self.core.embed_idx is None
            else [int(i) for i in self.core.embed_idx],
            "top_features": self.feature_attribution(15),
            "fit_meta": dict(self.core.fit_meta),
            "history_tail": self.core.history[-5:],
        }


# 允许 `python -m ae_repro.ae_detector` 直接自检
if __name__ == "__main__":  # pragma: no cover
    rng = np.random.default_rng(0)
    Xn = np.abs(rng.normal(0.3, 0.06, size=(4000, 20))).astype(np.float32)
    Xa = np.abs(rng.normal(0.3, 0.06, size=(800, 20))).astype(np.float32)
    Xa[:, :5] += 0.35
    y = np.r_[np.zeros(len(Xn), int), np.ones(len(Xa), int)]
    X = np.r_[Xn, Xa]

    det = AePaperDetector(epochs=40, embed_dim=4, verbose=False)
    det.fit(Xn, X_test=X, y_test=y)
    out = det.score_batch(X)
    from sklearn.metrics import roc_auc_score

    print("[OK] signal_dim =", det.signal_dim)
    print("     score shape =", out.score.shape, " flag  shape =", out.flag.shape,
          " trust shape =", out.trust.shape, " emb =", None if out.embedding is None else out.embedding.shape)
    print("     AUC =", round(float(roc_auc_score(y, out.score)), 4),
          " finite =", bool(np.isfinite(out.score).all() and np.isfinite(out.trust).all()))
    print("     detect() 旧接口 =", det.detect(X[0]))
