# -*- coding: utf-8 -*-
"""
_contract.py —— 平台 `agentenvs/detectors/base.py` 的**逐字副本**（仅供独立运行自检用）。

为什么需要它：`ae_detector.py` 要能被两种方式使用：
  1. 拷进仓库 —— 走 `from .base import ...`，用的是仓库里的真契约；
  2. 在本交付包里直接跑 —— 走这个副本，保证契约字段与仓库**完全一致**。

本文件内容取自 https://github.com/ZhongRuiYu-mine/NetworkSecurity 的
`agentenvs/detectors/base.py`（main 分支）。**不要修改本文件**；如果仓库那边
契约升级了，请重新同步，并跑 `python -m ae_repro.verify_contract` 复核。

平台契约（摘要）
----------------
    DetectorBase.fit(X_normal, **kw) -> self
    DetectorBase.score_batch(X) -> DetectorOutput
    DetectorOutput(score, flag, trust, embedding=None, extra={})
    信号向量布局 [score, flag, trust, embedding...]，长度 = 3 + embed_dim
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

SIGNAL_SCORE = 0
SIGNAL_FLAG = 1
SIGNAL_TRUST = 2
SIGNAL_TAIL = 3


@dataclass
class DetectorOutput:
    """检测器对一批流量的输出（契约对象，环境只读）。"""

    score: np.ndarray
    flag: np.ndarray
    trust: np.ndarray
    embedding: Optional[np.ndarray] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.score)
        self.score = np.asarray(self.score, dtype=np.float32).reshape(n)
        self.flag = (np.asarray(self.flag).reshape(n) > 0.5).astype(np.float32)
        self.trust = np.clip(np.asarray(self.trust, dtype=np.float32).reshape(n), 0.0, 1.0)
        if self.embedding is not None:
            self.embedding = np.asarray(self.embedding, dtype=np.float32)
            if self.embedding.ndim == 1:
                self.embedding = self.embedding.reshape(n, -1)
            if self.embedding.shape[0] != n:
                raise ValueError("embedding 行数与 score 不一致")

    def as_signal(self, embed_dim: int = 0) -> np.ndarray:
        base = np.stack([self.score, self.flag, self.trust], axis=1).astype(np.float32)
        if embed_dim <= 0:
            return base
        if self.embedding is None:
            tail = np.zeros((base.shape[0], embed_dim), dtype=np.float32)
        else:
            tail = self.embedding[:, :embed_dim]
            if tail.shape[1] < embed_dim:
                pad = np.zeros((base.shape[0], embed_dim - tail.shape[1]), dtype=np.float32)
                tail = np.concatenate([tail, pad], axis=1)
        return np.concatenate([base, tail], axis=1).astype(np.float32)

    def __len__(self) -> int:
        return int(self.score.shape[0])


class DetectorBase(abc.ABC):
    embed_dim: int = 0
    name: str = "detector"

    @abc.abstractmethod
    def fit(self, X_normal: np.ndarray, **kwargs: Any) -> "DetectorBase":
        raise NotImplementedError

    @abc.abstractmethod
    def score_batch(self, X: np.ndarray) -> DetectorOutput:
        raise NotImplementedError

    def score(self, x: np.ndarray) -> DetectorOutput:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        return self.score_batch(x)

    def detect(self, X: np.ndarray) -> Dict[str, float]:
        out = self.score(X if np.ndim(X) > 1 else np.asarray(X)[None, :])
        return {
            "anomaly_score": float(out.score[0]),
            "anomaly_flag": int(out.flag[0]),
            "trust": float(out.trust[0]),
        }

    @property
    def signal_dim(self) -> int:
        return SIGNAL_TAIL + int(self.embed_dim)

    def reset(self) -> None:
        """有状态检测器在此清理内部状态。"""

    def save(self, path: str) -> None:
        raise NotImplementedError(f"{type(self).__name__} 未实现 save()")

    def load(self, path: str) -> "DetectorBase":
        raise NotImplementedError(f"{type(self).__name__} 未实现 load()")

    @staticmethod
    def _calibrate_threshold(
        scores_normal: np.ndarray,
        alpha: float = 0.05,
        scores_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
    ) -> float:
        scores_normal = np.asarray(scores_normal, dtype=np.float64).ravel()
        if scores_test is not None and y_test is not None and len(np.unique(y_test)) > 1:
            from sklearn.metrics import f1_score

            cand = np.unique(np.quantile(scores_test, np.linspace(0.5, 0.999, 200)))
            best_f1, best_thr = -1.0, float(np.quantile(scores_normal, 1 - alpha))
            for t in cand:
                f1 = f1_score(y_test, (scores_test >= t).astype(int), zero_division=0)
                if f1 > best_f1:
                    best_f1, best_thr = float(f1), float(t)
            return best_thr
        return float(np.quantile(scores_normal, 1.0 - alpha))


DETECTOR_REGISTRY: Dict[str, type] = {}


def register_detector(key: str):
    def _wrap(cls: type) -> type:
        DETECTOR_REGISTRY[key] = cls
        cls.name = key
        return cls

    return _wrap


def list_detectors() -> List[str]:
    return sorted(DETECTOR_REGISTRY)


def make_detector(key: str, **kwargs: Any) -> DetectorBase:
    if key not in DETECTOR_REGISTRY:
        raise KeyError(
            f"未知检测器 {key!r}，已注册：{list_detectors()}。"
            f"自定义检测器请用 @register_detector('{key}') 注册，或直接把对象传给环境。"
        )
    return DETECTOR_REGISTRY[key](**kwargs)


def coerce_detector(det: Any, **kwargs: Any) -> DetectorBase:
    if det is None:
        return make_detector("null", **kwargs)
    if isinstance(det, str):
        return make_detector(det, **kwargs)
    if isinstance(det, DetectorBase):
        return det
    if hasattr(det, "score_batch") or hasattr(det, "detect"):
        return det  # type: ignore[return-value]
    raise TypeError(f"无法识别的检测器对象：{type(det)!r}")
