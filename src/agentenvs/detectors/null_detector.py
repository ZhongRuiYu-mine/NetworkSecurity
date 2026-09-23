"""Null 检测器：不检测，只返回中性信号。

用途：
  * 消融实验的 baseline（MADDPG 完全靠原始特征自己学）；
  * 校验「可插拔接口」是否真的解耦 —— 换掉检测器后环境行为仍然一致。
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .base import DetectorBase, DetectorOutput, register_detector


@register_detector("null")
class NullDetector(DetectorBase):
    """永远输出 score=0 / flag=0 / trust=0.5 的空检测器。"""

    embed_dim = 0

    def __init__(
        self,
        signal_score: float = 0.0,
        signal_flag: int = 0,
        signal_trust: float = 0.5,
        embed_dim: int = 0,
        **_: Any,
    ) -> None:
        self._score = float(signal_score)
        self._flag = int(signal_flag)
        self._trust = float(signal_trust)
        self.embed_dim = int(embed_dim)

    def fit(self, X_normal: np.ndarray, **kwargs: Any) -> "NullDetector":
        return self

    def score_batch(self, X: np.ndarray) -> DetectorOutput:
        n = int(np.asarray(X).shape[0])
        emb: Optional[np.ndarray] = (
            np.zeros((n, self.embed_dim), dtype=np.float32) if self.embed_dim else None
        )
        return DetectorOutput(
            score=np.full(n, self._score, dtype=np.float32),
            flag=np.full(n, self._flag, dtype=np.float32),
            trust=np.full(n, self._trust, dtype=np.float32),
            embedding=emb,
        )

    def save(self, path: str) -> None:  # 无状态，无需落盘
        return None

    def load(self, path: str) -> "NullDetector":
        return self
