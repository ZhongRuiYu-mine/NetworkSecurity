"""检测机制 1/3：Isolation Forest（孤立森林）。

对齐 `chethuhn/.../IF.py` 里的做法：
  * 在良性（Monday 等）流量上 fit；
  * `score_samples` 越高越正常，取负得到「越高越异常」的 score；
  * 阈值用良性分数分位数标定（控制误报率），有标签时用 F1 扫描。

embedding：IF 没有天然的低维残差向量，默认 embed_dim=0。
如果想让 critic 看到更多信息，可设 `embed_dim=k`，用每棵树的路径长度
（近似）作为 k 维嵌入 —— 这里用「前 k 棵树 path length」实现。
"""
from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np

from .base import DetectorBase, DetectorOutput, register_detector


@register_detector("if")
@register_detector("isolation_forest")
class IsolationForestDetector(DetectorBase):
    def __init__(
        self,
        n_estimators: int = 200,
        max_samples: Any = 256,          # 与 IF.py 一致：固定采样量，避免大树
        contamination: Any = "auto",
        max_features: float = 1.0,
        bootstrap: bool = False,
        random_state: Optional[int] = 42,
        alpha: float = 0.05,             # 目标误报率（用于阈值分位数标定）
        threshold: Optional[float] = None,
        embed_dim: int = 0,              # >0 时用树路径长度做嵌入
        n_jobs: Optional[int] = 1,       # 默认单线程；在沙箱/受限环境里 joblib 多进程可能被拒
        **_: Any,
    ) -> None:
        self.n_estimators = int(n_estimators)
        self.max_samples = max_samples
        self.contamination = contamination
        self.max_features = max_features
        self.bootstrap = bool(bootstrap)
        self.random_state = random_state
        self.alpha = float(alpha)
        self.embed_dim = int(embed_dim)
        self.n_jobs = n_jobs
        self.model = None
        self.threshold = float(threshold) if threshold is not None else None
        self._normal_stats: Optional[tuple] = None  # (mean, std) 用于 trust 归一化

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        X_normal: np.ndarray,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> "IsolationForestDetector":
        from sklearn.ensemble import IsolationForest

        X_normal = np.asarray(X_normal, dtype=np.float32)
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            contamination=self.contamination,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.model.fit(X_normal)

        s_normal = self._raw_score(X_normal)
        self._normal_stats = (float(s_normal.mean()), float(s_normal.std() + 1e-8))

        if self.threshold is None:
            s_test = self._raw_score(np.asarray(X_test, dtype=np.float32)) if X_test is not None else None
            self.threshold = self._calibrate_threshold(
                s_normal, alpha=self.alpha, scores_test=s_test, y_test=y_test
            )
        return self

    # ------------------------------------------------------------------ score
    def _raw_score(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("IsolationForestDetector 尚未 fit()")
        # score_samples 越高越正常 -> 取负，得到越高越异常
        return (-self.model.score_samples(X)).astype(np.float32)

    def score_batch(self, X: np.ndarray) -> DetectorOutput:
        X = np.asarray(X, dtype=np.float32)
        score = self._raw_score(X)

        thr = float(self.threshold) if self.threshold is not None else float(np.median(score) + 3.0)
        flag = (score >= thr).astype(np.float32)

        # trust：分数相对良性分布标准化后做 sigmoid，越偏离良性越"确定异常"
        mu, sd = self._normal_stats if self._normal_stats else (float(score.mean()), float(score.std() + 1e-8))
        z = (score - mu) / sd
        trust = 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))
        # 二值一致性：告警时 trust 表示置信度，不告警时 trust 表示"确信正常"
        trust = np.where(flag > 0, trust, 1.0 - trust).astype(np.float32)

        emb = None
        if self.embed_dim > 0:
            emb = self._path_embedding(X)
        return DetectorOutput(score=score, flag=flag, trust=trust, embedding=emb,
                              extra={"threshold": thr})

    def _path_embedding(self, X: np.ndarray) -> np.ndarray:
        """用前 embed_dim 棵树的平均路径长度作为嵌入（近似 anomaly 拆解）。"""
        k = min(self.embed_dim, self.n_estimators)
        try:
            depths = self.model.estimators_[:k]
            pls = np.stack([d.decision_path(X).sum(axis=1).ravel() for d in depths], axis=1)
        except Exception:
            return np.zeros((X.shape[0], self.embed_dim), dtype=np.float32)
        emb = np.zeros((X.shape[0], self.embed_dim), dtype=np.float32)
        emb[:, :k] = pls.astype(np.float32)
        return emb

    # ------------------------------------------------------------------ 落盘
    def save(self, path: str) -> None:
        import joblib

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "threshold": self.threshold,
                "normal_stats": self._normal_stats,
                "params": dict(
                    n_estimators=self.n_estimators, max_samples=self.max_samples,
                    contamination=self.contamination, alpha=self.alpha,
                    embed_dim=self.embed_dim, random_state=self.random_state,
                ),
            },
            path,
        )

    def load(self, path: str) -> "IsolationForestDetector":
        import joblib

        blob = joblib.load(path)
        self.model = blob["model"]
        self.threshold = blob["threshold"]
        self._normal_stats = blob["normal_stats"]
        return self

    def reset(self) -> None:
        return None  # 无状态
