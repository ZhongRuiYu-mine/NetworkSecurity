"""
防守方（检测器）插件接口 —— 这是本项目里唯一需要为「三种检测机制」改动的地方。

设计目标
--------
MADDPG 环境 **不关心** 检测算法内部是怎么算的，它只消费一个固定契约：

    DetectorBase.fit(X_normal)          -> 在良性流量上训练/标定（可选）
    DetectorBase.score_batch(X)         -> DetectorOutput
    DetectorBase.signal_dim             -> 固定维度的信号向量长度

只要实现了这个契约，IF / Kalman / Autoencoder / 或者任何自定义检测算法
都可以直接通过 `make_detector("your_name", cfg)` 或直接传对象给环境使用。

统一的信号向量 (signal vector)
------------------------------
    [0] anomaly_score      连续异常分（越高越异常，已做单调方向校正）
    [1] anomaly_flag       0/1 二值告警（由阈值产生）
    [2] trust              该检测器在当前输入上的可信度 in [0,1]
                           （Kalman 用 NIS 一致性，IF/AE 用分数显著度）
    [3..3+embed_dim)       可选的 embedding（IF 无 / AE 用重构残差向量 /
                           Kalman 用标准化新息向量）

环境会把这个信号向量拼到每个智能体的观测后面，因此**所有检测器的信号维度
必须一致**（由 `signal_dim` 决定，环境启动时会校验）。
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

# 信号向量的公共布局，环境按这个约定切片，请勿随意改动。
SIGNAL_SCORE = 0
SIGNAL_FLAG = 1
SIGNAL_TRUST = 2
SIGNAL_TAIL = 3  # embedding 从这一维开始


@dataclass
class DetectorOutput:
    """检测器对一批流量的输出（契约对象，环境只读）。"""

    score: np.ndarray            # (n,) 连续异常分，越高越异常
    flag: np.ndarray             # (n,) {0,1} 二值告警
    trust: np.ndarray            # (n,) [0,1] 可信度
    embedding: Optional[np.ndarray] = None   # (n, k) 或 None
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
        """打包成环境使用的固定维度信号矩阵 (n, 3 + embed_dim)。"""
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
    """
    所有防守方检测器的基类。

    子类必须实现:
        name         -> 字符串标识
        fit(...)     -> 在良性流量上训练/标定
        score_batch  -> DetectorOutput

    可选覆写:
        signal_dim   -> 默认 3 + embed_dim
        reset()      -> 有状态检测器（如 Kalman）在 episode 边界重置
        save/load    -> 落盘已训练好的模型（强烈建议实现，避免每次重训）
    """

    #: 子类声明自己的 embedding 维度，0 表示没有 embedding
    embed_dim: int = 0
    #: 内置检测器名（注册表 key），自定义检测器可留空
    name: str = "detector"

    # ------------------------------------------------------------------ 核心接口
    @abc.abstractmethod
    def fit(self, X_normal: np.ndarray, **kwargs: Any) -> "DetectorBase":
        """用良性流量拟合。允许 no-op（例如纯阈值法）。返回 self 便于链式调用。"""
        raise NotImplementedError

    @abc.abstractmethod
    def score_batch(self, X: np.ndarray) -> DetectorOutput:
        """对 (n, d) 流量特征打分。必须返回 DetectorOutput。"""
        raise NotImplementedError

    # ------------------------------------------------------------------ 便捷封装
    def score(self, x: np.ndarray) -> DetectorOutput:
        """单条流量打分（环境内部按 batch 调用，这里只是方便调试）。"""
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        return self.score_batch(x)

    def detect(self, X: np.ndarray) -> Dict[str, float]:
        """
        兼容旧接口（原 `cyber_defense_env.py` 里用的 `detector.detect(X)`），
        返回首条流量的 dict 形式，方便沿用旧脚本。
        """
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
        """有状态的检测器在此清理内部状态（每个 episode 开始时调用）。"""

    # ------------------------------------------------------------------ 落盘
    def save(self, path: str) -> None:
        raise NotImplementedError(f"{type(self).__name__} 未实现 save()")

    def load(self, path: str) -> "DetectorBase":
        raise NotImplementedError(f"{type(self).__name__} 未实现 load()")

    # ------------------------------------------------------------------ 阈值工具
    @staticmethod
    def _calibrate_threshold(
        scores_normal: np.ndarray,
        alpha: float = 0.05,
        scores_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
    ) -> float:
        """
        选阈值：
          1. 有测试集标签 -> 直接扫 F1 最大阈值；
          2. 否则 -> 用良性分数的高分位（默认 95%），控制误报率约等于 alpha。
        """
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


# ---------------------------------------------------------------------- 注册表
DETECTOR_REGISTRY: Dict[str, type] = {}


def register_detector(key: str):
    """装饰器：把检测器注册进全局注册表，`make_detector(key)` 即可构造。"""

    def _wrap(cls: type) -> type:
        DETECTOR_REGISTRY[key] = cls
        cls.name = key
        return cls

    return _wrap


def list_detectors() -> List[str]:
    return sorted(DETECTOR_REGISTRY)


def make_detector(key: str, **kwargs: Any) -> DetectorBase:
    """
    工厂函数 —— 实验里切换检测机制的**唯一入口**。

        det = make_detector("if", contamination=0.05)
        det = make_detector("kalman", process_noise=1e-4)
        det = make_detector("autoencoder", latent_dim=8, epochs=20)
    """
    if key not in DETECTOR_REGISTRY:
        raise KeyError(
            f"未知检测器 {key!r}，已注册：{list_detectors()}。"
            f"自定义检测器请用 @register_detector('{key}') 注册，或直接把对象传给环境。"
        )
    return DETECTOR_REGISTRY[key](**kwargs)


def coerce_detector(det: Any, **kwargs: Any) -> DetectorBase:
    """接受 str / DetectorBase / None，统一成 DetectorBase 实例。"""
    if det is None:
        return make_detector("null", **kwargs)
    if isinstance(det, str):
        return make_detector(det, **kwargs)
    if isinstance(det, DetectorBase):
        return det
    # 鸭子类型：只要实现了 score_batch/detect 也放行，便于对接你已有的检测代码
    if hasattr(det, "score_batch") or hasattr(det, "detect"):
        return det  # type: ignore[return-value]
    raise TypeError(f"无法识别的检测器对象：{type(det)!r}")
