"""检测机制 2/3：Kalman Filter（卡尔曼滤波状态估计）异常检测。

模型（对齐 `papers/kalfil.pdf`、`融合状态估计和深度学习的智能电网异常检测模型.pdf`）
--------------------------------------------------------------------------------
把「正常网络流量的特征向量」看作被噪声驱动的隐状态 x_t，用逐维随机游走模型：

    z_t = x_t + v_t ,  v ~ N(0, R)                 观测方程
    x_t = x_{t-1} + w_t , w ~ N(0, Q)              状态方程

递归最小方差估计（Kalman 更新）：

    x_{t|t-1} = x_{t-1|t-1}                        状态预测
    P_{t|t-1} = λ² · P_{t-1|t-1} + Q               协方差预测（λ≤1 为遗忘因子）
    S_t       = P_{t|t-1} + R                      新息协方差
    innovation= z_t − x_{t|t-1}                    新息
    NIS_t     = innovation² / S_t                  归一化新息平方
    K_t       = P_{t|t-1} / S_t                    卡尔曼增益
    x_{t|t}   = x_{t|t-1} + K_t · innovation
    P_{t|t}   = (1 − K_t) · P_{t|t-1}

  * **异常分 = 逐维 NIS 的均值**（近似 χ² 统计量）：滤波器认为"这帧流量不该出现在
    当前正常行为估计附近"时分数升高；
  * **embedding = 有符号标准化新息** innovation/√S：让 MADDPG 的 critic 看到偏离
    **方向**（哪个特征偏高/偏低），而不只是幅度；
  * 有状态、流式：滤波器跨流量保持记忆，`reset()` 在 episode 边界调用。

三种基线模式（`mode`）—— 这是 Kalman 类检测器最容易踩坑的地方
------------------------------------------------------------
    "frozen"  基线冻结在良性训练集上（无时序记忆），检出"偏离历史正常"的流量。
              i.i.d. 的表格数据（如 CICIDS2017 逐条流量）上表现最好。
    "decay"   基线缓慢跟随（缺省）：既能检测突然的偏移，也能适应正常的分布漂移
              (concept drift)。track_rate 控制跟随速度。
    "forget"  基线快速跟随（Q 大 / 等效窗口 = window 条流量）：只检出**突发**偏离。
              注意：如果攻击是持续性的，基线会被攻击拉过去，之后就不再告警——
              这是纯时序跟踪模型的固有局限，做对比实验时值得单独说明。

推荐用法
--------
    det = make_detector("kalman")                       # decay 模式，最稳
    det = make_detector("kalman", mode="frozen")        # 与 IF / AE 公平对比
    det = make_detector("kalman", mode="forget", window=30)   # 模拟"只看近期流量"
"""
from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np

from .base import DetectorBase, DetectorOutput, register_detector

_MODES = ("frozen", "decay", "forget")


@register_detector("kalman")
@register_detector("kf")
class KalmanFilterDetector(DetectorBase):
    def __init__(
        self,
        mode: str = "decay",
        # ---- 状态/观测噪声 ----
        process_noise: Optional[float] = None,      # Q；None -> 由 window / mode 推导
        measurement_noise: Optional[float] = None,  # R；None -> 用良性数据逐维方差估计
        forgetting: float = 1.0,                    # λ，<1 时对旧状态指数遗忘
        initial_covariance: float = 1.0,
        # ---- 基线跟随 ----
        window: float = 200.0,       # mode="forget" 时的等效记忆长度（流量条数）
        track_rate: float = 3e-3,    # mode="decay" 时基线每步跟随比例
        anomaly_track_scale: float = 0.05,  # 告警时跟随速率的缩放（越小越"不信任异常样本"）
        # ---- 标定 / 输出 ----
        alpha: float = 0.05,         # 目标误报率（无标签时按良性分位数定阈值）
        threshold: Optional[float] = None,
        embed_dim: int = 4,          # 用 |新息| 最大的前 k 维作为嵌入
        warmup: int = 30,            # 前 warmup 步只更新状态、不告警
        **_: Any,
    ) -> None:
        if mode not in _MODES:
            raise ValueError(f"mode 必须是 {_MODES} 之一，收到 {mode!r}")
        self.mode = mode
        self._q_override = process_noise
        self._r_override = measurement_noise
        self.forgetting = float(np.clip(forgetting, 1e-3, 1.0))
        self.initial_covariance = float(initial_covariance)
        self.window = float(window)
        self.track_rate = float(track_rate)
        self.anomaly_track_scale = float(anomaly_track_scale)
        self.alpha = float(alpha)
        self.threshold = float(threshold) if threshold is not None else None
        self.embed_dim = int(embed_dim)
        self.warmup = int(warmup)

        self.dim: Optional[int] = None
        self.x: Optional[np.ndarray] = None     # 状态估计（正常行为基线）(d,)
        self.P: Optional[np.ndarray] = None     # 估计协方差 (d,)
        self.Q: Optional[np.ndarray] = None
        self.R: Optional[np.ndarray] = None
        self._step = 0
        self._score_mu = 0.0
        self._score_sd = 1.0

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        X_normal: np.ndarray,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> "KalmanFilterDetector":
        X_normal = np.asarray(X_normal, dtype=np.float32)
        self.dim = int(X_normal.shape[1])

        # 观测噪声 R：良性数据的逐维方差（下限保护）
        if self._r_override is None:
            self.R = np.maximum(X_normal.var(axis=0).astype(np.float64), 1e-8)
        else:
            self.R = np.full(self.dim, float(self._r_override), dtype=np.float64)

        # 过程噪声 Q
        if self._q_override is not None:
            q = float(self._q_override)
        elif self.mode == "forget":
            q = self.initial_covariance / max(self.window, 1.0)
        else:
            # frozen / decay：Q 取小值，让滤波器基本"记住"训练基线
            q = 1e-6 * self.initial_covariance
        self.Q = np.full(self.dim, q, dtype=np.float64)

        # 用良性数据把状态推到稳态（避免 x=0 起步时的巨大瞬时新息）
        self._reset_state(keep_baseline=False)
        burn = min(len(X_normal), max(self.warmup * 5, 500))
        self._run(X_normal[:burn], calibrating=True, update_state=True)

        # 估计"正常状态下 NIS 分布"，用于阈值与 trust 归一化
        self._reset_state(keep_baseline=False)
        scores = self._run(X_normal, calibrating=True, update_state=True)[0]
        self._score_mu = float(scores.mean())
        self._score_sd = float(scores.std() + 1e-8)
        # 保留跑完良性数据后的稳态状态作为 episode 起点基线
        self._baseline = self.x.copy()
        self._reset_state(keep_baseline=True)

        if self.threshold is None:
            s_test = None
            if X_test is not None:
                s_test = self._run(np.asarray(X_test, dtype=np.float32), calibrating=True)[0]
                self._reset_state(keep_baseline=True)
            self.threshold = self._calibrate_threshold(
                scores, alpha=self.alpha, scores_test=s_test, y_test=y_test
            )
        return self

    # ------------------------------------------------------------------ 核心
    def _run(
        self,
        X: np.ndarray,
        calibrating: bool = False,
        update_state: bool = True,
    ):
        """顺序处理一批流量；返回 (score, flag, trust, innovation_norm, steps)。"""
        n = X.shape[0]
        score = np.empty(n, dtype=np.float64)
        flag = np.empty(n, dtype=np.float32)
        trust = np.empty(n, dtype=np.float32)
        innov_norm = np.empty((n, self.dim), dtype=np.float32)
        steps = np.empty(n, dtype=np.int64)
        lam2 = self.forgetting ** 2

        thr = float(self.threshold) if self.threshold is not None else float(
            self._score_mu + 3.0 * self._score_sd
        )
        # 只有 mode="decay" 才有独立的基线跟随项（frozen 不跟随，forget 靠 Q 跟随）
        track = self.mode == "decay" and not calibrating

        for i in range(n):
            z = X[i].astype(np.float64)
            # --- 预测 ---
            P_pred = lam2 * self.P + self.Q
            # --- 新息 ---
            innov = z - self.x
            S = np.maximum(P_pred + self.R, 1e-12)
            nis = (innov * innov) / S
            s = float(np.mean(nis))

            warn = not calibrating and self._step >= self.warmup
            f = 1.0 if (warn and s >= thr) else 0.0
            if calibrating:
                f = 0.0

            zscore = (s - self._score_mu) / self._score_sd
            conf = 1.0 / (1.0 + np.exp(-np.clip(0.5 * zscore, -30.0, 30.0)))
            t = conf if f > 0 else (1.0 - conf)

            score[i] = s
            flag[i] = f
            trust[i] = t
            innov_norm[i] = (innov / np.sqrt(S)).astype(np.float32)
            steps[i] = self._step

            # --- 卡尔曼校正 ---
            if update_state:
                K = P_pred / S
                self.x = self.x + K * innov
                self.P = (1.0 - K) * P_pred

                # --- 基线跟随（模拟正常流量分布的缓慢漂移）---
                if track:
                    gain = self.track_rate * (
                        1.0 if f == 0.0 else self.anomaly_track_scale
                    )
                    self.x = self.x + gain * innov
            self._step += 1

        return score, flag, trust, innov_norm, steps

    def score_batch(self, X: np.ndarray) -> DetectorOutput:
        if self.dim is None or self.x is None:
            raise RuntimeError("KalmanFilterDetector 尚未 fit()")
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X[None, :]
        score, flag, trust, innov_norm, steps = self._run(X)

        emb = None
        if self.embed_dim > 0:
            k = min(self.embed_dim, innov_norm.shape[1])
            order = np.argsort(-np.abs(innov_norm), axis=1)[:, :k]
            emb = np.take_along_axis(innov_norm, order, axis=1).astype(np.float32)
            if k < self.embed_dim:
                emb = np.concatenate(
                    [emb, np.zeros((emb.shape[0], self.embed_dim - k), dtype=np.float32)], axis=1
                )
        return DetectorOutput(
            score=score.astype(np.float32), flag=flag, trust=trust, embedding=emb,
            extra={"threshold": self.threshold, "step": steps, "mode": self.mode},
        )

    # ------------------------------------------------------------------ 状态
    def _reset_state(self, keep_baseline: bool = True) -> None:
        if self.dim is None:
            return
        if keep_baseline and getattr(self, "_baseline", None) is not None:
            self.x = self._baseline.copy()
        else:
            self.x = np.zeros(self.dim, dtype=np.float64)
        self.P = np.full(self.dim, self.initial_covariance, dtype=np.float64)
        self._step = 0

    def reset(self) -> None:
        """episode 边界：清空时序记忆，回到训练得到的稳态基线。"""
        self._reset_state(keep_baseline=True)

    # ------------------------------------------------------------------ 落盘
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez(
            path,
            dim=self.dim, Q=self.Q, R=self.R,
            baseline=getattr(self, "_baseline", self.x),
            threshold=self.threshold, score_mu=self._score_mu, score_sd=self._score_sd,
            mode=self.mode, window=self.window, track_rate=self.track_rate,
            forgetting=self.forgetting, embed_dim=self.embed_dim,
        )

    def load(self, path: str) -> "KalmanFilterDetector":
        if not path.endswith(".npz"):
            path = path + ".npz"
        blob = np.load(path, allow_pickle=False)
        self.dim = int(blob["dim"])
        self.Q = blob["Q"].astype(np.float64)
        self.R = blob["R"].astype(np.float64)
        self._baseline = blob["baseline"].astype(np.float64)
        self.threshold = float(blob["threshold"])
        self._score_mu = float(blob["score_mu"])
        self._score_sd = float(blob["score_sd"])
        self.mode = str(blob["mode"])
        self._reset_state(keep_baseline=True)
        return self
