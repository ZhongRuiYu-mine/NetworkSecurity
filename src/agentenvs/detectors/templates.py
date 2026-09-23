"""
自定义检测器模板 / 示例集合 —— 复制这个文件里的类就能接入自己的检测算法。

注册方式（二选一）：
  1. 在这里加 @register_detector("your_key")，之后 `make_detector("your_key")`
     和 `train.py --detector your_key` 都能直接用；
  2. 不注册也行，直接把实例传给环境：
        env = make_env(ds, detector=YourDetector(...))

自检：
    python -c "from agentenvs.detectors.templates import run_selfcheck; run_selfcheck()"
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .base import DetectorBase, DetectorOutput, register_detector


# ===========================================================================
# 示例 1：最小可用检测器（均值 + 马氏距离），20 行核心代码
# ===========================================================================
@register_detector("zscore_demo")
class ZScoreDemoDetector(DetectorBase):
    """
    最简示例：把每条流量对各特征做标准化，异常分 = 标准化偏离的 L2 范数。

    注意三件事（所有检测器都逃不掉）：
      * 只用良性流量 fit（mu/sigma 都来自良性集）；
      * 分数方向必须是"越高越异常"；
      * 阈值用良性分数的分位数标定，控制误报率。
    """

    embed_dim = 4          # 我们会输出 4 维新息向量作为 embedding

    def __init__(self, alpha: float = 0.05, threshold: Optional[float] = None,
                 embed_dim: int = 4, **_: Any) -> None:
        self.alpha = float(alpha)
        self.threshold = threshold
        self.embed_dim = int(embed_dim)
        self.mu: Optional[np.ndarray] = None
        self.sigma: Optional[np.ndarray] = None

    # ---- 必需 1：在良性流量上训练/标定 ----
    def fit(self, X_normal: np.ndarray, X_test=None, y_test=None, **kw) -> "ZScoreDemoDetector":
        X_normal = np.asarray(X_normal, dtype=np.float64)
        self.mu = X_normal.mean(axis=0)
        self.sigma = np.maximum(X_normal.std(axis=0), 1e-8)

        s_normal = self._raw_score(X_normal)
        if self.threshold is None:
            s_test = self._raw_score(np.asarray(X_test, dtype=np.float64)) if X_test is not None else None
            self.threshold = self._calibrate_threshold(
                s_normal, self.alpha, scores_test=s_test, y_test=y_test
            )
        return self

    # ---- 必需 2：打分 ----
    def _raw_score(self, X: np.ndarray) -> np.ndarray:
        z = (np.asarray(X, dtype=np.float64) - self.mu) / self.sigma
        return np.sqrt((z ** 2).mean(axis=1)).astype(np.float32)

    def score_batch(self, X: np.ndarray) -> DetectorOutput:
        if self.mu is None:
            raise RuntimeError("尚未 fit()")
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X[None, :]
        z = ((X - self.mu) / self.sigma).astype(np.float32)
        score = np.sqrt((z ** 2).mean(axis=1)).astype(np.float32)
        thr = float(self.threshold) if self.threshold is not None else 3.0
        flag = (score >= thr).astype(np.float32)
        trust = np.clip(score / max(thr, 1e-8) / 2.0, 0.0, 1.0).astype(np.float32)
        emb = None
        if self.embed_dim > 0:
            k = min(self.embed_dim, z.shape[1])
            order = np.argsort(-np.abs(z), axis=1)[:, :k]
            emb = np.take_along_axis(z, order, axis=1)
            if k < self.embed_dim:
                emb = np.pad(emb, ((0, 0), (0, self.embed_dim - k)))
        return DetectorOutput(score=score, flag=flag, trust=trust, embedding=emb)

    # ---- 可选：落盘（强烈建议，避免每次重训）----
    def save(self, path: str) -> None:
        np.savez(path, mu=self.mu, sigma=self.sigma, threshold=self.threshold)

    def load(self, path: str) -> "ZScoreDemoDetector":
        blob = np.load(path if path.endswith(".npz") else path + ".npz")
        self.mu, self.sigma = blob["mu"], blob["sigma"]
        self.threshold = float(blob["threshold"])
        return self


# ===========================================================================
# 示例 2：有状态 + 时序的检测器骨架（比如你自己的卡尔曼变体 / 在线学习模型）
# ===========================================================================
@register_detector("stateful_demo")
class StatefulDemoDetector(DetectorBase):
    """
    演示"有状态"检测器要写的三件事：
      * score_batch 里可以跨调用维持内部状态（滤波器、滑动窗口、在线聚类…）；
      * reset() 在**每个 episode 开始**时被环境调用，用来清记忆；
      * 标定阶段需要用内部副本跑一遍，别污染 episode 起点状态。
    """

    embed_dim = 0

    def __init__(self, alpha: float = 0.05, ema: float = 0.01, embed_dim: int = 0, **_: Any) -> None:
        self.alpha = float(alpha)
        self.ema = float(ema)
        self.embed_dim = int(embed_dim)
        self.baseline: Optional[np.ndarray] = None
        self.threshold: Optional[float] = None
        self._saved_baseline: Optional[np.ndarray] = None
        self._score_mu = 0.0
        self._score_sd = 1.0

    def fit(self, X_normal: np.ndarray, X_test=None, y_test=None, **kw) -> "StatefulDemoDetector":
        X_normal = np.asarray(X_normal, dtype=np.float32)
        self._saved_baseline = X_normal.mean(axis=0)          # 良性基线
        self.reset()

        # 标定：用内部状态跑一遍良性数据
        scores = np.array([self._update(x[None, :])[0] for x in X_normal], dtype=np.float64)
        self._score_mu, self._score_sd = float(scores.mean()), float(scores.std() + 1e-8)
        self.reset()                                          # 标定完复位
        if self.threshold is None:
            self.threshold = self._calibrate_threshold(scores, self.alpha,
                                                       scores_test=None, y_test=y_test)
        return self

    def _update(self, X: np.ndarray) -> np.ndarray:
        """顺序更新内部状态，返回本批的异常分。"""
        out = np.empty(X.shape[0], dtype=np.float32)
        for i, x in enumerate(X):
            out[i] = float(np.sqrt(((x - self.baseline) ** 2).mean()))
            self.baseline = (1 - self.ema) * self.baseline + self.ema * x   # 缓慢跟随
        return out

    def score_batch(self, X: np.ndarray) -> DetectorOutput:
        if self.baseline is None:
            raise RuntimeError("尚未 fit()")
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X[None, :]
        score = self._update(X)
        thr = float(self.threshold) if self.threshold is not None else float(
            self._score_mu + 3 * self._score_sd
        )
        flag = (score >= thr).astype(np.float32)

        z = (score - self._score_mu) / self._score_sd
        conf = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        trust = np.where(flag > 0, conf, 1.0 - conf).astype(np.float32)
        return DetectorOutput(score=score, flag=flag, trust=trust)

    def reset(self) -> None:
        """episode 边界：清空时序记忆。"""
        if self._saved_baseline is not None:
            self.baseline = self._saved_baseline.copy()


# ===========================================================================
# 示例 3：包装已有检测代码（不用改原代码，只做适配）
# ===========================================================================
class WrapExistingModel(DetectorBase):
    """
    把任意已有的检测器包成契约对象。适用于：你已经有一份 `model.score(X)`。

        det = WrapExistingModel(my_model, score_fn=lambda m, X: -m.decision_function(X),
                                flag_fn=lambda m, X: (m.predict(X) == -1).astype(float),
                                embed_dim=0)
        env = make_env(ds, detector=det)
    """

    def __init__(
        self,
        model: Any,
        score_fn,
        flag_fn=None,
        trust_fn=None,
        embed_fn=None,
        embed_dim: int = 0,
        fit_fn=None,
        alpha: float = 0.05,
        threshold: Optional[float] = None,
        **_: Any,
    ) -> None:
        self.model = model
        self.score_fn, self.flag_fn = score_fn, flag_fn
        self.trust_fn, self.embed_fn, self.fit_fn = trust_fn, embed_fn, fit_fn
        self.embed_dim = int(embed_dim)
        self.alpha = float(alpha)
        self.threshold = threshold
        self._normal_stats = None

    def fit(self, X_normal: np.ndarray, X_test=None, y_test=None, **kw) -> "WrapExistingModel":
        if self.fit_fn is not None:
            self.fit_fn(self.model, X_normal)
        s = np.asarray(self.score_fn(self.model, np.asarray(X_normal)), dtype=np.float64).ravel()
        self._normal_stats = (float(s.mean()), float(s.std() + 1e-8))
        if self.threshold is None:
            s_test = None
            if X_test is not None:
                s_test = np.asarray(self.score_fn(self.model, np.asarray(X_test))).ravel()
            self.threshold = self._calibrate_threshold(s, self.alpha, s_test, y_test)
        return self

    def score_batch(self, X: np.ndarray) -> DetectorOutput:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X[None, :]
        score = np.asarray(self.score_fn(self.model, X), dtype=np.float32).ravel()
        if self.flag_fn is not None:
            flag = np.asarray(self.flag_fn(self.model, X), dtype=np.float32).ravel()
        else:
            thr = float(self.threshold) if self.threshold is not None else float(
                (self._normal_stats or (0.0, 1.0))[0]
            )
            flag = (score >= thr).astype(np.float32)
        if self.trust_fn is not None:
            trust = np.asarray(self.trust_fn(self.model, X), dtype=np.float32).ravel()
        else:
            mu, sd = self._normal_stats or (float(score.mean()), float(score.std() + 1e-8))
            conf = 1.0 / (1.0 + np.exp(-np.clip((score - mu) / sd, -30, 30)))
            trust = np.where(flag > 0, conf, 1.0 - conf).astype(np.float32)
        emb = None
        if self.embed_dim > 0 and self.embed_fn is not None:
            emb = np.asarray(self.embed_fn(self.model, X), dtype=np.float32)
            if emb.ndim == 1:
                emb = emb.reshape(-1, 1)
        return DetectorOutput(score=score, flag=flag, trust=trust, embedding=emb)


# ===========================================================================
# 自检：确认模板可以直接接入环境
# ===========================================================================
def run_selfcheck(num_agents: int = 3, max_steps: int = 20) -> int:
    """把上面三个示例接进环境各跑一遍，全部通过返回 0。"""
    from agentenvs import EnvConfig, make_synthetic
    from agentenvs.cyber_defense_env import CyberDefenseEnv
    from .base import make_detector

    ds = make_synthetic(n_normal=600, n_per_class=150, feat_dim=20, num_attack_classes=4)
    y_bin = (ds.y_test != 0).astype(int)
    from sklearn.metrics import roc_auc_score

    checks = {
        "zscore_demo": make_detector("zscore_demo", embed_dim=4),
        "stateful_demo": make_detector("stateful_demo", embed_dim=4),
    }

    for name, det in checks.items():
        det.fit(ds.X_normal_train, X_test=ds.X_test, y_test=y_bin)
        out = det.score_batch(ds.X_test[:256])
        auc = roc_auc_score(y_bin[:256], out.score)
        print(f"  [OK] {name:<16} signal_dim={det.signal_dim} AUC(前256条)={auc:.3f}")

        env = CyberDefenseEnv(
            X_data=ds.X_train, y_labels=ds.y_train, detector=det, spec=ds.spec,
            config=EnvConfig(num_agents=num_agents, max_steps=max_steps, seed=0, detector_embed_dim=4),
            X_normal_for_detector=ds.X_normal_train, auto_fit_detector=False,
        )
        obs = env.reset()
        assert obs.shape == (num_agents, env.obs_dim), obs.shape
        a = np.zeros((num_agents, env.action_dim), dtype=np.float32)
        c = np.zeros(num_agents, dtype=np.int64)
        next_obs, rew, done, info = env.step_with_classes(a, c)
        assert rew.shape == (num_agents,)
        print(f"       env: obs_dim={env.obs_dim} 动作维度={env.action_dim} 单步奖励={rew.round(3).tolist()}")

    # 包装已有模型的示例
    class _FakeModel:
        def fit(self, X): self.mu = X.mean(axis=0)
        def score(self, X): return np.sqrt(((X - self.mu) ** 2).mean(axis=1))

    m = _FakeModel()
    wrapped = WrapExistingModel(m, score_fn=lambda mm, X: mm.score(X), embed_dim=4,
                                fit_fn=lambda mm, X: mm.fit(X),
                                embed_fn=lambda mm, X: mm.score(X)[:, None] * np.ones((len(X), 4)))
    wrapped.fit(ds.X_normal_train, X_test=ds.X_test, y_test=y_bin)
    print(f"  [OK] WrapExistingModel signal_dim={wrapped.signal_dim}")
    print("  模板自检通过 ✅")
    return 0


if __name__ == "__main__":
    # Windows 控制台默认 GBK，下面的 ✅ 会触发 UnicodeEncodeError 让自检**假失败**；
    # 这里显式把 stdout 切成 UTF-8（不支持时用 replace 兜底）。
    import sys as _sys

    try:
        _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    raise SystemExit(run_selfcheck())
