"""评估与指标工具：在多智能体分类上算 Accuracy / Precision / Recall / F1 / Kappa。

两个层次：
1. `evaluate_agents(...)`  在独立测试集上逐条评估（不推进 episode 的随机采样）；
2. `agent_predictions(...)` 常用的"多数投票"聚合，得到集成判定。

对齐论文的评估指标：Accuracy、Precision、Recall、F1-Score、Kappa coefficient、
Detection Rate、Success Rate。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np


# --------------------------------------------------------------------------- 基础指标
def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int, labels: Sequence[str]):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1).astype(np.float64)
    pred_sum = cm.sum(axis=0).astype(np.float64)
    precision = np.divide(tp, pred_sum, out=np.zeros_like(tp), where=pred_sum > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(tp), where=(precision + recall) > 0)

    accuracy = float(tp.sum() / max(len(y_true), 1))
    # Cohen's Kappa
    pe = float((support * pred_sum).sum() / max(len(y_true), 1) ** 2)
    kappa = float((accuracy - pe) / (1 - pe)) if abs(1 - pe) > 1e-12 else 0.0

    # 二分类视角（BENIGN vs ATTACK）
    bin_cm = np.array(
        [
            [cm[0, 0], cm[0, 1:].sum()],
            [cm[1:, 0].sum(), cm[1:, 1:].sum()],
        ],
        dtype=np.float64,
    )
    det_prec = bin_cm[1, 1] / max(bin_cm[1, 1] + bin_cm[0, 1], 1e-9)
    det_rec = bin_cm[1, 1] / max(bin_cm[1, 1] + bin_cm[1, 0], 1e-9)

    return {
        "accuracy": accuracy,
        "macro_precision": float(precision[1:].mean()) if num_classes > 1 else 0.0,
        "macro_recall": float(recall[1:].mean()) if num_classes > 1 else 0.0,
        "macro_f1": float(f1[1:].mean()) if num_classes > 1 else 0.0,
        "weighted_f1": float((f1 * support).sum() / max(support.sum(), 1e-9)),
        "kappa": kappa,
        "detection_rate": float(det_rec),
        "false_alarm_rate": float(bin_cm[0, 1] / max(bin_cm[0].sum(), 1e-9)),
        "per_class": {
            labels[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(num_classes)
        },
        "confusion_matrix": cm,
        "binary_confusion_matrix": bin_cm.astype(np.int64),
        "labels": list(labels),
    }


# --------------------------------------------------------------------------- 智能体评估
@dataclass
class AgentEvalResult:
    """一次评估的完整结果。"""

    agents: List[Dict[str, Any]] = field(default_factory=list)
    vote: Dict[str, Any] = field(default_factory=dict)
    detector: Optional[Dict[str, Any]] = None
    n_samples: int = 0
    rewards: Optional[np.ndarray] = None
    step_info: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [f"评估样本数: {self.n_samples}"]
        lines.append("  ---- 各智能体 ----")
        for a in self.agents:
            dist = a.get("pred_distribution")
            dist_s = ""
            if dist:
                dist_s = "  判定分布=" + "/".join(f"{k}:{v:.2f}" for k, v in dist.items())
            lines.append(
                f"  {a['name']:<20} 专注={a['focus']:<12} "
                f"acc={a['accuracy']:.4f}  F1={a['macro_f1']:.4f}  "
                f"检测率={a['detection_rate']:.4f}  误报率={a['false_alarm_rate']:.4f}"
                + dist_s
            )
        if self.vote:
            lines.append("  ---- 多智能体投票（集成判定）----")
            lines.append(
                f"  投票             acc={self.vote['accuracy']:.4f}  "
                f"F1={self.vote['macro_f1']:.4f}  Kappa={self.vote['kappa']:.4f}  "
                f"检测率={self.vote['detection_rate']:.4f}  误报率={self.vote['false_alarm_rate']:.4f}"
            )
        if self.detector:
            lines.append("  ---- 检测器（防守方机制）----")
            lines.append(
                f"  检测器           det_F1={self.detector['f1']:.4f}  "
                f"检测率={self.detector['detection_rate']:.4f}  "
                f"误报率={self.detector['false_alarm_rate']:.4f}"
            )
        return "\n".join(lines)


def stratified_subset(
    X: np.ndarray, y: np.ndarray, max_samples: Optional[int], seed: int = 0
):
    """
    按类别分层抽样一个评估子集。

    真实 CICIDS 测试集里 BENIGN 占 ~80%，如果直接取前 N 条会全是良性流量，
    评估出来的准确率虚高、F1 恒为 0。这里按类别等比例（每类保底 50 条）
    抽样，保证评估集里一定含各类攻击。
    """
    y = np.asarray(y).astype(int)
    if max_samples is None or max_samples >= len(y):
        return X, y
    rng = np.random.default_rng(seed)
    classes = [int(c) for c in np.unique(y)]
    counts = {c: int((y == c).sum()) for c in classes}
    total = max(sum(counts.values()), 1)
    n_total = int(max_samples)

    per_class = {
        c: min(counts[c], max(int(round(n_total * counts[c] / total)), min(50, counts[c])))
        for c in classes
    }
    # 超预算就按当前分配最多的类逐条裁剪（保持类别都为非空）
    while sum(per_class.values()) > n_total:
        biggest = max(per_class, key=lambda k: (per_class[k], k))
        if per_class[biggest] <= 1:
            break
        per_class[biggest] -= 1

    idx: List[int] = []
    for c in classes:
        pick = np.where(y == c)[0]
        rng.shuffle(pick)
        idx.extend(pick[: per_class[c]].tolist())
    idx_arr = np.array(sorted(idx), dtype=np.int64)
    return np.asarray(X)[idx_arr], y[idx_arr]


def evaluate_agents(
    env,
    act_fn: Callable[[np.ndarray], Any],
    X: np.ndarray,
    y: np.ndarray,
    max_samples: Optional[int] = None,
    stratify: bool = True,
    seed: int = 0,
    verbose: bool = False,
) -> AgentEvalResult:
    """
    在给定数据集上评估多智能体策略。

    act_fn(obs) -> (actions (N, action_dim), class_idx (N,))
        通常是 lambda o: maddpg.select_actions(o, deterministic=True)

    做法：临时把环境的采样序列设成给定的 X/y（保持环境其它逻辑完全不变），
    逐步推进，收集每个智能体的判定与投票结果。
    """
    env = env
    if stratify:
        X, y = stratified_subset(X, y, max_samples, seed=seed)
    n = len(y)
    if max_samples is not None:
        n = min(n, int(max_samples))
    idx = np.arange(n)

    # 临时替换环境的数据视图：这样检测器信号、奖励、观测构造都走原路径
    saved = (env.X_data, env.y_labels, env._episode_idx, env.cfg.shuffle, env.current_idx)
    env.X_data, env.y_labels = X[:n], y[:n]
    env.cfg.shuffle = False
    env._episode_idx = idx

    n_cls = env.num_classes
    labels = env.spec.names
    per_agent_pred = np.zeros((env.num_agents, n), dtype=np.int64)
    det_flag = np.zeros(n, dtype=np.int64)
    det_score = np.zeros(n, dtype=np.float64)
    rewards = np.zeros((env.num_agents, n), dtype=np.float64)

    try:
        env.current_step = 0
        env.stats = env._fresh_stats()
        env.detector.reset()
        for t in range(n):
            env.current_idx = int(idx[t])
            env._refresh_signal()
            obs = env._get_observations()
            actions, cls = act_fn(obs)
            _, rew, done, info = env.step_with_classes(actions, cls)
            per_agent_pred[:, t] = info["class_pred"]
            det_flag[t] = info["detector_flag"]
            det_score[t] = info["detector_score"]
            rewards[:, t] = rew
            if verbose and (t + 1) % 500 == 0:
                print(f"    评估进度 {t + 1}/{n}")
    finally:
        env.X_data, env.y_labels, env._episode_idx, env.cfg.shuffle, env.current_idx = saved

    # 投票：把所有智能体的判定做多数投票（平票时取最"激进"的非 benign 类）
    vote_pred = _majority_vote(per_agent_pred, n_cls)

    y_true = np.asarray(y[:n]).astype(int)
    agents = []
    for i in range(env.num_agents):
        m = classification_metrics(y_true, per_agent_pred[i], n_cls, labels)
        m["name"] = env.agents[i]["name"]
        m["id"] = i
        m["focus"] = env.spec.name_of(int(env.focus[i]))
        m["mean_reward"] = float(rewards[i].mean())
        counts = np.bincount(per_agent_pred[i], minlength=n_cls) / max(n, 1)
        m["pred_distribution"] = {labels[k]: float(counts[k]) for k in range(n_cls)}
        agents.append(m)

    vote_m = classification_metrics(y_true, vote_pred, n_cls, labels)
    vote_m["mean_reward"] = float(rewards.mean())

    y_bin = (y_true != 0).astype(int)
    det_m = classification_metrics(y_bin, det_flag, 2, ["BENIGN", "ATTACK"])

    return AgentEvalResult(
        agents=agents,
        vote=vote_m,
        detector={
            "f1": det_m["macro_f1"],
            "detection_rate": det_m["detection_rate"],
            "false_alarm_rate": det_m["false_alarm_rate"],
            "precision": det_m["per_class"]["ATTACK"]["precision"],
            "recall": det_m["per_class"]["ATTACK"]["recall"],
            "mean_score": float(det_score.mean()),
        },
        n_samples=n,
        rewards=rewards.mean(axis=0),
        step_info={"detector_flags": det_flag, "detector_scores": det_score,
                   "vote_pred": vote_pred, "per_agent_pred": per_agent_pred, "y_true": y_true},
    )


def _majority_vote(preds: np.ndarray, num_classes: int) -> np.ndarray:
    n = preds.shape[1]
    out = np.zeros(n, dtype=np.int64)
    for t in range(n):
        counts = np.bincount(preds[:, t], minlength=num_classes)
        top = int(np.argmax(counts))
        if counts[top] == 1 or (counts == counts.max()).sum() > 1:
            # 平票：优先选择非 BENIGN 的众数类（安全侧偏好）
            tied = np.where(counts == counts.max())[0]
            non_benign = tied[tied != 0]
            top = int(non_benign[0]) if len(non_benign) else 0
        out[t] = top
    return out


def format_report(res: AgentEvalResult) -> str:
    return res.summary()
