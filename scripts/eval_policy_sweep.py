# -*- coding: utf-8 -*-
r"""eval_policy_sweep.py —— 逐攻击样本评估"检测→判定"成功率，并做失败归因分解。

为什么需要它
------------
原来的 `eval_if_only.py / eval_all_detectors.py` 只回答"检测器打分好不好"（P/R/F1/AUC），
`train.py` 只回答"MADDPG 策略好不好"，而且实测后者**已经退化**（见 `results/`）：
5 个智能体里 4 个塌缩到单一类别，投票准确率 0.0754、macro-F1 0.0009、Kappa -0.0219。

所以"每一次攻击它都有什么反应""成功率多少""失败原因是什么"这类问题，
用现有产物**一个都答不出来**。本脚本补的就是这一层：把"一条攻击流量"
拆成一条可归因的链路

    目标 i（真实标签）
      └─ 检测器是否告警 det_flag_i        —— 机制层
      └─ 决策层输出类别 ŷ_i               —— 策略层
           ├─ 告警∩判对     → 成功
           ├─ 告警∩判错     → 误分类（分给谁了？）
           ├─ 漏报∩判对     → 检测器拖累（策略本来能对）
           └─ 漏报∩判错     → 双重失败

并额外给出与"成功率"配套、但更诚实的两个量：

  * `accuracy_when_alerted` —— **检测器告警时的类别判定正确率**。
    这是"在给定检测器质量下，判定规则还能做多好"的直接答案，
    也是把失败归因给「机制层」还是「判定层」的关键。
  * `majority_share` —— 判定结果里占比最大的那一类的比例。
    `> 0.8` 即认为该规则已退化成常数输出（塌缩）。

三个判定规则（都**不训练**，纯推理，可复现）
------------------------------------------
| policy        | 定义                                  | 读法 |
|---|---|---|
| `detector`    | 告警 → 先验攻击类，不告警 → BENIGN     | "完全信任检测器"会怎样 |
| `specialist`  | 告警 → 该智能体自己的 `focus` 类       | 逐智能体的专精上限；另有 5 体投票 |
| `benign_only` | 恒判 BENIGN                           | **退化下界**（成功率按定义恒为 0） |

`benign_only` 是关键对照：它**什么都不判**。任何 > 0 的成功率都来自
"检测器告警 + 专精分工"，不是运气。

> 早期版本曾有一个 `oracle` 规则（完美检测器 + 先验攻击类），实测它输出与
> 常量规则完全一致，只是 flag 不同 —— 那种"上界"是假的，已删除。
> 现在**不构造假的完美检测器上界**；要看上界请直接读 `accuracy_when_alerted`。

产物
----
    results/policy_sweep.json    逐机制 / 逐判定规则 / 逐类的全部数字（含混淆矩阵）
    results/policy_sweep.md      可直接贴进报告的表

用法
----
    $env:PYTHONPATH = "$PWD\src"
    python scripts/eval_policy_sweep.py                        # 全套
    python scripts/eval_policy_sweep.py --detectors kalman_frozen,null
    python scripts/eval_policy_sweep.py --smoke                # 小样本自检
    python scripts/eval_policy_sweep.py --policies detector,benign_only

口径声明
--------
* 本脚本用的是**协议 v1** 的数据包（`PROTOCOL.md` §1：评估集已排除 Monday）。
  它**不能**复现旧 docx 表 5.1 的数字（那套是 29000 行 + F1 扫描口径）。
* 决策层是"无训练常量规则"，因此这里的成功率**不含任何策略学习**。
  MADDPG 的真实成功率请用 `scripts/train.py` 训练后另行评估（见 README §6）。
"""
from __future__ import annotations

# --- 合并仓库引导（scripts/_bootstrap） ---
import os as _os, sys as _sys
REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_os.path.join(REPO_ROOT, "src"), REPO_ROOT):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
del _os, _sys, _p
# --- 合并仓库引导结束 ---

import argparse
import json
import os
import platform as _platform
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from _repo import ARTIFACTS, BUNDLE, RESULTS

#: 机制清单：显示名 -> (注册表 key, 构造参数, 是否可从 artifacts 加载)
DETECTORS: Dict[str, Dict[str, Any]] = {
    "if":            {"key": "if",          "kwargs": {},                   "artifact": None},
    "kalman_frozen": {"key": "kalman",      "kwargs": {"mode": "frozen"},   "artifact": None},
    "kalman_decay":  {"key": "kalman",      "kwargs": {"mode": "decay"},    "artifact": None},
    "autoencoder":   {"key": "autoencoder", "kwargs": {},                   "artifact": None},
    "ae_paper":      {"key": "ae_paper",    "kwargs": {},
                      "artifact": "ae_paper.npz"},
    "null":          {"key": "null",        "kwargs": {},                   "artifact": None},
}
DEFAULT_ORDER = ["if", "kalman_frozen", "autoencoder", "ae_paper", "null"]
ALL_POLICIES = ["detector", "specialist", "benign_only"]
EPS = 1e-9


# --------------------------------------------------------------------------- 决策层
class Policy:
    """无训练的决策规则：把「检测器给了什么证据」翻译成「类别判定」。

    设计原则：**只保留能给出非退化数字的规则**。
    早期版本里有个 `oracle`（完美检测器 + 先验攻击类），实测它和 `prior`
    输出完全一样（都恒为一个常数类），只是 flag 不同 —— 这种"上界"是假的，
    所以删掉了。现在三条规则分别对应三种真实的读法：

      * `detector`   —— 机制层的翻译：告警就说"是先验攻击类"。
                        回答"如果完全信任检测器会怎样"。
      * `specialist` —— 每个智能体按自己的 `focus` 判类（**逐智能体**）。
                        回答"一个专精型智能体在给定检测器下能打到多少"，
                        也是"每一次攻击是什么反应"的类别级答案。
      * `majority`   —— 恒为训练集多数类（=BENIGN）。
                        这是**什么都不学**的下界；策略比它还差 = 已退化。
    """

    name = "policy"

    def __init__(self, num_classes: int, embed_dim: int = 4) -> None:
        self.K = int(num_classes)
        self.embed_dim = int(embed_dim)
        self.attack_mode: int = 0     # 训练集先验最频繁的攻击类
        self.any_mode: int = 0        # 训练集先验最频繁的类（通常 = BENIGN）
        self.focus: Optional[np.ndarray] = None   # (num_agents,)

    def fit_prior(self, y_train: np.ndarray) -> None:
        y = np.asarray(y_train).astype(int)
        cnt = np.bincount(y, minlength=self.K)
        self.any_mode = int(np.argmax(cnt))
        attack = cnt.copy()
        attack[0] = -1
        self.attack_mode = int(np.argmax(attack)) if attack.max() > 0 else 0

    def set_focus(self, focus: np.ndarray) -> None:
        self.focus = np.asarray(focus).astype(np.int64)

    def decide(self, flag: np.ndarray, agent_idx: int = 0) -> np.ndarray:
        """返回 (n,) 的类别判定。`agent_idx` 只在 `specialist` 里有意义。"""
        raise NotImplementedError


class DetectorPolicy(Policy):
    """告警 → 先验攻击类；不告警 → BENIGN。"""

    name = "detector"

    def decide(self, flag, agent_idx: int = 0):
        return np.where(np.asarray(flag) > 0, self.attack_mode, 0).astype(np.int64)


class SpecialistPolicy(Policy):
    """告警 → 该智能体自己的专注类；不告警 → BENIGN。"""

    name = "specialist"

    def decide(self, flag, agent_idx: int = 0):
        focus = int(self.focus[agent_idx]) if self.focus is not None else self.attack_mode
        return np.where(np.asarray(flag) > 0, focus, 0).astype(np.int64)


class MajorityPolicy(Policy):
    """恒判 BENIGN —— 真正的「什么都不判」下界。

    ★ 这里刻意**不用** `np.argmax(先验)`：训练集里 DoS(11717) 比 BENIGN(11062) 还多，
    argmax 会得到 DoS，于是这个"退化基线"会白捡一堆 DoS 的成功率，
    看上去跟 `detector` 一样好 —— 第一次冒烟就是这样，属于把基线做假了。
    恒判 BENIGN 的成功率**按定义恒为 0**，才是干净的对照。
    """

    name = "benign_only"

    def decide(self, flag, agent_idx: int = 0):
        return np.zeros(len(flag), dtype=np.int64)


def make_policy(name: str, num_classes: int, embed_dim: int) -> Policy:
    table = {
        "detector": DetectorPolicy,
        "specialist": SpecialistPolicy,
        "benign_only": MajorityPolicy,
        # 兼容旧名（早期版本用过 majority）
        "majority": MajorityPolicy,
    }
    if name not in table:
        raise ValueError(f"未知 policy {name!r}，可选 {sorted(table)}")
    return table[name](num_classes, embed_dim)


# --------------------------------------------------------------------------- 工具
def det_arrays_from_env(env) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把当前检测器输出拉成裸 numpy（flag 直接取阈值判定结果）。"""
    out = env._detector_out
    flag = np.asarray(out.flag).astype(int).ravel()
    score = np.asarray(out.score, dtype=np.float64).ravel()
    if out.embedding is None:
        emb = np.zeros((len(score), 0), dtype=np.float64)
    else:
        emb = np.asarray(out.embedding, dtype=np.float64)
        if emb.ndim == 1:
            emb = emb[:, None]
    return flag, score, emb


def run_detector_pass(env, X: np.ndarray, reset: bool = True) -> Dict[str, np.ndarray]:
    """顺序把检测器跑一遍 X（批量推理，不做任何策略决策）。

    ★ 必须顺序：Kalman 是有状态检测器，顺序不同结果就不同。
    ★ 必须临时把 `env.X_data` 换成 X：`_refresh_signal()` 读的是 `X_data[current_idx]`，
      直接改 `current_idx` 会拿训练集的行去打评估集的标签（第一次写就踩了这个坑）。
    """
    X = np.asarray(X, dtype=np.float32)
    n = len(X)
    det_flag = np.zeros(n, dtype=np.int64)
    det_score = np.zeros(n, dtype=np.float64)
    embs: List[np.ndarray] = []
    trust = np.zeros(n, dtype=np.float64)

    saved_X, saved_idx = env.X_data, env.current_idx
    env.X_data = X
    if reset:
        env.detector.reset()
    try:
        for i in range(n):
            env.current_idx = i
            env._refresh_signal()
            f, s, e = det_arrays_from_env(env)
            det_flag[i] = int(f[0])
            det_score[i] = float(s[0])
            trust[i] = float(np.asarray(env._detector_out.trust).ravel()[0])
            embs.append(e[0] if e.size else np.zeros(0))
    finally:
        env.X_data, env.current_idx = saved_X, saved_idx
    dim = embs[0].size if embs else 0
    emb = np.asarray(embs, dtype=np.float64) if dim else np.zeros((n, 0))
    return {"det_flag": det_flag, "det_score": det_score, "det_emb": emb,
            "det_trust": trust}


def calib_n(smoke: bool) -> Optional[int]:
    """训练集工作点用多少行。冒烟模式降采样（Kalman 顺序跑 3.7 万行很慢）。"""
    return 6000 if smoke else None


def sample_ordered(X: np.ndarray, y: np.ndarray, max_n: Optional[int],
                   seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """按类别等比例抽样，但**返回时按原始下标排序**。

    为什么不用 `X[:n]`：数据包是按天拼接的（Tuesday→Friday），取前缀只会拿到一天，
    指标随当天的攻击构成剧烈漂移，看起来像检测器坏了。
    为什么不用打乱顺序：Kalman 是**有状态**检测器，顺序本身就是输入的一部分；
    打乱会让评估集与训练集的状态轨迹不同，串起一条假的"漂移"。
    """
    y = np.asarray(y).astype(int)
    if max_n is None or max_n >= len(y):
        return np.asarray(X, dtype=np.float32), y
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    per_class = max(1, int(max_n) // max(len(classes), 1))
    idx: List[int] = []
    for c in classes:
        pool = np.where(y == c)[0]
        take = min(per_class, len(pool))
        idx.extend(rng.choice(pool, size=take, replace=False).tolist())
    idx_arr = np.array(sorted(idx), dtype=np.int64)
    return np.asarray(X, dtype=np.float32)[idx_arr], y[idx_arr]


def env_summary(env) -> Dict[str, Any]:
    """环境的静态形状与分工（对齐平台自检里打印的那几项）。"""
    return {
        "num_agents": int(env.num_agents),
        "num_classes": int(env.num_classes),
        "obs_dim": int(env.obs_dim),
        "action_dim": int(env.action_dim),
        "num_responses": int(env.num_responses),
        "feature_dim": int(env.feat_dim),
        "detector_signal_dim": int(env.detector_signal_dim),
        "agent_focus": [str(env.spec.name_of(int(f))) for f in env.focus],
        "agent_names": [a["name"] for a in env.agents],
    }


def collect_train_pass(env, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """在带标签训练集上跑一遍：标定 prior，并统计检测器的经验工作点。"""
    det = run_detector_pass(env, X, reset=True)
    y = np.asarray(y).astype(int)
    is_attack = (y != 0)
    f = det["det_flag"].astype(bool)
    tp = int((f & is_attack).sum())
    fp = int((f & ~is_attack).sum())
    fn = int((~f & is_attack).sum())
    tn = int((~f & ~is_attack).sum())
    prec = tp / (tp + fp + EPS)
    rec = tp / (tp + fn + EPS)
    return {
        "det": det,
        "y": y,
        "detector_operating_point": {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / (prec + rec + EPS),
            "false_positive_rate": fp / (fp + tn + EPS),
            "flag_rate": float(f.mean()),
        },
    }


# --------------------------------------------------------------------------- 单机制评估
def evaluate_policy(env, policy: Policy, y: np.ndarray,
                    det: Dict[str, np.ndarray], class_names: List[str],
                    agent_id: int = 0) -> Dict[str, Any]:
    """单个 (机制, 决策层, 智能体) 的类别判定评估。

    `det` 必须已经按 **同一顺序**跑完（见 `run_detector_pass`），本函数只做决策 + 统计。

    三个口径的量刻意分开，不要混：
      * `success_rate_on_attacks` —— 告警 ∩ 判对，**部署视角**的成功率；
      * `accuracy_when_alerted`  —— 只在告警样本上算类别正确率，
        **它是判定"这次失败该归给检测器还是归给判定规则"的关键**；
      * 对比 `majority`（恒判 BENIGN）那一行 —— 它的成功率恒为 0，
        所以任何 > 0 的成功率都来自"检测器 + 专精分工"，不是运气。
    """
    y = np.asarray(y).astype(int)
    K = env.num_classes
    n = len(y)

    flag = det["det_flag"]
    pred = np.asarray(policy.decide(flag, agent_idx=agent_id), dtype=np.int64)

    # ---- 混淆矩阵 ----
    cm = np.zeros((K, K), dtype=np.int64)
    for t, p in zip(y, pred):
        cm[int(t), int(p)] += 1
    support = cm.sum(axis=1)
    pred_sum = cm.sum(axis=0)
    tp = np.diag(cm).astype(np.float64)
    recall = np.divide(tp, support, out=np.zeros(K), where=support > 0)
    precision = np.divide(tp, pred_sum, out=np.zeros(K), where=pred_sum > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(K), where=(precision + recall) > 0)
    acc = float(tp.sum() / max(n, 1))
    macro_f1 = float(f1[1:].mean()) if K > 1 else 0.0
    # Kappa
    pe = float((support * pred_sum).sum() / max(n, 1) ** 2)
    kappa = float((acc - pe) / (1 - pe)) if abs(1 - pe) > 1e-12 else 0.0
    majority_share = float(pred_sum.max() / max(n, 1))

    # ---- 告警条件下的类别正确率（"是谁的锅"的核心量）----
    alerted = flag > 0
    n_alerted = int(alerted.sum())
    acc_when_alerted = float((pred[alerted] == y[alerted]).mean()) if n_alerted else 0.0
    attack_mask = (y != 0)
    n_attack = int(attack_mask.sum())
    # 攻击且告警 → 类别正确（真·成功）
    success = int(((pred == y) & alerted & attack_mask).sum())
    success_rate_on_attacks = success / max(n_attack, 1)

    # ---- 失败归因（只在攻击样本上分解）----
    det_hit = alerted & attack_mask          # 检测器告警
    cls_hit = (pred == y) & attack_mask      # 类别判对
    decomp = {
        "success_alerted_and_correct": int((det_hit & cls_hit).sum()),
        "class_error_alerted": int((det_hit & ~cls_hit).sum()),
        "detector_missed_class_correct": int((~det_hit & cls_hit).sum()),
        "double_failure": int(((~det_hit) & (~cls_hit) & attack_mask).sum()),
        "n_attack": n_attack,
    }

    # ---- 逐真实类别 ----
    per_class: Dict[str, Any] = {}
    for i, name in enumerate(class_names):
        m = (y == i)
        cnt = int(m.sum())
        if cnt == 0:
            per_class[name] = {"n": 0}
            continue
        alerted_m = flag[m] > 0
        entry = {
            "n": cnt,
            "pred_distribution": {
                class_names[j]: float((pred[m] == j).mean()) for j in range(K)
            },
            "class_accuracy": float((pred[m] == y[m]).mean()),
            "class_recall": float(recall[i]),
            "precision": float(precision[i]),
            "f1": float(f1[i]),
            "alert_rate": float(alerted_m.mean()),
            "accuracy_when_alerted": (
                float((pred[m][alerted_m] == y[m][alerted_m]).mean())
                if alerted_m.any() else 0.0
            ),
            "success_rate": float(((pred[m] == y[m]) & alerted_m).mean()),
        }
        if i != 0:
            entry["alerted_and_correct"] = int(((pred[m] == y[m]) & alerted_m).sum())
            entry["alerted_and_wrong"] = int(((pred[m] != y[m]) & alerted_m).sum())
            entry["missed_and_correct"] = int(((pred[m] == y[m]) & ~alerted_m).sum())
            entry["missed_and_wrong"] = int(((pred[m] != y[m]) & ~alerted_m).sum())
        per_class[name] = entry

    return {
        "policy": policy.name,
        "n": n,
        "n_attack": n_attack,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "kappa": kappa,
        "majority_share": majority_share,
        "success_rate_on_attacks": success_rate_on_attacks,
        "n_success": success,
        "accuracy_when_alerted": acc_when_alerted,
        "n_alerted": n_alerted,
        "flag_rate": float(alerted.mean()),
        "flag_agreement_with_class_pred":
            float(((pred != 0).astype(int) == alerted.astype(int)).mean()),
        "decomposition": decomp,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "pred_distribution": {
            class_names[j]: float((pred == j).mean()) for j in range(K)
        },
    }


# --------------------------------------------------------------------------- 主流程
def build_detector(name: str, embed_dim: int, seed: int | None):
    """优先从 `artifacts/` 加载已训练检测器；找不到就返回 (None, None) 让调用方现训。

    先查 `artifacts/ae_paper.npz` 能省掉 `ae_paper` 那 600 s 的训练（见
    `results/detector_comparison.md` 表 4）。加载失败**不静默回退**成现训——
    否则"用的是哪份权重"就说不清了，而本脚本的全部数字都挂在这个权重上。
    """
    from agentenvs.detectors import make_detector

    spec = DETECTORS[name]
    art = spec.get("artifact")
    if not art:
        return None, None

    candidates = [
        os.path.join(ARTIFACTS, art),
        os.path.join(REPO_ROOT, "src", "ae_repro", "artifacts", art),
    ]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        return None, None

    kwargs = dict(spec["kwargs"])
    kwargs["embed_dim"] = embed_dim
    det = make_detector(spec["key"], **kwargs)
    det.load(path)
    return det, {"source": "artifact", "path": path, "fit_seconds": 0.0}


def evaluate_one(name: str, ds, embed_dim: int, seed: int, policies: List[str],
                 smoke: bool, agent_id: int) -> Dict[str, Any]:
    from agentenvs import CyberDefenseEnv

    spec = DETECTORS[name]
    t_fit = 0.0
    det, loaded = build_detector(name, embed_dim, seed)

    kw_smoke: Dict[str, Any] = {}
    if smoke and spec["key"] in ("autoencoder", "ae_paper"):
        kw_smoke = {"epochs": 2, "hidden_dims": (16, 8), "patience": 2}

    if det is None:
        from agentenvs.detectors import make_detector

        kwargs = dict(spec["kwargs"])
        kwargs["embed_dim"] = embed_dim
        kwargs.update(kw_smoke)
        if spec["key"] != "null":
            kwargs.setdefault("seed", seed)
        det = make_detector(spec["key"], **kwargs)
        t0 = time.time()
        det.fit(ds.X_normal_train)          # ★ 主口径：只喂良性，不传 y_test
        t_fit = time.time() - t0
        source = {"source": "fresh_fit"}
    else:
        source = loaded

    env = CyberDefenseEnv(
        X_data=ds.X_train, y_labels=ds.y_train, detector=det,
        spec=ds.spec, X_normal_for_detector=ds.X_normal_train,
        auto_fit_detector=False,
        detector_embed_dim=embed_dim, seed=seed,
    )
    mt = env_summary(env)

    # ---- 训练集 pass：检测器经验工作点 ----
    # ★ prior 直接由**全量** y_train 统计（只是计数，不需要跑检测器）；
    # ★ 工作点用**分层且保序**的子集：直接取前 N 行会拿到单一天的流量
    #   （第一次冒烟就踩到了：取 X_test[:3000] 全是 Thursday，IF 检出率 0、
    #   FPR 5.4%，看起来像"检测器坏了"，其实是子集不代表全体）。
    X_cal, y_cal = sample_ordered(ds.X_train, ds.y_train,
                                  calib_n(smoke), seed=seed)
    tr = collect_train_pass(env, X_cal, y_cal)

    out: Dict[str, Any] = {
        "name": name,
        "registry_key": spec["key"],
        "embed_dim": int(getattr(det, "embed_dim", 0)),
        "signal_dim": int(getattr(det, "signal_dim", 0)),
        "threshold": float(getattr(det, "threshold", 0.0) or 0.0),
        "threshold_mode": getattr(det, "threshold_mode", None),
        "detector_source": source,
        "fit_seconds": round(t_fit, 2),
        "env": {
            "num_agents": mt["num_agents"],
            "obs_dim": mt["obs_dim"],
            "action_dim": mt["action_dim"],
            "num_classes": mt["num_classes"],
            "class_names": class_names_of(ds),
            "feature_dim": int(ds.X_train.shape[1]),
            "detector_signal_dim": mt["detector_signal_dim"],
            "agent_focus": mt["agent_focus"],
        },
        "train_detector_operating_point": tr["detector_operating_point"],
    }

    # ---- 评估 pass（协议 v1 评估集；Monday 已排除）----
    X_ev, y_ev = sample_ordered(ds.X_test, ds.y_test, 3000 if smoke else None,
                               seed=seed)
    det_ev = run_detector_pass(env, X_ev, reset=True)

    # ---- agent-0 的输入侧证据分解（用于解释策略退化）----
    agent0 = characterize_agent0(env, y_ev, det_ev, agent_id=agent_id)

    # ---- 逐 (决策层, 智能体) 评估 ----
    policies_out: Dict[str, Any] = {}
    for pname in policies:
        pol = make_policy(pname, env.num_classes, embed_dim)
        pol.fit_prior(ds.y_train)
        pol.set_focus(env.focus)
        if pname == "specialist":
            # 专精型：每个智能体都要单独评（focus 不同 → 会判成不同的攻击类）
            per_agent = []
            for aid in range(env.num_agents):
                res = evaluate_policy(env, pol, y_ev, det_ev,
                                      class_names_of(ds), agent_id=aid)
                res["agent_id"] = aid
                res["agent_name"] = env.agents[aid]["name"]
                res["focus_class"] = env.spec.name_of(int(env.focus[aid]))
                per_agent.append(res)
            vote = evaluate_vote(env, pol, y_ev, det_ev, class_names_of(ds))
            policies_out[pname] = {
                "policy": pname,
                "is_per_agent": True,
                "per_agent": per_agent,
                "vote": vote,
                "policy_prior": {"attack_mode": pol.attack_mode,
                                 "any_mode": pol.any_mode},
            }
            continue
        res = evaluate_policy(env, pol, y_ev, det_ev,
                              class_names_of(ds), agent_id=agent_id)
        res["policy_prior"] = {"attack_mode": pol.attack_mode,
                               "any_mode": pol.any_mode}
        policies_out[pname] = res

    # ---- 附：检测器层自身的二分类指标（与 PROTOCOL 表 1 同源，便于对账）----
    from ae_repro.ae_core import binary_metrics

    y_bin = (y_ev != 0).astype(int)
    thr = float(getattr(det, "threshold", 0.0) or 0.0)
    out["detector_binary"] = binary_metrics(y_bin, det_ev["det_score"], thr)

    # ---- 附：MADDPG 真实策略（若 results/<detector>/ 存在则报告，否则标注缺失）----
    out["maddpg_policy_artifact"] = probe_maddpg_artifact(name)

    out["policies"] = policies_out
    out["agent0_characterization"] = agent0
    return out


def class_names_of(ds) -> List[str]:
    return [str(c) for c in ds.class_names]


def evaluate_vote(env, policy: Policy, y: np.ndarray,
                  det: Dict[str, np.ndarray], class_names: List[str]) -> Dict[str, Any]:
    """5 个专精智能体做多数投票（复用平台的平票规则：平票取非 BENIGN）。

    这条**直接对应平台的 `utils.metrics._majority_vote`**，
    所以它的数字可以和 `results/<detector>/eval_*.json` 里的 `vote` 对账。
    """
    y = np.asarray(y).astype(int)
    K = env.num_classes
    n = len(y)
    flag = det["det_flag"]

    per_agent = np.zeros((env.num_agents, n), dtype=np.int64)
    for aid in range(env.num_agents):
        per_agent[aid] = policy.decide(flag, agent_idx=aid)

    vote = np.zeros(n, dtype=np.int64)
    for t in range(n):
        counts = np.bincount(per_agent[:, t], minlength=K)
        top = int(np.argmax(counts))
        if counts[top] == 1 or (counts == counts.max()).sum() > 1:
            tied = np.where(counts == counts.max())[0]
            non_benign = tied[tied != 0]
            top = int(non_benign[0]) if len(non_benign) else 0
        vote[t] = top

    attack_mask = (y != 0)
    n_attack = int(attack_mask.sum())
    alerted = flag > 0
    success = int(((vote == y) & alerted & attack_mask).sum())
    pred_sum = np.bincount(vote, minlength=K)
    acc = float((vote == y).mean())
    pe = float((np.bincount(y, minlength=K) * pred_sum).sum() / max(n, 1) ** 2)
    kappa = float((acc - pe) / (1 - pe)) if abs(1 - pe) > 1e-12 else 0.0
    per_class_f1 = []
    for i in range(1, K):
        tp = int(((vote == i) & (y == i)).sum())
        fp = int(((vote == i) & (y != i)).sum())
        fn = int(((vote != i) & (y == i)).sum())
        p = tp / (tp + fp + EPS)
        r = tp / (tp + fn + EPS)
        per_class_f1.append(2 * p * r / (p + r + EPS))
    return {
        "success_rate_on_attacks": success / max(n_attack, 1),
        "n_success": success,
        "n_attack": n_attack,
        "accuracy": acc,
        "macro_f1": float(np.mean(per_class_f1)) if per_class_f1 else 0.0,
        "kappa": kappa,
        "majority_share": float(pred_sum.max() / max(n, 1)),
        "accuracy_when_alerted": (
            float((vote[alerted] == y[alerted]).mean()) if alerted.any() else 0.0
        ),
        "flag_rate": float(alerted.mean()),
        "decomposition": {
            "success_alerted_and_correct": success,
            "class_error_alerted": int(((vote != y) & alerted & attack_mask).sum()),
            "detector_missed_class_correct": int(
                ((vote == y) & ~alerted & attack_mask).sum()),
            "double_failure": int(((vote != y) & ~alerted & attack_mask).sum()),
            "n_attack": n_attack,
        },
        "per_class": {
            class_names[i]: {
                "n": int((y == i).sum()),
                "success_rate": (
                    float(((vote == i) & (y == i) & alerted)[y == i].mean())
                    if (y == i).any() else 0.0
                ),
                "alert_rate": (
                    float(alerted[y == i].mean()) if (y == i).any() else 0.0
                ),
            }
            for i in range(K)
        },
        "per_agent_pred_distribution": {
            env.agents[aid]["name"]: {
                class_names[j]: float((per_agent[aid] == j).mean()) for j in range(K)
            }
            for aid in range(env.num_agents)
        },
    }


def characterize_agent0(env, y_ev: np.ndarray, det_ev: Dict[str, np.ndarray],
                        agent_id: int = 0) -> Dict[str, Any]:
    """解释决策层为什么会退化：把「检测器给了什么证据」按真实类别拆开。

    智能体要判对，至少要满足两条：检测器在该类上会告警；该类的特征/分数
    与其它类可分。这两条都能从 det_ev 直接算出来。
    """
    y = np.asarray(y_ev).astype(int)
    K = env.num_classes
    flag = det_ev["det_flag"].astype(bool)
    score = det_ev["det_score"]
    focus = int(env.focus[agent_id])

    by_class: Dict[str, Any] = {}
    for j in range(K):
        m = (y == j)
        if not m.any():
            by_class[env.spec.name_of(j)] = {"n": 0}
            continue
        by_class[env.spec.name_of(j)] = {
            "n": int(m.sum()),
            "detector_alert_rate": float(flag[m].mean()),
            "score_median": float(np.median(score[m])),
            "score_p95": float(np.percentile(score[m], 95)),
        }

    # 检测器把所有类都推到同一个方向的程度：类间分数中位数极差 / 中位数
    meds = [v["score_median"] for v in by_class.values() if v.get("n")]
    spread = (max(meds) - min(meds)) / (abs(np.median(meds)) + EPS) if len(meds) > 1 else 0.0

    return {
        "agent_id": agent_id,
        "agent_name": env.agents[agent_id]["name"],
        "focus_class": env.spec.name_of(focus) if focus < K else "?",
        "note": ("本仓库没有该机制的 MADDPG 权重，故只能报告**输入侧证据**；"
                 "MADDPG 真实预测见 maddpg_policy_artifact"),
        "detector_flag_rate": float(flag.mean()),
        "by_true_class": by_class,
        "score_median_spread_across_classes": float(spread),
    }


def probe_maddpg_artifact(name: str) -> Dict[str, Any]:
    """找策略训练产物并读出来；没有就明确写 missing。

    ★ `results/_legacy_platform/` 是从原 `safenetwork/.../outputs/` 逐字节拷贝进来的
    **旧口径**产物（29000 行评估集 + F1 扫描阈值）。它的数字**不能**和本脚本
    协议 v1 的表放在同一列比较，所以这里额外打上 `protocol: legacy` 标记。
    """
    import glob

    cands: List[str] = []
    for pat in (os.path.join(RESULTS, name, "eval_*.json"),
                os.path.join(RESULTS, name + "_only", "eval_*.json"),
                os.path.join(RESULTS, "_legacy_platform", name, "eval_*.json")):
        cands.extend(glob.glob(pat))
    if not cands:
        return {"available": False,
                "reason": "该机制没有 MADDPG 策略产物（见 PROTOCOL.md §6 第 3 条）"}

    # 优先非 legacy 的产物；同为 legacy 时取 eval_<name>.json 而非 eval_best.json
    def rank(p: str) -> Tuple[int, int, str]:
        legacy = 1 if "_legacy_platform" in p else 0
        is_best = 1 if "eval_best" in os.path.basename(p) else 0
        return (legacy, is_best, p)

    path = sorted(cands, key=rank)[0]
    try:
        with open(path, "r", encoding="utf-8") as fh:
            blob = json.load(fh)
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "reason": f"读取失败 {path}: {exc}"}
    vote = blob.get("vote") or {}
    is_legacy = "_legacy_platform" in path
    return {
        "available": True,
        "path": path,
        "protocol": "legacy（旧口径：29000 行 + F1 扫描，不可与协议 v1 混引）"
                    if is_legacy else "v1",
        "n_samples": blob.get("n_samples"),
        "vote": {k: vote.get(k) for k in
                 ("accuracy", "macro_f1", "kappa", "detection_rate",
                  "false_alarm_rate", "mean_reward")},
        "agents": [
            {"name": a.get("name"), "focus": a.get("focus"),
             "accuracy": a.get("accuracy"), "macro_f1": a.get("macro_f1"),
             "kappa": a.get("kappa"), "detection_rate": a.get("detection_rate"),
             "false_alarm_rate": a.get("false_alarm_rate"),
             "mean_reward": a.get("mean_reward")}
            for a in (blob.get("agents") or [])
        ],
    }


# --------------------------------------------------------------------------- 输出
def fmt(v, nd: int = 4) -> str:
    if v is None:
        return "n/a"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if f != f:  # NaN
        return "n/a"
    return f"{f:.{nd}f}"


def ok_results(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """过滤掉初始化失败的机制（它们没有 `policies` 字段，直接进表格会 KeyError）。"""
    return [r for r in payload["results"] if "policies" in r]


def _flat_policy_rows(r: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    """把 `policies` 展平成 (标签, 指标字典) 列表；`specialist` 会展开成逐智能体 + 投票。"""
    out: List[Tuple[str, Dict[str, Any]]] = []
    for pname, p in r["policies"].items():
        if p.get("is_per_agent"):
            for a in p["per_agent"]:
                out.append((f"{pname}/{a.get('agent_name', '?')}", a))
            out.append((f"{pname}/投票", p["vote"]))
        else:
            out.append((pname, p))
    return out


def to_markdown(payload: Dict[str, Any]) -> str:
    L: List[str] = []
    A = L.append
    rows = ok_results(payload)
    A("# 策略成功率扫描（协议 v1）")
    A("")
    A(f"> 生成时间：{payload['generated_at']}　|　数据包：`{payload['data']['path']}`")
    A(f"> 评估集：{payload['data']['X_test']}（协议 v1，已排除 Monday）")
    A(f"> 环境：{payload['env']['python']}，numpy {payload['env']['numpy']}，"
      f"torch {payload['env']['torch']}")
    A("")
    A("## 口径声明（先读这一段再引用数字）")
    A("")
    A("* 三条判定规则都是**无训练规则**，因此本表**不含任何策略学习**。"
      "它回答的是「这套检测器 + 这种分工方式能到多少」，**不是 MADDPG 的成绩**。")
    A("* `成功率 = 检测器告警 且 类别判定正确`，只在**真实攻击样本**上统计。")
    A("* `benign_only`（恒判 BENIGN）的成功率**按定义恒为 0** —— 它是干净的退化下界，"
      "任何 > 0 的成功率都来自检测器 + 专精分工，不是运气。")
    A("* `specialist` 每个智能体只判自己的 `focus` 类，因此单体的 macro-F1 会很低"
      "（它不做 6 分类）；看它要看**逐类成功率**和**投票**两栏。")
    A("* 旧 docx 表 5.1 是 29000 行 + F1 扫描口径，**与本表不可混引**。")
    A("")
    A("## 表 1　各机制 / 各判定规则总览")
    A("")
    A("| 机制 | 判定规则 | 成功率(攻击) | 告警时判对 | 告警率 | macro-F1 | Kappa | 多数类占比 | 塌缩? |")
    A("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        for label, p in _flat_policy_rows(r):
            collapsed = "⚠️ 是" if p["majority_share"] > 0.8 else "否"
            A(f"| `{r['name']}` | `{label}` | **{fmt(p['success_rate_on_attacks'])}** | "
              f"{fmt(p['accuracy_when_alerted'])} | {fmt(p['flag_rate'])} | "
              f"{fmt(p['macro_f1'])} | {fmt(p['kappa'])} | {fmt(p['majority_share'])} | "
              f"{collapsed} |")
    A("")
    A("> `塌缩?` 判据：`多数类占比 > 0.8`。这一列用来一眼看出规则是否已退化成常数输出。")
    A("")
    A("## 表 2　失败归因分解（攻击样本）")
    A("")
    A("四类互斥且穷尽：`成功` + `告警但判错` + `漏报但判对` + `双重失败` = 攻击总数。")
    A("")
    A("| 机制 | 判定规则 | 攻击数 | 成功(告警+判对) | 告警但判错 | 漏报但判对 | 双重失败 |")
    A("|---|---|---|---|---|---|---|")
    for r in rows:
        for label, p in _flat_policy_rows(r):
            d = p["decomposition"]
            A(f"| `{r['name']}` | `{label}` | {d['n_attack']} | "
              f"{d['success_alerted_and_correct']} | {d['class_error_alerted']} | "
              f"{d['detector_missed_class_correct']} | {d['double_failure']} |")
    A("")
    A("## 表 3　逐攻击类别的反应（★ 直接回答「每一次攻击是什么反应」）")
    A("")
    if rows:
        classes = rows[0]["env"]["class_names"]
        attacks = [c for c in classes if c != "BENIGN"]
        A("**这一栏才是「每条攻击流量被怎么处理」的直接答案**：")
        A("")
        A("* `机制层成功率` = 该类样本里**检测器告警**的比例。它不含任何类别判定，"
          "是机制层能力的**干净度量**，也是现有产物能负责任地回答的部分；")
        A("* `端到端成功率` = 告警 **且** 类别判对。⚠️ **现有产物答不了这一栏**："
          "本仓库的检测器只输出二值告警，**不输出攻击类别**，"
          "MADDPG 的 6 分类头又只有 IF 一版且已退化（见 `results/README.md`）。"
          "表里这行用的是最朴素的兜底规则（告警 → 先验攻击类），"
          "它系统性偏低，**只可作为下界**，不是机制层的真实上限；")
        A("* 两栏的差 = **一个正常工作的分类头能补回多少**，"
          "这正是 `PROTOCOL.md` §6 第 3 条（补跑策略训练）要解决的问题。")
        A("")
        A("| 机制 | 指标 | " + " | ".join(attacks) + " |")
        A("|---" * (len(attacks) + 2) + "|")
        for r in rows:
            p = r["policies"].get("detector")
            if not p:
                continue
            mech, e2e = [], []
            for c in attacks:
                e = p["per_class"].get(c) or {}
                if not e.get("n"):
                    mech.append("—")
                    e2e.append("—")
                else:
                    mech.append(f"**{fmt(e.get('alert_rate'))}**")
                    e2e.append(fmt(e.get("success_rate")))
            A(f"| `{r['name']}` | **机制层成功率**（告警率） | " + " | ".join(mech) + " |")
            A(f"| `{r['name']}` | 端到端成功率（下界） | " + " | ".join(e2e) + " |")
    A("")
    A("## 表 4　专精分工下能捡回多少（判定规则 = `specialist` 投票）")
    A("")
    A("5 个智能体各自按 `focus` 判类后多数投票。它比 `detector` 好多少，"
      "就是「分工」这个设计值多少；两类都为 0 的攻击族说明**不是分工不够，是检测器没信号**。")
    A("")
    if rows:
        classes = rows[0]["env"]["class_names"]
        attacks = [c for c in classes if c != "BENIGN"]
        A("| 机制 | " + " | ".join(attacks) + " |")
        A("|---" * (len(attacks) + 1) + "|")
        for r in rows:
            p = (r["policies"].get("specialist") or {}).get("vote")
            if not p:
                continue
            cells = []
            for c in attacks:
                e = p["per_class"].get(c) or {}
                cells.append(fmt(e.get("success_rate")) if e.get("n") else "—")
            A(f"| `{r['name']}` | " + " | ".join(cells) + " |")
    A("")
    A("## 表 5　机制层自身的经验工作点（评估集，与 `detector_comparison.md` 同源）")
    A("")
    A("| 机制 | 阈值 | P | R | F1 | AUC | FPR | FP‰ | P@1% |")
    A("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        m = r["detector_binary"]
        A(f"| `{r['name']}` | {fmt(m['threshold'], 5)} | {fmt(m['precision'])} | "
          f"{fmt(m['recall'])} | **{fmt(m['f1'])}** | {fmt(m.get('roc_auc'))} | "
          f"{fmt(m['false_positive_rate'])} | {fmt(m.get('fp_per_1000_flows'), 1)} | "
          f"{fmt(m.get('precision_at_1pct'))} |")
    A("")
    A("## 表 6　已有 MADDPG 策略产物（不是本脚本生成的，仅作对照）")
    A("")
    A("⚠️ 这些产物来自合并前的 `safenetwork/.../outputs/`（已逐字节拷进 "
      "`results/_legacy_platform/`），用的是**旧口径**：29000 行评估集（含 14.02% "
      "自评行）+ F1 扫描阈值。**只能和旧 docx 表 5.1 比，不能和本文件其他表比。**")
    A("")
    A("| 机制 | 产物 | 口径 | 样本数 | 投票准确率 | macro-F1 | Kappa | 检测率 | 误报率 | 平均回报 |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        a = r.get("maddpg_policy_artifact") or {}
        if not a.get("available"):
            A(f"| `{r['name']}` | ❌ 无 | — | — | — | — | — | — | — | — |")
            continue
        v = a["vote"]
        A(f"| `{r['name']}` | `{os.path.basename(a['path'])}` | "
          f"{'legacy' if str(a.get('protocol', '')).startswith('legacy') else 'v1'} | "
          f"{a.get('n_samples')} | {fmt(v['accuracy'])} | "
          f"{fmt(v['macro_f1'])} | {fmt(v['kappa'])} | {fmt(v['detection_rate'])} | "
          f"{fmt(v['false_alarm_rate'])} | {fmt(v['mean_reward'])} |")
    A("")
    A("## 表 7　已有 MADDPG 的逐智能体分解（诊断塌缩）")
    A("")
    A("| 机制 | 智能体 | 专注 | 准确率 | macro-F1 | Kappa | 检测率 | 误报率 | 平均回报 |")
    A("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        a = r.get("maddpg_policy_artifact") or {}
        if not a.get("available"):
            continue
        for ag in a["agents"]:
            A(f"| `{r['name']}` | {ag['name']} | {ag['focus']} | {fmt(ag['accuracy'])} | "
              f"{fmt(ag['macro_f1'])} | {fmt(ag['kappa'])} | {fmt(ag['detection_rate'])} | "
              f"{fmt(ag['false_alarm_rate'])} | {fmt(ag['mean_reward'])} |")
    A("")
    A("> `检测率 1.0 / 误报率 1.0` 说明该智能体**对所有输入都判同一个攻击类**；"
      "`Kappa ≈ 0` 说明它的判断与真值无关。这两条同时出现 = 该智能体已完全塌缩。")
    A("")
    A("## 表 8　检测器给了多少证据（按真实类别）")
    A("")
    A("这张表回答「为什么某类成功率是 0」：如果 `detector_alert_rate` 就是 0，"
      "那是**机制层的问题**，换判定规则也没用。")
    A("")
    for r in rows:
        ch = r.get("agent0_characterization") or {}
        bc = ch.get("by_true_class") or {}
        A(f"### `{r['name']}`（阈值 {fmt(r['threshold'], 5)}，"
          f"全类告警率 {fmt(ch.get('detector_flag_rate'))}）")
        A("")
        A("| 真实类别 | 样本数 | 检测器告警率 | 分数中位数 | 分数 P95 |")
        A("|---|---|---|---|---|")
        for cname, v in bc.items():
            if not v.get("n"):
                continue
            A(f"| {cname} | {v['n']} | {fmt(v['detector_alert_rate'])} | "
              f"{fmt(v['score_median'])} | {fmt(v['score_p95'])} |")
        A("")
    if any("error" in r for r in payload["results"]):
        A("## 初始化失败的机制（这是错误，不是结果）")
        A("")
        for r in payload["results"]:
            if "error" in r:
                A(f"* `{r['name']}`：{r['error']}")
        A("")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="逐攻击样本的策略成功率扫描（协议 v1）")
    ap.add_argument("--data", default=BUNDLE)
    ap.add_argument("--detectors", default=",".join(DEFAULT_ORDER))
    ap.add_argument("--policies", default=",".join(ALL_POLICIES))
    ap.add_argument("--embed-dim", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--agent-id", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out-dir", default=RESULTS)
    args = ap.parse_args(argv)

    # ★ 冒烟结果不能覆盖正式产物：默认输出目录在 --smoke 时自动改到 results/_smoke/。
    #   （踩过：冒烟跑完把 3000 行的数字写进了 results/policy_sweep.md，
    #   文件名没有任何"这是冒烟"的标记，容易被当成正式结果引用。）
    out_dir = args.out_dir
    if args.smoke and out_dir == RESULTS:
        out_dir = os.path.join(RESULTS, "_smoke")

    from agentenvs import load_bundle

    ds = load_bundle(args.data)
    policies = [s.strip() for s in args.policies.split(",") if s.strip()]

    print("=" * 78)
    print("  策略成功率扫描（PROTOCOL.md v1）")
    print("=" * 78)
    print(f"数据包   : {args.data}")
    print(f"训练/测试: {ds.X_train.shape} / {ds.X_test.shape}")
    print(f"类别空间 : {class_names_of(ds)}")
    print(f"决策层   : {policies}")

    results = []
    for name in [s.strip() for s in args.detectors.split(",") if s.strip()]:
        if name not in DETECTORS:
            print(f"[跳过] 未知机制 {name!r}")
            continue
        print("\n" + "-" * 78)
        t0 = time.time()
        try:
            r = evaluate_one(name, ds, args.embed_dim, args.seed, policies,
                             args.smoke, args.agent_id)
        except Exception as exc:  # noqa: BLE001
            print(f"  {name:16s} [失败] {type(exc).__name__}: {exc}")
            results.append({"name": name, "error": f"{type(exc).__name__}: {exc}"})
            continue
        results.append(r)
        for label, p in _flat_policy_rows(r):
            print(f"  {name:16s} {label:26s} 成功率={p['success_rate_on_attacks']:.4f}  "
                  f"告警时判对={p['accuracy_when_alerted']:.4f}  "
                  f"macro-F1={p['macro_f1']:.4f}  多数类={p['majority_share']:.3f}")
        print(f"  {'':16s} 用时 {time.time() - t0:.1f}s")

    import sklearn
    try:
        import torch
        tv = torch.__version__
    except Exception:  # noqa: BLE001
        tv = "未安装"

    payload = {
        "protocol": "v1",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "protocol_doc": "PROTOCOL.md",
        "env": {"python": _platform.python_version(), "numpy": np.__version__,
                "sklearn": sklearn.__version__, "torch": tv,
                "platform": _platform.platform()},
        "data": {"path": os.path.abspath(args.data),
                 "X_normal_train": list(ds.X_normal_train.shape),
                 "X_train": list(ds.X_train.shape),
                 "X_test": list(ds.X_test.shape),
                 "class_names": class_names_of(ds),
                 "embed_dim": args.embed_dim, "seed": args.seed,
                 "smoke": bool(args.smoke)},
        "policies": policies,
        "policy_definitions": {
            "detector": "检测器告警 → 先验攻击类；不告警 → BENIGN（完全信任检测器）",
            "specialist": "检测器告警 → 该智能体自己的 focus 类；另有 5 体多数投票",
            "benign_only": "恒判 BENIGN（按定义成功率恒为 0 的退化下界）",
        },
        "results": results,
    }

    os.makedirs(out_dir, exist_ok=True)
    jp = os.path.join(out_dir, "policy_sweep.json")
    mp = os.path.join(out_dir, "policy_sweep.md")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    with open(mp, "w", encoding="utf-8") as f:
        f.write(to_markdown(payload))
    print("\n" + "=" * 78)
    print(f"[OK] 结果已写入 {jp}")
    print(f"[OK] 报告已写入 {mp}")
    return 0


if __name__ == "__main__":
    try:
        import sys
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    raise SystemExit(main())
