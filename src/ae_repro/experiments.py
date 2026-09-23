# -*- coding: utf-8 -*-
"""
experiments.py —— 复现实验入口：在 chethuhn CICIDS2017 上跑论文自编码器并产出报告。

主口径（跟随平台，便于和 IF / Kalman 横向对比）
----------------------------------------------
    train  = Monday 良性流量（60000 行，min–max 统计量也只来自它）
    test   = 周二~周五带标签流量的分层测试集
    检测器**只在良性流量上训练**，阈值只用良性分数标定（不看测试标签）

同时报告两种阈值口径 + 一个乐观上界：
    * `evt`        —— 论文 B 的 EVT/GPD 尾部标定（目标误报率 α）
    * `percentile` —— 平台默认口径：良性分数 1-α 分位
    * `best_f1`    —— 测试集上扫出的最优 F1 阈值（**乐观上界，不是上线口径**）

用法
----
    python -m ae_repro.experiments --data _cache/cicids_sample.npz --out results
    python -m ae_repro.experiments --quick            # 只跑主配置，快速冒烟
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .ae_core import AEConfig, AutoencoderCore, binary_metrics, best_f1_threshold, fit_evt_threshold
from . import data_prep, paths

# ------------------------------------------------------------------ 论文参考指标
PAPER_REFERENCE = {
    "AUTO.pdf (主复现，第 1 阶段 Autoencoder 单独结果)": {
        "source": "El Emary et al., Int. J. Res. Metaverse, vol.3 no.1, pp.46-63, 2026, Table 1",
        "ROC_AUC": 0.949,
        "precision": 0.82,
        "recall": 0.79,
        "F1": 0.80,
        "threshold": 0.52,
        "dataset": "区块链元宇宙交易数据（**非 CICIDS2017**，特征与量纲都不同）",
        "note": "论文摘要写 0.952 / hybrid 表 1 又写 0.92，正文自相矛盾；"
                "阈值 0.52 是作用在 min–max 归一化后的分数上，与本文口径不可直接比较。",
    },
    "Autoencoder-Based_Anomaly_Detection.pdf (仅借鉴 EVT 阈值与归因)": {
        "source": "Lozano-Paredes et al., IEEE Access, vol.14, 2026",
        "metric": "按攻击类型的 F1（MS 0.78~0.88 / DM ≤0.79 / DoS 召回 1.00）",
        "dataset": "IEC 61850 GOOSE 变电站流量（**非 CICIDS2017**）",
        "note": "SEQ/TEMP 双视图特征依赖 GOOSE 的 stNum/sqNum/重传时序，"
                "CICIDS2017 的流特征里没有对应物，因此只借鉴其 EVT 阈值与逐特征归因。",
    },
}


# ------------------------------------------------------------------ 实验配置
@dataclass
class ExpConfig:
    name: str
    label: str
    core: AEConfig
    note: str = ""


def make_suite(quick: bool = False) -> List[ExpConfig]:
    base = dict(
        hidden_dims=(32, 16, 8), latent_dim=4, decoder="asymmetric", batchnorm=True,
        activation="leaky_relu", output_activation="none", dropout=0.1,
        feature_transform="none", per_feature_norm=True,
        epochs=250, batch_size=128, lr=1e-3, weight_decay=0.0,
        lr_decay=0.9, lr_decay_every=20, val_fraction=0.15, patience=80,
        embed_dim=4, threshold_mode="percentile", alpha=0.05, seed=42,
    )
    main = ExpConfig(
        name="ae_paper_main",
        label="复现主配置",
        core=AEConfig(**base),
        note="编码器 78→32→16→8→4 + 浅层解码器，BN + LeakyReLU(0.1) + dropout 0.1；"
             "直接吃平台 min–max 后的 [0,1] 特征；打分用逐特征标准化残差；"
             "Adam 1e-3 + 每 20 轮 ×0.9，早停 patience=80；阈值取良性 95% 分位。",
    )
    if quick:
        return [main]

    return [
        main,
        ExpConfig(
            name="ablation_paper_literal",
            label="消融：论文字面实现",
            core=AEConfig(**{**base, "feature_transform": "none", "per_feature_norm": False}),
            note="直接吃 [0,1] 特征、打分用原始 mean_j (x_j-x̂_j)²，即 AUTO.pdf 的字面描述；"
                 "作为「照抄论文」的基线。",
        ),
        ExpConfig(
            name="ablation_zscore_input",
            label="消融：额外做 z-score 输入（B 论文做法）",
            core=AEConfig(**{**base, "feature_transform": "zscore"}),
            note="平台已经 min–max 过，再标准化一次会把攻击的幅度信息压掉。",
        ),
        ExpConfig(
            name="ablation_log1p_input",
            label="消融：log1p + z-score 输入",
            core=AEConfig(**{**base, "feature_transform": "log1p_zscore"}),
            note="额外压缩重尾后再标准化。",
        ),
        ExpConfig(
            name="ablation_no_dropout",
            label="消融：去掉 dropout",
            core=AEConfig(**{**base, "dropout": 0.0}),
            note="量化 dropout 对 PR-AUC 的贡献。",
        ),
        ExpConfig(
            name="ablation_symmetric",
            label="消融：对称解码器",
            core=AEConfig(**{**base, "decoder": "symmetric"}),
            note="解码器改成编码器镜像（AUTO 论文的字面描述）。",
        ),
        ExpConfig(
            name="ablation_no_bn",
            label="消融：去掉 BatchNorm",
            core=AEConfig(**{**base, "batchnorm": False}),
            note="检验 BN 对小瓶颈 AE 的贡献。",
        ),
        ExpConfig(
            name="ablation_sigmoid_out",
            label="消融：输出层加 sigmoid（B 论文写法）",
            core=AEConfig(**{**base, "output_activation": "sigmoid"}),
            note="输出层 sigmoid 会把重构值压进 (0,1)，实测掉点。",
        ),
        ExpConfig(
            name="ablation_relu_30ep",
            label="消融：ReLU + 30 轮（贴近平台内置 AE）",
            core=AEConfig(**{**base, "activation": "relu", "epochs": 30, "patience": 30,
                             "lr_decay": 1.0, "output_activation": "none"}),
            note="更接近平台内置 AutoencoderDetector 的训练风格，作为交叉对照。",
        ),
    ]


# ------------------------------------------------------------------ 漂移诊断
def per_day_calibration(
    y_bin: np.ndarray,
    scores: np.ndarray,
    day: Optional[np.ndarray],
    global_thr: float,
    alpha: float = 0.05,
) -> Optional[Dict[str, Any]]:
    """补充口径：**每天只用当天的良性样本**重新标定阈值。

    这不是上线方案（真实场景里你拿不到"当天的良性样本"这个前提），
    而是用来回答"跨天掉点到底是模型不行还是阈值被漂移带偏了"：
    如果换成本天良性标定后 F1 大幅回升，说明模型排序没问题、问题在阈值。

    只使用良性样本，不使用任何标签，因此不存在标签泄漏。
    """
    if day is None:
        return None
    per_day: Dict[str, Any] = {}
    all_pred = np.zeros_like(y_bin)
    for d in sorted({str(x) for x in day}):
        m = np.array([str(x) == d for x in day])
        benign = m & (y_bin == 0)
        if benign.sum() < 50:
            continue
        thr_d = float(np.quantile(scores[benign], 1.0 - alpha))
        pred_d = scores[m] >= thr_d
        all_pred[m] = pred_d
        per_day[d] = {
            "n_benign_calibration": int(benign.sum()),
            "threshold": thr_d,
            "vs_global_threshold": thr_d - float(global_thr),
            "fpr_on_this_day_benign": float(pred_d[y_bin[m] == 0].mean()),
            "detection_rate_on_this_day_attacks": (
                float(pred_d[y_bin[m] == 1].mean()) if (y_bin[m] == 1).any() else None
            ),
        }
    out = binary_metrics(y_bin, scores, float(global_thr))
    rec = binary_metrics(y_bin, np.where(all_pred, 1.0, 0.0), 0.5)
    return {
        "per_day": per_day,
        "pooled_metrics_with_per_day_thresholds": rec,
        "note": "每天用当天良性样本重标定阈值（仅用良性、不用标签）；"
                "用于区分「模型排序不行」与「全局阈值被跨天漂移带偏」。",
    }


# ------------------------------------------------------------------ 单次实验
def run_one(exp: ExpConfig, data: Dict[str, Any], want_latency: bool = True) -> Dict[str, Any]:
    Xn = np.asarray(data["X_normal_train"], dtype=np.float32)
    Xte = np.asarray(data["X_test"], dtype=np.float32)
    yte = np.asarray(data["y_test"]).astype(int)
    class_names = [str(c) for c in data["class_names"]]
    feat_names = [str(f) for f in data["feature_names"]]
    day_te = np.asarray(data["day_test"]) if "day_test" in data else None

    y_bin = (yte != 0).astype(int)

    t0 = time.perf_counter()
    core = AutoencoderCore(exp.core)
    core.fit(Xn, X_test=Xte, y_test=y_bin)
    fit_seconds = time.perf_counter() - t0

    scores = core.score(Xte)

    # ---------------- 三种阈值 ----------------
    s_benign = core.score(Xn)
    evt = fit_evt_threshold(s_benign, alpha=exp.core.alpha,
                            tail_fraction=exp.core.evt_tail_fraction)
    thr_pct = float(np.quantile(s_benign, 1.0 - exp.core.alpha))
    thr_f1, f1_oracle = best_f1_threshold(y_bin, scores)

    rules = {
        "evt": (float(evt.threshold), "EVT/GPD 尾部标定（论文 B 做法，只用良性分数）"),
        "percentile": (thr_pct, f"良性分数 {1 - exp.core.alpha:.0%} 分位（平台默认口径）"),
        "best_f1": (float(thr_f1), "测试集上扫出的最优 F1 阈值（乐观上界，不可上线用）"),
    }
    metrics_at: Dict[str, Dict[str, float]] = {}
    for k, (thr, _desc) in rules.items():
        metrics_at[k] = binary_metrics(y_bin, scores, thr)
        metrics_at[k]["description"] = _desc

    # 锚定口径：平台的 AutoencoderDetector 默认就是「分位数阈值 + 分位数 trust」，
    # 所以主口径取 percentile，保证把本文件贴进平台后行为与报告一致。
    MAIN_RULE = "percentile"
    main = metrics_at[MAIN_RULE]
    thr_main = rules[MAIN_RULE][0]

    # 漂移诊断：每天用当天良性样本重标定阈值（只用良性、不用标签）
    pd_calib = per_day_calibration(y_bin, scores, day_te, thr_main, alpha=exp.core.alpha)
    if pd_calib is not None:
        metrics_at["per_day_calib"] = pd_calib["pooled_metrics_with_per_day_thresholds"]
        metrics_at["per_day_calib"]["description"] = pd_calib["note"]
        metrics_at["per_day_calib"]["threshold"] = float("nan")  # 每天不同，见 per_day

    # ---------------- 分类别 / 分天分解（都在主口径阈值下） ----------------
    pred = scores >= thr_main
    by_class: Dict[str, Dict[str, Any]] = {}
    for ci, cname in enumerate(class_names):
        m = yte == ci
        if not m.any():
            continue
        if ci == 0:
            by_class[cname] = {
                "n": int(m.sum()),
                "false_positive_rate": float(pred[m].mean()),
                "flagged": int(pred[m].sum()),
            }
        else:
            by_class[cname] = {
                "n": int(m.sum()),
                "detection_rate": float(pred[m].mean()),
                "flagged": int(pred[m].sum()),
            }
    by_day: Dict[str, Dict[str, Any]] = {}
    if day_te is not None:
        for d in sorted(set(str(x) for x in day_te)):
            m = np.array([str(x) == d for x in day_te])
            if not m.any():
                continue
            yb_d, y_bin_d, pred_d = yte[m], y_bin[m], pred[m]
            by_day[d] = {
                "n": int(m.sum()),
                "n_benign": int((y_bin_d == 0).sum()),
                "n_attack": int((y_bin_d == 1).sum()),
                "attack_ratio": float(y_bin_d.mean()),
                "detection_rate_on_attacks": float(pred_d[y_bin_d == 1].mean()) if (y_bin_d == 1).any() else None,
                "false_positive_rate_on_benign": float(pred_d[y_bin_d == 0].mean()) if (y_bin_d == 0).any() else None,
                # 分数分布用中位数/分位数而非均值：CICIDS 个别良性流的
                # Flow Bytes/s 之类会算出极端值，均值会被单条样本带跑。
                "benign_score_median": float(np.median(scores[m][y_bin_d == 0])) if (y_bin_d == 0).any() else None,
                "benign_score_p95": float(np.quantile(scores[m][y_bin_d == 0], 0.95)) if (y_bin_d == 0).any() else None,
                "benign_score_mean": float(scores[m][y_bin_d == 0].mean()) if (y_bin_d == 0).any() else None,
                "attack_score_median": float(np.median(scores[m][y_bin_d == 1])) if (y_bin_d == 1).any() else None,
                "attack_score_mean": float(scores[m][y_bin_d == 1].mean()) if (y_bin_d == 1).any() else None,
            }

    # ---------------- 归一化敏感性：按天看良性分数漂移 ----------------
    drift = {}
    if day_te is not None:
        s_mon = s_benign
        mu_ref, sd_ref = float(s_mon.mean()), float(max(s_mon.std(), 1e-8))
        for d in sorted(set(str(x) for x in day_te)):
            m = np.array([str(x) == d for x in day_te]) & (y_bin == 0)
            if not m.any():
                continue
            drift[d] = {
                "benign_score_mean": float(scores[m].mean()),
                "z_vs_monday": float((scores[m].mean() - mu_ref) / sd_ref),
                "benign_frac_above_threshold": float((scores[m] >= thr_main).mean()),
            }

    # ---------------- 归因 / 延迟 ----------------
    top = np.argsort(-core.feature_importance)[:15]
    attribution = [
        {"feature": feat_names[i] if i < len(feat_names) else f"f{i}",
         "mean_sq_residual": float(core.feature_importance[i])}
        for i in top
    ]
    embed_idx = [] if core.embed_idx is None else [int(i) for i in core.embed_idx]
    embed_names = [feat_names[i] if i < len(feat_names) else f"f{i}" for i in embed_idx]
    latency = core.latency(Xte, repeats=2) if want_latency else None

    # ---------------- 曲线数据（给画图用，降采样） ----------------
    from sklearn.metrics import roc_curve, precision_recall_curve

    curves: Dict[str, List[float]] = {}
    if len(np.unique(y_bin)) > 1:
        fpr, tpr, _ = roc_curve(y_bin, scores)
        prec, rec, _ = precision_recall_curve(y_bin, scores)
        def _thin(a, k=300):
            a = np.asarray(a)
            if a.size <= k:
                return [float(x) for x in a]
            idx = np.linspace(0, a.size - 1, k).astype(int)
            return [float(x) for x in a[idx]]
        curves = {"roc_fpr": _thin(fpr), "roc_tpr": _thin(tpr),
                  "pr_recall": _thin(rec), "pr_precision": _thin(prec)}

    return {
        "name": exp.name,
        "label": exp.label,
        "note": exp.note,
        "config": exp.core.to_dict(),
        # 基率校正口径（提案第 4 节）：把 FPR/TPR 换算到 1% / 0.1% 攻击占比
        "base_rate_metrics": {
            "at_prior_1pct": {
                "fp_per_1000_flows": metrics_at[MAIN_RULE]["fp_per_1000_flows"],
                "precision": metrics_at[MAIN_RULE]["precision_at_1pct"],
            },
            "at_prior_0.1pct": {
                "fp_per_1000_flows": metrics_at[MAIN_RULE]["fp_per_1000_flows"],
                "precision": metrics_at[MAIN_RULE]["precision_at_0.1pct"],
            },
        },
        "fit": {
            "seconds": float(fit_seconds),
            "n_params": int(core.fit_meta.get("n_params", 0)),
            "epochs_run": int(core.fit_meta.get("epochs_run", 0)),
            "best_val_mse": core.fit_meta.get("best_val_mse"),
            "train_mse_last": core.history[-1]["train_mse"] if core.history else None,
            "val_mse_last": core.history[-1]["val_mse"] if core.history else None,
        },
        "thresholds": {
            "evt": {"value": float(evt.threshold), "u": evt.u, "xi": evt.xi,
                    "beta": evt.beta, "zeta_u": evt.zeta_u, "n_exceed": evt.n_exceed,
                    "alpha": exp.core.alpha, "fallback": bool(evt.fallback),
                    "expression": "ξ* = u + (β/ξ)·((α/ζ_u)^(-ξ) − 1)"},
            "percentile": {"value": thr_pct, "alpha": exp.core.alpha},
            "best_f1": {"value": float(thr_f1), "f1": float(f1_oracle)},
        },
        "benign_score_stats": {"mean": core.score_mu, "std": core.score_sigma,
                               "min": float(s_benign.min()), "max": float(s_benign.max())},
        "metrics": metrics_at,
        "main_rule": MAIN_RULE,
        "main_metrics": main,
        "by_class": by_class,
        "by_day": by_day,
        "per_day_calibration": pd_calib,
        "drift_vs_monday": drift,
        "attribution_top15": attribution,
        "embedding": {"dim": exp.core.embed_dim, "feature_idx": embed_idx,
                      "feature_names": embed_names},
        "latency": latency,
        "history": core.history,
        "curves": curves,
    }


# ------------------------------------------------------------------ 报告
def _day_counts(data: Dict[str, Any]) -> Optional[Dict[str, int]]:
    if "day_test" not in data:
        return None
    days, counts = np.unique(np.asarray(data["day_test"], dtype=str), return_counts=True)
    return {str(d): int(c) for d, c in zip(days, counts)}


def _read_eval_flag(data_path: str) -> Optional[bool]:
    """从数据包旁边的 .meta.json 里取「是否已排除检测器训练数据」这个开关。"""
    meta_path = os.path.splitext(data_path)[0] + ".meta.json"
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, "r", encoding="utf-8") as fh:
        m = json.load(fh)
    val = (m.get("config") or {}).get("eval_exclude_normal_source")
    return None if val is None else bool(val)


def _precision_at_prior(tpr: float, fpr: float, prior: float) -> float:
    """把 TPR/FPR 换算到指定攻击占比（基率）下的精确率。"""
    hit = prior * float(tpr)
    alarm = hit + (1.0 - prior) * float(fpr)
    return 1.0 if alarm <= 0 else hit / alarm


def _rows_identical_to_training(bundle_path: str) -> Optional[float]:
    """旧口径数据包里，测试集有多大比例与检测器训练集**逐位完全相同**。"""
    if not os.path.exists(bundle_path):
        return None
    blob = np.load(bundle_path, allow_pickle=True)
    Xn = np.ascontiguousarray(blob["X_normal_train"], dtype=np.float32)
    Xte = np.ascontiguousarray(blob["X_test"], dtype=np.float32)
    if Xn.ndim != 2 or Xte.ndim != 2 or Xn.shape[1] != Xte.shape[1]:
        return None
    a = Xn.view([("", Xn.dtype)] * Xn.shape[1]).ravel()
    ua = np.unique(a)
    b = Xte.view([("", Xte.dtype)] * Xte.shape[1]).ravel()
    idx = np.clip(np.searchsorted(ua, b), 0, len(ua) - 1)
    return float((ua[idx] == b).mean())


def load_old_protocol(old_report_path: str, old_bundle_path: str) -> Optional[Dict[str, Any]]:
    """读回修复前那一版报告，供"新旧口径对照"用（缺文件就返回 None）。

    旧报告里只存了阈值下的 P/R/F1/FPR；FP‰ 与 P@1% 由 TPR/FPR 直接换算
    （同一测试集、同一阈值，是精确的，不是估算）。
    """
    if not os.path.exists(old_report_path):
        return None
    with open(old_report_path, "r", encoding="utf-8") as fh:
        old = json.load(fh)
    h = dict(old.get("headline") or {})
    recall = float(h.get("recall") or 0.0)
    fpr = float(h.get("false_positive_rate") or 0.0)
    env = old.get("environment") or {}
    return {
        "source": os.path.abspath(old_report_path),
        "generated_at": old.get("generated_at"),
        "protocol": "旧口径：检测器的拟合数据（Monday 良性）仍在测试集里 → 自评",
        "n_test": env.get("n_test"),
        "n_features": env.get("n_features"),
        "torch": env.get("torch"),
        "metrics": {
            "threshold": h.get("threshold"),
            "roc_auc": h.get("roc_auc"),
            "pr_auc": h.get("pr_auc"),
            "precision": h.get("precision"),
            "recall": recall,
            "f1": h.get("f1"),
            "false_positive_rate": fpr,
            "fp_per_1000_flows": fpr * 1000.0,
            "precision_at_1pct": _precision_at_prior(recall, fpr, 0.01),
            "precision_at_0.1pct": _precision_at_prior(recall, fpr, 0.001),
        },
        "test_rows_identical_to_detector_training": _rows_identical_to_training(old_bundle_path),
        "note": "FP‰ / P@1% 由旧报告的 TPR、FPR 换算（同一测试集同一阈值，精确值）；"
                "主配置、随机种子与阈值口径都与新口径一致，唯一差别是测试集是否排除 Monday。",
    }


def _fmt(x: Any, nd: int = 4) -> str:
    """安全格式化数字（None / nan -> '-'）。"""
    if x is None:
        return "-"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if v != v:  # NaN
        return "-"
    return f"{v:.{nd}f}"


def _fmt_table(rows: Sequence[Sequence[Any]], header: Sequence[str]) -> str:
    cols = len(header)
    widths = [len(str(h)) for h in header]
    for r in rows:
        for i in range(cols):
            widths[i] = max(widths[i], len(str(r[i])))
    line = "  ".join(str(h).ljust(widths[i]) for i, h in enumerate(header))
    sep = "  ".join("-" * widths[i] for i in range(cols))
    body = ["  ".join(str(r[i]).ljust(widths[i]) for i in range(cols)) for r in rows]
    return "\n".join([line, sep] + body)


def build_report(results: List[Dict[str, Any]], meta: Dict[str, Any]) -> Dict[str, Any]:
    main = next(r for r in results if r["name"] == "ae_paper_main")
    return {
        "title": "AUTO.pdf 自编码器阶段 —— CICIDS2017(chethuhn) 复现报告",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "environment": meta,
        "protocol": {
            "train": "Monday 良性流量（min–max 统计量仅来自良性）",
            "test": "周二~周五 带标签流量的分层测试集",
            "detector_training_rule": "只在良性流量上训练；阈值只用良性分数标定",
            "eval_exclude_normal_source": bool(meta.get("eval_exclude_normal_source", True)),
            "note": "测试集经过分层抽样，攻击占比远高于线上真实占比。"
                    "注意区分两个指标：FPR = FP/(FP+TN) 只在良性样本上计算，"
                    "**与攻击占比无关**，可以直接外推（换算成 FP‰）；"
                    "受基率影响的是 precision —— 因此报告给出 `precision@1%`，"
                    "即把攻击占比校正回 1% 之后的精确率。"
                    "另外测试集已排除检测器的拟合数据（Monday 良性），否则是自评。",
        },
        "paper_reference": PAPER_REFERENCE,
        "deviation_from_paper": [
            "数据集不同：论文用区块链元宇宙交易数据，本复现只能用 chethuhn CICIDS2017。",
            "只复现第 1 阶段 AE：不实现 Isolation Forest 与 0.6/0.4 加权融合，"
            "因此对比对象是论文 Table 1 的 Autoencoder 行（0.949/0.82/0.79/0.80）。",
            "论文未给出 AE 的层数/瓶颈维度/epoch/学习率，本文取 B 论文的配置并做消融，"
            "所有默认值都在 config 里可复现。",
            "阈值口径不同：论文用固定 0.52（作用于其自身 min–max 归一化后的分数），"
            "本复现主口径用良性分位数（与平台内置 AE 一致），另报告 EVT/GPD 与最优 F1。",
            "评估协议不同：论文是 10 折交叉验证，本复现跟随平台协议做跨天测试；"
            "CICIDS2017 存在明显的跨天分布漂移，跨天指标会显著低于同分布交叉验证。",
        ],
        "results": results,
        "headline": {
            "main_config": main["label"],
            "threshold_rule": main["main_rule"],
            "threshold": main["metrics"][main["main_rule"]]["threshold"],
            "roc_auc": main["main_metrics"].get("roc_auc"),
            "pr_auc": main["main_metrics"].get("pr_auc"),
            "precision": main["main_metrics"]["precision"],
            "recall": main["main_metrics"]["recall"],
            "f1": main["main_metrics"]["f1"],
            "false_positive_rate": main["main_metrics"]["false_positive_rate"],
            "fp_per_1000_flows": main["main_metrics"]["fp_per_1000_flows"],
            "precision_at_1pct": main["main_metrics"]["precision_at_1pct"],
            "precision_at_0.1pct": main["main_metrics"]["precision_at_0.1pct"],
            "n_params": main["fit"]["n_params"],
            "train_seconds": main["fit"]["seconds"],
        },
    }


def print_summary(report: Dict[str, Any]) -> None:
    print("\n" + "=" * 100)
    print("AUTO.pdf 自编码器阶段 —— CICIDS2017(chethuhn) 复现结果")
    print("=" * 100)
    rows = []
    for r in report["results"]:
        m = r["main_metrics"]
        rows.append([
            r["label"],
            f"{m.get('roc_auc', float('nan')):.4f}",
            f"{m.get('pr_auc', float('nan')):.4f}",
            f"{m['precision']:.4f}",
            f"{m['recall']:.4f}",
            f"{m['f1']:.4f}",
            f"{m['false_positive_rate']:.4f}",
            f"{m['fp_per_1000_flows']:.1f}",
            f"{m['precision_at_1pct']:.4f}",
            f"{r['fit']['n_params']}",
            f"{r['thresholds']['percentile']['value']:.4f}",
        ])
    print(_fmt_table(rows, ["实验", "ROC-AUC", "PR-AUC", "P", "R", "F1", "FPR",
                            "FP‰", "P@1%", "参数量", "分位阈值"]))
    print("\n注：P/R/F1/FPR/FP‰ 均在主口径（良性 95% 分位阈值）下计算。"
          "\n    FPR 与 FP‰ 只在良性样本上计算，与攻击占比无关，可以直接外推；"
          "\n    受基率影响的是 precision：P@1% 是把攻击占比校正回 1% 之后的精确率"
          "（本表里的 P 是测试集内 17%~26% 攻击占比下的值，会明显高估可部署性）。")

    main = next(r for r in report["results"] if r["name"] == "ae_paper_main")
    print("\n--- 主配置在各阈值口径下的表现 ---")
    rows = []
    for k in ("percentile", "evt", "best_f1", "per_day_calib"):
        m = main["metrics"].get(k)
        if not m:
            continue
        rows.append([k, _fmt(m.get("threshold"), 4), _fmt(m["precision"]), _fmt(m["recall"]),
                     _fmt(m["f1"]), _fmt(m["false_positive_rate"])])
    print(_fmt_table(rows, ["口径", "阈值", "P", "R", "F1", "FPR"]))
    evt = main["thresholds"]["evt"]
    print(f"  EVT(ξ={evt['xi']:.3f}, β={evt['beta']:.3f}) "
          f"{'退回了经验分位数（一致性检验未通过）' if evt.get('fallback') else '通过一致性检验'}")
    pdc = main.get("per_day_calibration")
    if pdc:
        print("  每天重标定阈值诊断（每天只用当天良性样本，不使用标签）：")
        for d, v in pdc["per_day"].items():
            dr = v["detection_rate_on_this_day_attacks"]
            print(f"    {d:<10} thr={v['threshold']:.4f} (全局{v['vs_global_threshold']:+.4f})  "
                  f"误报={v['fpr_on_this_day_benign']:.4f}  "
                  f"检出={'—' if dr is None else f'{dr:.4f}'}")
        print("    注：当天误报率≈5% 是构造出来的（阈值就是当天良性分数的 95% 分位），"
              "不构成「阈值可校准」的证据；")
        print("        有信息量的是阈值本身差多少——差几倍就说明跨天漂移有多大。")

    print("\n--- 主配置：按攻击类别的检出率（主口径阈值） ---")
    rows = []
    for cls, v in main["by_class"].items():
        if "detection_rate" in v:
            rows.append([cls, v["n"], f"{v['detection_rate']:.4f}", v["flagged"]])
        else:
            rows.append([cls + "(良性)", v["n"], f"误报率 {v['false_positive_rate']:.4f}", v["flagged"]])
    print(_fmt_table(rows, ["类别", "样本数", "检出率/误报率", "告警数"]))

    print("\n--- 主配置：按天的漂移诊断（主口径阈值） ---")
    rows = []
    for d, v in main["by_day"].items():
        dr = v["detection_rate_on_attacks"]
        fpr = v["false_positive_rate_on_benign"]
        rows.append([d, v["n"], _fmt(v["attack_ratio"], 3),
                     "-" if dr is None else _fmt(dr),
                     "-" if fpr is None else _fmt(fpr),
                     _fmt(v.get("benign_score_median")), _fmt(v.get("attack_score_median"))])
    print(_fmt_table(rows, ["天", "测试样本", "攻击占比", "攻击检出率", "良性误报率",
                            "良性分数中位数", "攻击分数中位数"]))

    print("\n--- 主配置：重构误差贡献 Top10（论文第 7 步的归因） ---")
    for a in main["attribution_top15"][:10]:
        print(f"  {a['feature']:<36} {a['mean_sq_residual']:.6f}")

    if main.get("latency"):
        lat = main["latency"]
        print(f"\n--- 推理开销 ---  吞吐 {lat['throughput_samples_per_sec']:.0f} 条/秒，"
              f"单条 {lat['ms_per_sample']:.4f} ms，batch256 约 {lat['ms_per_batch256']:.2f} ms")

    print("\n--- 与论文 Table 1 的对照 ---")
    ref = PAPER_REFERENCE["AUTO.pdf (主复现，第 1 阶段 Autoencoder 单独结果)"]
    print(f"  论文 AE（区块链元宇宙数据）: ROC-AUC={ref['ROC_AUC']}  P={ref['precision']}  "
          f"R={ref['recall']}  F1={ref['F1']}")
    h = report["headline"]
    print(f"  本复现（CICIDS2017）      : ROC-AUC={h['roc_auc']:.4f}  P={h['precision']:.4f}  "
          f"R={h['recall']:.4f}  F1={h['f1']:.4f}")
    print(f"  基率校正（提案第 4 节）   : FP‰={h['fp_per_1000_flows']:.1f}  "
          f"P@1%={h['precision_at_1pct']:.4f}  P@0.1%={h['precision_at_0.1pct']:.4f}")
    print("  ⚠ 数据集不同、阈值口径不同，两个数字不是同一件事的两种测法，只能定性对照。")
    print("=" * 100 + "\n")

    pc = report.get("protocol_comparison")
    if pc:
        print("--- 新旧评估口径对照（同一模型、同一阈值口径，只差测试集是否排除检测器训练数据）---")
        rows = []
        for tag, key in [("旧口径（自评）", "old_protocol"), ("新口径（已修复）", "new_protocol")]:
            m = pc[key]["metrics"]
            rows.append([
                tag, pc[key].get("n_test"),
                _fmt(m.get("roc_auc")), _fmt(m.get("precision")), _fmt(m.get("recall")),
                _fmt(m.get("f1")), _fmt(m.get("false_positive_rate")),
                _fmt(m.get("fp_per_1000_flows"), 1), _fmt(m.get("precision_at_1pct")),
            ])
        print(_fmt_table(rows, ["口径", "n_test", "ROC-AUC", "P", "R", "F1", "FPR", "FP‰", "P@1%"]))
        ov = pc["old_protocol"].get("test_rows_identical_to_detector_training")
        if ov is not None:
            print(f"  旧口径测试集里与检测器训练集逐位重复的比例：{ov:.2%}")
        print()


# ------------------------------------------------------------------ 画图
def _setup_cjk_font() -> None:
    """图片里的中文标签需要 CJK 字体，否则 matplotlib 会画成一堆方框。"""
    import matplotlib

    matplotlib.rcParams["font.sans-serif"] = [
        "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC",
        "WenQuanYi Zen Hei", "Arial Unicode MS", "DejaVu Sans",
    ]
    matplotlib.rcParams["axes.unicode_minus"] = False


def make_plots(report: Dict[str, Any], out_dir: str) -> List[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        _setup_cjk_font()
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] 跳过画图（{exc}）")
        return []

    made: List[str] = []
    os.makedirs(out_dir, exist_ok=True)
    main = next(r for r in report["results"] if r["name"] == "ae_paper_main")

    # 1) 训练/验证损失
    hist = main["history"]
    if hist:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ep = [h["epoch"] for h in hist]
        ax.plot(ep, [h["train_mse"] for h in hist], label="train MSE")
        ax.plot(ep, [h["val_mse"] for h in hist], label="val MSE")
        ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.set_ylabel("MSE (log)")
        ax.set_title("AE 训练曲线（只在良性流量上训练）")
        ax.legend()
        ax.grid(alpha=0.3)
        p = os.path.join(out_dir, "fig1_training_curve.png")
        fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig); made.append(p)

    # 2) ROC + PR
    if main["curves"]:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
        c = main["curves"]
        axes[0].plot(c["roc_fpr"], c["roc_tpr"], lw=2,
                     label=f"AE (AUC={main['main_metrics'].get('roc_auc', float('nan')):.4f})")
        axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
        axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
        axes[0].set_title("ROC"); axes[0].legend(); axes[0].grid(alpha=0.3)
        axes[1].plot(c["pr_recall"], c["pr_precision"], lw=2,
                     label=f"AE (PR-AUC={main['main_metrics'].get('pr_auc', float('nan')):.4f})")
        prior = main["main_metrics"]["n_attack"] / max(main["main_metrics"]["n"], 1)
        axes[1].axhline(prior, color="gray", ls="--", lw=0.8, label=f"随机基线({prior:.2f})")
        axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
        axes[1].set_title("Precision–Recall"); axes[1].legend(); axes[1].grid(alpha=0.3)
        p = os.path.join(out_dir, "fig2_roc_pr.png")
        fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig); made.append(p)

    # 3) 各实验对比柱状图
    names = [r["label"] for r in report["results"]]
    f1s = [r["main_metrics"]["f1"] for r in report["results"]]
    aucs = [r["main_metrics"].get("roc_auc", 0.0) for r in report["results"]]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(7, 1.5 * len(names)), 4.4))
    ax.bar(x - 0.2, aucs, 0.4, label="ROC-AUC")
    ax.bar(x + 0.2, f1s, 0.4, label="F1（主口径 percentile 阈值）")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    ax.set_title("消融对比（CICIDS2017，跨天测试）")
    p = os.path.join(out_dir, "fig3_ablation.png")
    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig); made.append(p)

    # 4) 归因 Top15
    att = main["attribution_top15"][::-1]
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.barh([a["feature"] for a in att], [a["mean_sq_residual"] for a in att], color="slateblue")
    ax.set_xlabel("mean squared reconstruction error")
    ax.set_title("逐特征重构误差归因 Top15（论文第 7 步）")
    ax.grid(alpha=0.3, axis="x")
    p = os.path.join(out_dir, "fig4_attribution.png")
    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig); made.append(p)

    return made


# ------------------------------------------------------------------ main
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="AUTO.pdf 自编码器阶段复现实验")
    ap.add_argument("--data", default=paths.BUNDLE)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    ap.add_argument("--quick", action="store_true", help="只跑主配置")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--csv-dir", default=None, help="数据包不存在时用它现场构建")
    ap.add_argument("--old-report", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results", "repro_report.old_protocol.json"),
        help="修复前那一版 repro_report.json，用于生成「新旧口径对照」")
    ap.add_argument("--old-bundle", default=paths.OLD_BUNDLE,
                    help="修复前那一版数据包，用于统计测试集与训练集的重复比例")
    args = ap.parse_args(argv)

    if not os.path.exists(args.data):
        print(f"[info] 找不到数据包 {args.data}，现场构建……")
        cfg = data_prep.CicidsConfig()
        if args.csv_dir:
            cfg.csv_dir = args.csv_dir
        cfg.out_path = args.data
        data_prep.build(cfg)
    data = data_prep.load_npz(args.data)

    meta = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "data_path": os.path.abspath(args.data),
        "n_features": int(np.asarray(data["feature_names"]).shape[0]),
        "class_names": [str(c) for c in data["class_names"]],
        "n_normal_train": int(np.asarray(data["X_normal_train"]).shape[0]),
        "n_test": int(np.asarray(data["X_test"]).shape[0]),
        "test_by_day": _day_counts(data),
        "eval_exclude_normal_source": _read_eval_flag(args.data),
    }
    try:
        import torch

        meta["torch"] = torch.__version__
        meta["cuda"] = bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        meta["torch"] = None

    suite = make_suite(quick=args.quick)
    results: List[Dict[str, Any]] = []
    os.makedirs(args.out, exist_ok=True)
    for exp in suite:
        print(f"\n>>> 开始实验：{exp.label}  ({exp.name})", flush=True)
        r = run_one(exp, data)
        m = r["main_metrics"]
        print(f"    ROC-AUC={m.get('roc_auc', float('nan')):.4f}  PR-AUC={m.get('pr_auc', float('nan')):.4f}"
              f"  P={m['precision']:.4f}  R={m['recall']:.4f}  F1={m['f1']:.4f}"
              f"  FPR={m['false_positive_rate']:.4f}  ({r['fit']['seconds']:.1f}s)", flush=True)
        results.append(r)
        # 边跑边落盘，避免中途失败丢结果
        partial = build_report(results, meta)
        partial["partial"] = len(results) < len(suite)
        with open(os.path.join(args.out, "repro_report.json"), "w", encoding="utf-8") as fh:
            json.dump(partial, fh, ensure_ascii=False, indent=2, default=str)

    report = build_report(results, meta)
    report["partial"] = False
    # 新旧评估口径对照：旧口径（测试集含检测器训练数据）来自修复前那一版报告
    old = load_old_protocol(args.old_report, args.old_bundle)
    if old is not None:
        new_h = report["headline"]
        report["protocol_comparison"] = {
            "old_protocol": old,
            "new_protocol": {
                "source": "本次运行（见同目录 repro_report.json）",
                "protocol": "修复后：测试集排除检测器的拟合数据（Monday 良性）",
                "n_test": meta["n_test"],
                "n_features": meta["n_features"],
                "test_by_day": meta.get("test_by_day"),
                "metrics": {
                    "threshold": new_h.get("threshold"),
                    "roc_auc": new_h.get("roc_auc"),
                    "pr_auc": new_h.get("pr_auc"),
                    "precision": new_h.get("precision"),
                    "recall": new_h.get("recall"),
                    "f1": new_h.get("f1"),
                    "false_positive_rate": new_h.get("false_positive_rate"),
                    "fp_per_1000_flows": new_h.get("fp_per_1000_flows"),
                    "precision_at_1pct": new_h.get("precision_at_1pct"),
                    "precision_at_0.1pct": new_h.get("precision_at_0.1pct"),
                },
                "test_rows_identical_to_detector_training": _rows_identical_to_training(args.data),
                "note": "主配置（复现主配置）+ 主口径（良性 95% 分位阈值）；"
                        "与旧口径唯一的差别是测试集是否包含 Monday 行。",
            },
        }
    json_path = os.path.join(args.out, "repro_report.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2, default=str)
    print(f"\n[out] 报告已写入 {json_path}")

    if not args.no_plots:
        made = make_plots(report, args.out)
        for p in made:
            print(f"[out] 图已写入 {p}")

    print_summary(report)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
