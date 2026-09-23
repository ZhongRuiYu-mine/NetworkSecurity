# -*- coding: utf-8 -*-
"""
train_detector.py —— 在复现协议数据上训练并落盘「复现版自编码器」，供组员直接复用。

产出（默认写到 `ae_repro/artifacts/`）
------------------------------------
    ae_paper.npz              权重 + 阈值 + 逐特征残差统计 + embedding 选维
    ae_paper.json             配置、训练元信息、阈值、EVT 参数、训练历史
    ae_paper.features.json    特征名（归因可读）
    ae_paper.metrics.json     在测试集上的检测指标（P/R/F1/FPR/AUC…）

用法
----
    # ① 用已有数据包训练（推荐，秒级）
    python -m ae_repro.train_detector --data _cache/cicids_sample.npz

    # ② 没有数据包时现场构建（约 10 秒解析 CSV，需 chethuhn 数据集）
    python -m ae_repro.train_detector --csv-dir "D:/.../versions/1"

之后组员只需要：
    det = make_detector("ae_paper").load("ae_repro/artifacts/ae_paper")   # 或
    from ae_repro.ae_detector import AePaperDetector
    det = AePaperDetector().load("ae_repro/artifacts/ae_paper")
    env = make_env(ds, detector=det, detector_embed_dim=4)
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from typing import Optional, Sequence

import numpy as np

from . import data_prep
from .ae_core import AEConfig, binary_metrics, best_f1_threshold, fit_evt_threshold
from .ae_detector import AePaperDetector

from .paths import BUNDLE as _BUNDLE

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ARTIFACTS = os.path.join(_HERE, "artifacts")
DEFAULT_DATA = _BUNDLE


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="训练并落盘复现版自编码器检测器")
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--csv-dir", default=None, help="数据包不存在时用它现场构建")
    ap.add_argument("--out", default=DEFAULT_ARTIFACTS)
    ap.add_argument("--name", default="ae_paper")
    ap.add_argument("--embed-dim", type=int, default=4, help="平台对比实验统一 4")
    ap.add_argument("--latent-dim", type=int, default=4)
    # 默认与 experiments.py 的「复现主配置」一致（epochs=250, patience=80），
    # 这样落盘的 artifact 就是报告里那一版模型，避免"交付的检测器 ≠ 报告的模型"。
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    if not os.path.exists(args.data):
        print(f"[info] 找不到 {args.data}，现场构建数据包……")
        cfg = data_prep.CicidsConfig()
        if args.csv_dir:
            cfg.csv_dir = args.csv_dir
        cfg.out_path = args.data
        data_prep.build(cfg)

    d = data_prep.load_npz(args.data)
    Xn = np.asarray(d["X_normal_train"], dtype=np.float32)
    Xte = np.asarray(d["X_test"], dtype=np.float32)
    yte = np.asarray(d["y_test"]).astype(int)
    ybin = (yte != 0).astype(int)
    feats = [str(f) for f in d["feature_names"]]

    print(f"[train] 良性训练集 {Xn.shape}  测试集 {Xte.shape}  特征 {len(feats)} 维")
    det = AePaperDetector(latent_dim=args.latent_dim, epochs=args.epochs,
                          alpha=args.alpha, embed_dim=args.embed_dim, seed=args.seed)
    t0 = time.perf_counter()
    det.fit(Xn, X_test=Xte, y_test=ybin, feature_names=feats)
    print(f"[train] 完成，用时 {time.perf_counter() - t0:.1f}s")
    print(det.describe())

    out = det.score_batch(Xte)
    assert out.score.shape == (Xte.shape[0],)
    assert out.flag.shape == (Xte.shape[0],)
    assert out.trust.shape == (Xte.shape[0],)
    assert out.embedding is not None and out.embedding.shape[1] == args.embed_dim
    assert np.isfinite(out.score).all() and np.isfinite(out.trust).all()
    sig = out.as_signal(args.embed_dim)
    assert sig.shape == (Xte.shape[0], 3 + args.embed_dim), sig.shape
    print(f"[check] 契约自检通过：signal_dim={det.signal_dim}  "
          f"[score,flag,trust,emb...] 矩阵 {sig.shape}")

    thr = float(det.threshold)
    evt = fit_evt_threshold(det.core.score(Xn), alpha=args.alpha)
    thr_f1, _ = best_f1_threshold(ybin, out.score)
    metrics = {
        "main_rule": "percentile",
        "threshold": thr,
        "at_threshold": binary_metrics(ybin, out.score, thr),
        "at_evt_threshold": binary_metrics(ybin, out.score, float(evt.threshold)),
        "at_best_f1_threshold": binary_metrics(ybin, out.score, float(thr_f1)),
        "evt": evt.to_dict(),
        "embedding": {
            "dim": args.embed_dim,
            "feature_idx": [] if det.core.embed_idx is None else [int(i) for i in det.core.embed_idx],
            "feature_names": [
                feats[i] for i in ([] if det.core.embed_idx is None else det.core.embed_idx)
                if 0 <= i < len(feats)
            ],
        },
        "attribution_top15": [
            {"feature": n, "mean_sq_residual": v} for n, v in det.feature_attribution(15)
        ],
        "latency": det.core.latency(Xte, repeats=2),
        "fit_meta": det.core.fit_meta,
    }

    os.makedirs(args.out, exist_ok=True)
    base = os.path.join(args.out, args.name)
    det.save(base)
    with open(base + ".metrics.json", "w", encoding="utf-8") as fh:
        json.dump({
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "data": os.path.abspath(args.data),
            "detector": "AePaperDetector (AUTO.pdf stage-1 AE reproduction)",
            "paper_reference": {
                "AE_row_Table1": {"ROC_AUC": 0.949, "precision": 0.82, "recall": 0.79, "F1": 0.80},
                "dataset_of_paper": "blockchain metaverse transactions (not CICIDS2017)",
            },
            "metrics": metrics,
        }, fh, ensure_ascii=False, indent=2, default=str)

    m = metrics["at_threshold"]
    print(f"\n[metrics] 主口径(分位阈值={thr:.4f})  ROC-AUC={m.get('roc_auc', float('nan')):.4f} "
          f"P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f} FPR={m['false_positive_rate']:.4f}")
    print(f"[metrics] 基率校正  FP‰={m['fp_per_1000_flows']:.1f}  "
          f"P@1%={m['precision_at_1pct']:.4f}  P@0.1%={m['precision_at_0.1pct']:.4f}")
    print(f"[metrics] trust 尺度 = {det.core.score_robust_scale:.4g}"
          f"（稳健 IQR/1.349；std={det.core.score_sigma:.4g}）")
    print(f"[out] 检测器已落盘：{base}.npz / .json / .metrics.json")
    print("\n组员加载方式：")
    print(f"    det = make_detector('ae_paper').load(r'{base}')")
    print(f"    env = make_env(ds, detector=det, detector_embed_dim={args.embed_dim})")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
