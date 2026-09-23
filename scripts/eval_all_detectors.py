# -*- coding: utf-8 -*-
r"""eval_all_detectors.py —— 按 PROTOCOL.md v1 统一口径评估所有检测机制。

为什么需要它
------------
合并前，三种机制的"对比"是在两套口径下算的：平台把 `y_test` 传进 `fit()` 走 F1 扫描，
`ae_repro` 走良性分位数；同一个 IF 在两套口径下 F1 是 0.591 / 0.465 / 0.297。
本脚本把口径钉死（见 PROTOCOL.md §2），一条命令产出可直接引用的对比表。

用法
----
    $env:PYTHONPATH = "$PWD\src"
    python scripts/eval_all_detectors.py                        # 全套（AE 训练较慢）
    python scripts/eval_all_detectors.py --smoke                # 冒烟：小样本 + 少轮数
    python scripts/eval_all_detectors.py --detectors if,kalman_frozen

产物
----
    results/detector_comparison.md      ★ 主表（主口径）+ 乐观上界表 + 逐类检出率
    results/detector_comparison.json    原始数字（含全部口径、配置、阈值）
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
import sys
import time
from typing import Any, Dict, List

import numpy as np

from _repo import BUNDLE, DATA, RESULTS

#: 机制清单：显示名 -> (注册表 key, 额外参数)
DETECTORS: Dict[str, Dict[str, Any]] = {
    "if":              {"key": "if",          "kwargs": {}},
    "kalman_frozen":   {"key": "kalman",      "kwargs": {"mode": "frozen"}},
    "kalman_decay":    {"key": "kalman",      "kwargs": {"mode": "decay"}},
    "autoencoder":     {"key": "autoencoder", "kwargs": {}},
    "ae_paper":        {"key": "ae_paper",    "kwargs": {}},
    "null":            {"key": "null",        "kwargs": {}},
}
DEFAULT_ORDER = ["if", "kalman_frozen", "kalman_decay", "autoencoder", "ae_paper", "null"]


def _per_class_detection(y: np.ndarray, flag: np.ndarray, class_names: List[str]) -> Dict[str, float]:
    """每个攻击类别的检出率（AUTO.pdf 口径）。"""
    out: Dict[str, float] = {}
    for i, name in enumerate(class_names):
        if i == 0:
            continue
        m = (y == i)
        if m.sum():
            out[name] = float(flag[m].mean())
    return out


def evaluate_one(name: str, spec: Dict[str, Any], ds, embed_dim: int,
                 seed: int, ae_epochs: int | None, smoke: bool) -> Dict[str, Any]:
    from agentenvs.detectors import make_detector
    from ae_repro.ae_core import binary_metrics, best_f1_threshold

    kwargs = dict(spec["kwargs"])
    kwargs["embed_dim"] = embed_dim
    if smoke and spec["key"] in ("autoencoder", "ae_paper"):
        kwargs["epochs"] = 2
        if spec["key"] == "autoencoder":
            kwargs["hidden_dims"] = (16, 8)
            kwargs["latent_dim"] = 4
        else:
            kwargs["hidden_dims"] = (16, 8)
            kwargs["patience"] = 2
    elif ae_epochs and spec["key"] in ("autoencoder", "ae_paper"):
        kwargs["epochs"] = ae_epochs
    if spec["key"] != "null":
        kwargs.setdefault("seed", seed)

    det = make_detector(spec["key"], **kwargs)

    t0 = time.time()
    # ★ 协议 v1 主口径：只喂良性流量，**不传** X_test/y_test
    #   （传了会触发 INTERFACE.md §2.8 的 F1 扫描 = 用测试标签选阈值）
    det.fit(ds.X_normal_train)
    fit_s = time.time() - t0

    t0 = time.time()
    out = det.score_batch(ds.X_test)
    score = np.asarray(out.score, dtype=np.float64)
    flag = np.asarray(out.flag).astype(int)
    infer_s = time.time() - t0

    y = ds.y_test.astype(int)
    y_bin = (y != 0).astype(int)

    thr_main = float(getattr(det, "threshold", 0.0) or 0.0)
    main = binary_metrics(y_bin, score, thr_main)

    # 辅口径（乐观上界）：用评估集标签扫 F1 —— 只能作为上界并列报告
    thr_f1, f1_at = best_f1_threshold(y_bin, score)
    optimistic = binary_metrics(y_bin, score, thr_f1)

    return {
        "name": name,
        "registry_key": spec["key"],
        "kwargs": {k: (list(v) if isinstance(v, tuple) else v) for k, v in kwargs.items()},
        "embed_dim": int(getattr(det, "embed_dim", 0)),
        "signal_dim": int(det.signal_dim),
        "threshold_mode": getattr(det, "threshold_mode", None),
        "main": main,                      # 主口径（良性分位数 alpha=0.05）
        "optimistic": optimistic,          # 辅口径（F1 扫描，用了测试标签）
        "per_class_detection_rate": _per_class_detection(y, flag, list(ds.class_names)),
        "flag_rate": float(flag.mean()),
        "timing_s": {"fit": round(fit_s, 2), "infer": round(infer_s, 3)},
        "score_stats": {"min": float(score.min()), "max": float(score.max()),
                        "mean": float(score.mean())},
        "n_duplicate_with_normal_train": None,   # 由 check_leakage() 填
    }


def check_leakage(ds) -> Dict[str, Any]:
    """核对评估集与检测器拟合集的逐位重复率（协议 v1 要求 ≤1%）。"""
    def h(X):
        return {r.tobytes() for r in np.ascontiguousarray(X.astype(np.float32))}
    hn = h(ds.X_normal_train)
    dup = np.array([r.tobytes() in hn
                    for r in np.ascontiguousarray(ds.X_test.astype(np.float32))])
    return {"n_test": int(len(dup)),
            "n_duplicate": int(dup.sum()),
            "duplicate_rate": float(dup.mean()),
            "target": "<=1%"}


def fmt(v: float, nd: int = 4) -> str:
    return "n/a" if v is None else f"{v:.{nd}f}"


def to_markdown(payload: Dict[str, Any]) -> str:
    rows = payload["results"]
    leak = payload["leakage"]
    L: List[str] = []
    A = L.append

    A("# 检测机制统一口径对比表（协议 v1）")
    A("")
    A(f"> 生成时间：{payload['generated_at']}　|　数据包：`{payload['data']['path']}`")
    A(f"> 数据包形状：训练 {payload['data']['X_train']}，测试 {payload['data']['X_test']}，"
      f"良性拟合集 {payload['data']['X_normal_train']}")
    A(f"> 环境：{payload['env']['python']}，numpy {payload['env']['numpy']}，"
      f"sklearn {payload['env']['sklearn']}，torch {payload['env']['torch']}")
    A("")
    A("**口径声明（PROTOCOL.md §2）**：主口径 = 良性 95% 分位阈值（`alpha=0.05`，"
      "`fit()` 只喂良性流量、不传 `y_test`），可上线、可外推；"
      "另一个表是 **F1 扫描（乐观上界）**，用了评估集标签选阈值，只可作为上界并列引用。")
    A("")
    A("**评估集自评污染检查**："
      f"`X_test` 与 `X_normal_train` 逐位重复 **{leak['n_duplicate']}/{leak['n_test']} "
      f"= {leak['duplicate_rate']*100:.2f}%**（协议目标 {leak['target']}）。"
      "合并前平台数据包是 4066/29000 = 14.02%。")
    A("")
    A("## 表 1　主口径（良性分位数 α=0.05）")
    A("")
    A("| 机制 | 阈值 | P | R（检出率） | F1 | ROC-AUC | PR-AUC | FPR | FP‰ | P@1% | P@0.1% |")
    A("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        m = r["main"]
        A(f"| `{r['name']}` | {fmt(m['threshold'], 5)} | {fmt(m['precision'])} | "
          f"{fmt(m['recall'])} | **{fmt(m['f1'])}** | {fmt(m.get('roc_auc'))} | "
          f"{fmt(m.get('pr_auc'))} | {fmt(m['false_positive_rate'])} | "
          f"{fmt(m['fp_per_1000_flows'], 1)} | {fmt(m['precision_at_1pct'])} | "
          f"{fmt(m['precision_at_0.1pct'])} |")
    A("")
    A("> `P@1%` / `P@0.1%` 是把攻击占比换成 1% / 0.1% 之后的精确率 —— "
      "本评估集是分层抽样的（攻击占 "
      f"{rows[0]['main']['n_attack']}/{rows[0]['main']['n']} = "
      f"{rows[0]['main']['n_attack']/rows[0]['main']['n']*100:.1f}%），"
      "直接引用 P 会高估可部署性。**FPR / FP‰ 只在良性样本上算，与攻击占比无关，可直接外推。**")
    A("")
    A("## 表 2　逐攻击类别检出率（主口径阈值）")
    A("")
    classes = list(rows[0]["per_class_detection_rate"].keys())
    A("| 机制 | " + " | ".join(classes) + " |")
    A("|---" * (len(classes) + 1) + "|")
    for r in rows:
        cells = [fmt(r["per_class_detection_rate"].get(c, float("nan"))) for c in classes]
        A(f"| `{r['name']}` | " + " | ".join(cells) + " |")
    A("")
    A("## 表 3　辅口径：F1 扫描（★ 乐观上界，用了评估集标签，不可作为主数字）")
    A("")
    A("| 机制 | 阈值 | P | R | F1 | FPR |")
    A("|---|---|---|---|---|---|")
    for r in rows:
        o = r["optimistic"]
        A(f"| `{r['name']}` | {fmt(o['threshold'], 5)} | {fmt(o['precision'])} | "
          f"{fmt(o['recall'])} | {fmt(o['f1'])} | {fmt(o['false_positive_rate'])} |")
    A("")
    A("## 表 4　配置与耗时")
    A("")
    A("| 机制 | 注册 key | embed_dim | signal_dim | 阈值口径 | fit(s) | 推理(s) | 关键参数 |")
    A("|---|---|---|---|---|---|---|---|")
    for r in rows:
        kw = {k: v for k, v in r["kwargs"].items() if k not in ("embed_dim", "seed")}
        A(f"| `{r['name']}` | `{r['registry_key']}` | {r['embed_dim']} | {r['signal_dim']} | "
          f"{r['threshold_mode'] or '—'} | {r['timing_s']['fit']} | {r['timing_s']['infer']} | "
          f"`{json.dumps(kw, ensure_ascii=False)}` |")
    A("")
    A("## 与旧口径的对照（为什么旧表不能用）")
    A("")
    A("`docs/reports/Isolation Forest.docx` 表 5.1 的三方对比是在**旧口径**下得到的：")
    A("")
    A("1. 评估集 29000 行，其中 4066 行（14.02%）与检测器拟合集逐位重复；")
    A("2. 三种机制都走 F1 扫描（用测试标签选阈值）。")
    A("")
    A("本表的数据包与阈值口径都按 `PROTOCOL.md` v1 重做，**取代旧表 5.1**。"
      "两套数字不可混引。")
    A("")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="统一口径评估所有检测机制（协议 v1）")
    ap.add_argument("--data", default=BUNDLE, help="规范数据包（默认 data/cicids_sample.npz）")
    ap.add_argument("--detectors", default=",".join(DEFAULT_ORDER))
    ap.add_argument("--embed-dim", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ae-epochs", type=int, default=None, help="覆盖 AE 类检测器的训练轮数")
    ap.add_argument("--smoke", action="store_true",
                    help="冒烟：AE 只训 2 轮、并在训练/评估集上降采样（验证脚本本身）")
    ap.add_argument("--out-dir", default=RESULTS)
    args = ap.parse_args(argv)

    from agentenvs import load_bundle

    ds = load_bundle(args.data)
    if args.smoke:
        # 注意：X 与 y 必须用**同一组下标**降采样，否则特征与标签错位，
        # AUC 会掉到 0.5 附近（这个坑第一次冒烟就踩到了）。
        rng = np.random.default_rng(0)

        def take(a, k=4000):
            k = min(len(a), k)
            return a[rng.choice(len(a), size=k, replace=False)]

        ds.X_normal_train = take(ds.X_normal_train)
        idx = rng.choice(len(ds.X_train), size=min(len(ds.X_train), 4000), replace=False)
        ds.X_train, ds.y_train = ds.X_train[idx], ds.y_train[idx]
        idx = rng.choice(len(ds.X_test), size=min(len(ds.X_test), 4000), replace=False)
        ds.X_test, ds.y_test = ds.X_test[idx], ds.y_test[idx]

    print("=" * 78)
    print("  统一口径检测机制评估（PROTOCOL.md v1）")
    print("=" * 78)
    print(f"数据包   : {args.data}")
    print(f"训练/测试: {ds.X_train.shape} / {ds.X_test.shape}")
    print(f"良性拟合集: {ds.X_normal_train.shape}")
    print(f"类别空间 : {list(ds.class_names)}")

    leak = check_leakage(ds)
    print(f"自评污染 : {leak['n_duplicate']}/{leak['n_test']} "
          f"= {leak['duplicate_rate']*100:.2f}%  (目标 {leak['target']})")

    results = []
    for name in [s.strip() for s in args.detectors.split(",") if s.strip()]:
        if name not in DETECTORS:
            print(f"[跳过] 未知机制 {name!r}，可选：{list(DETECTORS)}")
            continue
        print("\n" + "-" * 78)
        r = evaluate_one(name, DETECTORS[name], ds, args.embed_dim, args.seed,
                         args.ae_epochs, args.smoke)
        r["n_duplicate_with_normal_train"] = leak["n_duplicate"]
        m = r["main"]
        print(f"  {name:16s} 主口径 thr={m['threshold']:.5f}  "
              f"P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f} "
              f"AUC={m.get('roc_auc', float('nan')):.4f} FPR={m['false_positive_rate']:.4f}")
        print(f"  {'':16s} 乐观上界(F1扫描) F1={r['optimistic']['f1']:.4f} "
              f"| fit {r['timing_s']['fit']}s")
        results.append(r)

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
                 "class_names": [str(c) for c in ds.class_names],
                 "embed_dim": args.embed_dim, "seed": args.seed,
                 "smoke": bool(args.smoke)},
        "leakage": leak,
        "detectors": DETECTORS,
        "results": results,
    }

    os.makedirs(args.out_dir, exist_ok=True)
    jp = os.path.join(args.out_dir, "detector_comparison.json")
    mp = os.path.join(args.out_dir, "detector_comparison.md")
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
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    raise SystemExit(main())
