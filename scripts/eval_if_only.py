"""
Isolation Forest 检测器独立评估
- 只用良性流量训练
- 在测试集上评估：Precision / Recall / F1 / 检测率 / 误报率 / AUC
- 输出混淆矩阵、异常分数分布图
- 对照 IsoFor2.pdf 的评估方式
"""


# --- 合并仓库引导（scripts/_bootstrap） ---
import os as _os, sys as _sys
REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_os.path.join(REPO_ROOT, "src"), REPO_ROOT):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
del _os, _sys, _p
# --- 合并仓库引导结束 ---

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)

from agentenvs import load_bundle, make_detector


def main():
    # ---- 0. 解析命令行参数 ----
    detector_name = sys.argv[1] if len(sys.argv) > 1 else "if"
    print(f"检测器: {detector_name}")

    # ---- 1. 加载数据 ----
    from _repo import BUNDLE, RESULTS
    ds = load_bundle(BUNDLE)
    print("=" * 70)
    print(f"{detector_name} 独立评估")
    print("=" * 70)
    print(f"训练集: {ds.X_train.shape}, 测试集: {ds.X_test.shape}")
    print(f"良性训练样本: {ds.X_normal_train.shape}")
    print(f"类别: {ds.class_names}")

    # ---- 2. 创建 + 训练 ----
    det = make_detector(detector_name, embed_dim=4)
    det.fit(
        ds.X_normal_train,
        X_test=ds.X_test,
        y_test=(ds.y_test != 0).astype(int),
    )
    print(f"\n{detector_name} 训练完成")
    if hasattr(det, "threshold") and det.threshold is not None:
        print(f"  阈值: {det.threshold:.6f}")
    if hasattr(det, "_normal_stats") and det._normal_stats is not None:
        print(f"  良性分数均值: {det._normal_stats[0]:.6f}")
        print(f"  良性分数标准差: {det._normal_stats[1]:.6f}")

    # ---- 3. 在测试集上打分 ----
    out = det.score_batch(ds.X_test)
    scores = out.score
    flags = out.flag.astype(int)
    y_true = (ds.y_test != 0).astype(int)

    # ---- 4. 计算指标 ----
    precision = precision_score(y_true, flags, zero_division=0)
    recall = recall_score(y_true, flags, zero_division=0)
    f1 = f1_score(y_true, flags, zero_division=0)
    auc = roc_auc_score(y_true, scores)
    cm = confusion_matrix(y_true, flags)
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    print("\n" + "=" * 70)
    print(f"{detector_name} 检测器在 CICIDS2017 测试集上的表现")
    print("=" * 70)
    print(f"  Accuracy:         {accuracy:.4f}")
    print(f"  Precision:        {precision:.4f}")
    print(f"  Recall (检测率):  {recall:.4f}")
    print(f"  F1:               {f1:.4f}")
    print(f"  AUC:              {auc:.4f}")
    print(f"  误报率 (FPR):     {false_alarm_rate:.4f}")
    print(f"  混淆矩阵:  TP={tp}, FP={fp}, FN={fn}, TN={tn}")

    # ---- 5. 分类报告 ----
    print("\n二分类报告:")
    print(classification_report(y_true, flags, target_names=["Normal", "Attack"], digits=4))

    # ---- 6. 保存结果 ----
    out_dir = os.path.join(RESULTS, f"{detector_name}_only")
    os.makedirs(out_dir, exist_ok=True)
    result = {
        "detector": detector_name,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc),
        "false_alarm_rate": float(false_alarm_rate),
        "confusion_matrix": {"TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn)},
        "threshold": float(det.threshold) if getattr(det, "threshold", None) is not None else None,
        "n_test": int(len(y_true)),
        "n_attack": int(y_true.sum()),
    }
    json_path = f"{out_dir}/eval_{detector_name}_only.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] 结果已保存到 {json_path}")

    # ---- 7. 画 ROC 曲线 ----
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, scores)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"{detector_name} (AUC = {auc:.3f})", color="orange", lw=2)
    plt.plot([0, 1], [0, 1], "--", color="navy", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {detector_name}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    roc_path = f"{out_dir}/roc_curve.png"
    plt.savefig(roc_path, dpi=150)
    plt.close()
    print(f"[OK] ROC 曲线已保存到 {roc_path}")

    # ---- 8. 画异常分数分布 ----
    plt.figure(figsize=(8, 5))
    plt.hist(scores[y_true == 0], bins=60, alpha=0.6, label="Normal", color="steelblue", density=True)
    plt.hist(scores[y_true == 1], bins=60, alpha=0.6, label="Attack", color="crimson", density=True)
    if getattr(det, "threshold", None) is not None:
        plt.axvline(det.threshold, color="black", linestyle="--", label=f"Threshold = {det.threshold:.3f}")
    plt.xlabel("Anomaly Score (higher = more anomalous)")
    plt.ylabel("Density")
    plt.title(f"Anomaly Score Distribution - {detector_name}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    dist_path = f"{out_dir}/score_distribution.png"
    plt.savefig(dist_path, dpi=150)
    plt.close()
    print(f"[OK] 分数分布图已保存到 {dist_path}")

    # ---- 9. 保存检测器 ----
    det_path = f"{out_dir}/{detector_name}_det.npz"
    try:
        det.save(det_path)
        print(f"[OK] 检测器已保存到 {det_path}")
    except Exception as e:
        print(f"[警告] 检测器保存失败: {e}")

    print("\n" + "=" * 70)
    print("完成")
    print("=" * 70)


if __name__ == "__main__":
    main()