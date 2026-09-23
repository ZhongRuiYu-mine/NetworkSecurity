# -*- coding: utf-8 -*-
"""
CIC-IDS2017 真实数据 DPNet 训练脚本
=====================================================================
端到端管线：
  1. load_from_manifest  加载全部 8 个工作日 parquet（~231 万行）
  2. 分层切分（80/20，仅训练集参与后续重采样，杜绝泄漏）
  3. BENIGN 下采样至 CAP_BENIGN（原始 ~170 万，避免多数类淹没 +
     加速训练；攻击类全保留）
  4. 对稀有类（WebAttack* / Infiltration / Bot / Heartbleed）按
     one-vs-rest ADASYN 过采样至 TARGET_PER_RARE 条
  5. Standardizer 仅 fit 训练集，transform 训练/测试
  6. 特征向量 (77,) -> 序列 (T, F)  reshape（T*F = n_features）
  7. DPNet 双通道（3 层 Conv1d+MaxPool / 3 层 BiLSTM hidden=128）
     -> 融合 -> 2 层 Dense -> Softmax
  8. 训练 EPOCHS 轮，每轮打印 train/val 准确率与耗时
  9. 测试集混淆矩阵 + 每类 Precision/Recall/F1

运行：python train_cicids.py
（需要全局环境已装 cu128 torch + pyarrow）
"""


# --- 合并仓库引导（scripts/_bootstrap） ---
import os as _os, sys as _sys
REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_os.path.join(REPO_ROOT, "src"), REPO_ROOT):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
del _os, _sys, _p
# --- 合并仓库引导结束 ---

import json
import os
import sys
import time
import numpy as np

from _repo import ARTIFACTS

from icps_detection.datasets import (
    load_from_manifest,
    to_xy,
    DEFAULT_RARE_CLASSES,
)
from icps_detection.preprocessing import ADASYN, Standardizer, get_device
from icps_detection.dpnet import DPNet, train_epoch, evaluate

import torch
from torch.utils.data import TensorDataset, DataLoader

# ---------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------
CAP_BENIGN = 200_000        # 训练集 BENIGN 上限（原始约 170 万）
TARGET_PER_RARE = 20_000   # 每个稀有类 ADASYN 目标条数
TEST_SIZE = 0.2
EPOCHS = 10
BATCH = 1024
SEED = 42


# ---------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------
def section(title):
    print("\n" + "=" * 66 + f"\n{title}\n" + "=" * 66)


def stratified_split(X, y, test_size, seed):
    """按类别分层切分；样本极少类至少留 1 条测试。"""
    rng = np.random.default_rng(seed)
    tr_idx, te_idx = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n_te = max(1, int(round(len(idx) * test_size))) if len(idx) > 1 else 0
        te_idx.extend(idx[:n_te])
        tr_idx.extend(idx[n_te:])
    return np.asarray(tr_idx), np.asarray(te_idx)


def subsample_benign(X, y, cap, seed):
    """训练集内 BENIGN 随机下采样到 cap，攻击类全保留。"""
    rng = np.random.default_rng(seed)
    benign_idx = np.where(y == "BENIGN")[0]
    if len(benign_idx) > cap:
        keep = rng.choice(benign_idx, size=cap, replace=False)
        drop = np.setdiff1d(benign_idx, keep)
        mask = np.ones(len(y), dtype=bool)
        mask[drop] = False
        return X[mask], y[mask]
    return X, y


def oversample_rare_to_target(X, y, rare_classes, target, seed):
    """
    对每个稀有类做 one-vs-rest ADASYN，将其补到 target 条。
    每类独立计算 beta = (target - m_s) / (m_l - m_s) 以精确命中目标。
    """
    X_cur, y_cur = X.copy(), y.copy()
    for c in rare_classes:
        m_s = int((y_cur == c).sum())
        if m_s == 0:
            print(f"  [ADASYN] {c:22s} 训练集无样本，跳过")
            continue
        if m_s >= target:
            print(f"  [ADASYN] {c:22s} {m_s} >= {target}，跳过")
            continue
        m_l = int((y_cur != c).sum())
        G = target - m_s
        beta = min(max(G / max(m_l - m_s, 1), 1e-4), 1.0)
        k = min(5, m_s - 1)
        if k < 1:
            print(f"  [ADASYN] {c:22s} 样本过少({m_s})，跳过")
            continue
        sampler = ADASYN(beta=beta, k_neighbors=k, random_state=seed)
        y_bin = np.where(y_cur == c, 1, 0)
        X_new, _ = sampler.fit_resample(X_cur, y_bin)
        n_added = X_new.shape[0] - X_cur.shape[0]
        y_cur = np.concatenate(
            [y_cur, np.full(n_added, c, dtype=y_cur.dtype)])
        X_cur = X_new
        print(f"  [ADASYN] {c:22s} {m_s} -> {m_s + n_added}")
    return X_cur, y_cur


def pick_seq_shape(n_feat):
    """把 n_feat 因式分解为 (T, F)，偏好 T 在 [4, 16]。"""
    for T in range(min(n_feat, 16), 3, -1):
        if n_feat % T == 0:
            return T, n_feat // T
    return n_feat, 1


def per_class_report(y_true, y_pred, classes):
    """混淆矩阵 + 每类 P/R/F1（纯 numpy，无 sklearn 依赖）。"""
    n = len(classes)
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    short = [c[:10] for c in classes]
    print("\n混淆矩阵 (行=真实, 列=预测):")
    print(" " * 12 + " ".join(f"{s:>10}" for s in short))
    for i, c in enumerate(classes):
        print(f"{c[:10]:>12} " + " ".join(f"{cm[i,j]:>10}" for j in range(n)))

    print(f"\n{'class':22s}{'prec':>8}{'rec':>8}{'f1':>8}{'support':>10}")
    f1s = []
    per_class = []
    for i, c in enumerate(classes):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        sup = int(cm[i, :].sum())
        prec = tp / (tp + fp) if tp + fp > 0 else 0.0
        rec = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
        f1s.append(f1)
        per_class.append({"class": str(c), "precision": float(prec),
                          "recall": float(rec), "f1": float(f1),
                          "support": sup, "tp": tp, "fp": fp, "fn": fn})
        print(f"  {c:20s}{prec:8.3f}{rec:8.3f}{f1:8.3f}{sup:10d}")
    acc = cm.diagonal().sum() / max(cm.sum(), 1)
    macro_f1 = float(np.mean(f1s))
    print(f"\n总体准确率 = {acc:.4f}   宏平均 F1 = {macro_f1:.4f}")
    return float(acc), macro_f1, per_class


# ---------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------
def main():
    section("1. 加载 CIC-IDS2017 全部 8 个工作日")
    t0 = time.time()
    df, manifest, loaded = load_from_manifest()
    print(f"加载 {len(loaded)} 个文件，合计 {len(df):,} 行 "
          f"({time.time()-t0:.1f}s)")
    print("标签分布:")
    vc = df["Label"].value_counts()
    for c, n in vc.items():
        print(f"  {c:22s} {n:>10,}")

    section("2. 特征化 + 分层切分（80/20）")
    X, y, feat_names = to_xy(df)
    del df
    print(f"特征矩阵 {X.shape}  特征数 = {len(feat_names)}")
    tr_idx, te_idx = stratified_split(X, y, TEST_SIZE, SEED)
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_te, y_te = X[te_idx], y[te_idx]
    print(f"切分后 训练 {X_tr.shape}  测试 {X_te.shape}")

    section(f"3. BENIGN 下采样至 {CAP_BENIGN:,}")
    X_tr, y_tr = subsample_benign(X_tr, y_tr, CAP_BENIGN, SEED)
    print(f"下采样后训练集 {X_tr.shape}")
    print("训练集标签分布:")
    for c in np.unique(y_tr):
        print(f"  {c:22s} {int((y_tr==c).sum()):>10,}")

    section(f"4. 稀有类 ADASYN 过采样至 {TARGET_PER_RARE:,}/类")
    X_tr, y_tr = oversample_rare_to_target(
        X_tr, y_tr, DEFAULT_RARE_CLASSES, TARGET_PER_RARE, SEED)
    print(f"过采样后训练集 {X_tr.shape}")

    section("5. 标准化（仅训练集 fit）")
    scaler = Standardizer().fit(X_tr)
    X_tr = scaler.transform(X_tr).astype(np.float32)
    X_te = scaler.transform(X_te).astype(np.float32)

    section("6. 标签编码 + 序列 reshape")
    classes = list(np.unique(np.concatenate([y_tr, y_te])))
    cls2idx = {c: i for i, c in enumerate(classes)}
    y_tr_i = np.array([cls2idx[c] for c in y_tr], dtype=np.int64)
    y_te_i = np.array([cls2idx[c] for c in y_te], dtype=np.int64)
    print(f"类别 ({len(classes)}): {classes}")

    n_feat = X_tr.shape[1]
    T, F = pick_seq_shape(n_feat)
    print(f"特征数 {n_feat} -> 序列 (T={T}, F={F})")
    X_tr3 = X_tr.reshape(-1, T, F)
    X_te3 = X_te.reshape(-1, T, F)
    print(f"训练张量 {X_tr3.shape}  测试张量 {X_te3.shape}")

    section("7. 构造 DataLoader")
    device = get_device()
    print(f"设备: {device}")
    if device.type == "cuda":
        # 用户偏好：GPU 显存限制 80%，避免影响系统其他用途
        torch.cuda.set_per_process_memory_fraction(0.8, 0)
        free, total = torch.cuda.mem_get_info(0)
        print(f"GPU 显存: {total/1e9:.1f} GB 总，单进程上限 80% = "
              f"{total*0.8/1e9:.1f} GB；当前空闲 {free/1e9:.1f} GB")
    ds_tr = TensorDataset(torch.from_numpy(X_tr3), torch.from_numpy(y_tr_i))
    ds_te = TensorDataset(torch.from_numpy(X_te3), torch.from_numpy(y_te_i))
    loader_tr = DataLoader(ds_tr, batch_size=BATCH, shuffle=True,
                           num_workers=0, pin_memory=(device.type == "cuda"))
    loader_te = DataLoader(ds_te, batch_size=BATCH * 2, shuffle=False,
                           num_workers=0, pin_memory=(device.type == "cuda"))

    section("8. 构建 DPNet + 训练")
    model = DPNet(n_features=F, seq_len=T, num_classes=len(classes)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"DPNet 参数量 = {n_params:,}")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = torch.nn.NLLLoss()
    best_acc, best_state = 0.0, None
    for ep in range(EPOCHS):
        t0 = time.time()
        loss, acc_tr = train_epoch(model, loader_tr, opt, crit, device)
        acc_te, _, _, _ = evaluate(model, loader_te, device)
        dt = time.time() - t0
        print(f"  epoch {ep+1:2d}/{EPOCHS}  loss={loss:.4f}  "
              f"train_acc={acc_tr:.4f}  val_acc={acc_te:.4f}  ({dt:.1f}s)")
        if acc_te > best_acc:
            best_acc, best_state = acc_te, {k: v.clone() for k, v
                                            in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)

    section("9. 测试集最终评估")
    acc, yt, yp, _ = evaluate(model, loader_te, device)
    acc, macro_f1, per_class = per_class_report(yt, yp, classes)

    # ------------------------------------------------------------------
    # 落盘：必须与 icps_detection.fusion.NetworkProbe 的读取契约一致
    #   （NetworkProbe 需要 scaler_mu / scaler_sigma —— 合并前缺这两个键，
    #     导致重跑训练会覆盖掉 fusion 读不了的 checkpoint，见 GAP_ANALYSIS §2.4）
    # 指标同时落盘成 JSON，避免"评测结果只存在于终端输出里"。
    # ------------------------------------------------------------------
    os.makedirs(ARTIFACTS, exist_ok=True)
    ckpt = os.path.join(ARTIFACTS, "dpnet_cicids.pt")
    torch.save({"model_state": model.state_dict(),
                "classes": classes, "seq_T": T, "seq_F": F,
                "feature_names": feat_names,
                "scaler_mu": scaler.mu_, "scaler_sigma": scaler.sigma_,
                "test_acc": float(acc),
                "config": {"cap_benign": CAP_BENIGN,
                           "target_per_rare": TARGET_PER_RARE,
                           "test_size": TEST_SIZE, "epochs": EPOCHS,
                           "batch": BATCH, "seed": SEED}}, ckpt)
    metrics_path = os.path.join(ARTIFACTS, "dpnet_cicids.metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"test_acc": float(acc), "macro_f1": float(macro_f1),
                   "n_test": int(len(yt)),
                   "classes": [str(c) for c in classes],
                   "per_class": per_class}, f, indent=2, ensure_ascii=False)
    print(f"\n模型已保存: {ckpt}")
    print(f"指标已保存: {metrics_path}")
    print("[提示] 该 checkpoint 现在可被 scripts/run_fusion.py 的 NetworkProbe 读取")


if __name__ == "__main__":
    main()
