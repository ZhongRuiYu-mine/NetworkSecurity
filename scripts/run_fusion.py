# -*- coding: utf-8 -*-
"""
两层融合 ICPS 入侵检测 —— 联合仿真与评估
=====================================================================
构造一个"物理电网量测"与"网络流量特征"逐时间步对齐的 ICPS 场景，
让物理层（pandapower IEEE14 + Sage-Husa 自适应 EKF 卡方检验）与
网络层（CIC-IDS2017 训练的 DPNet）同时对同一事件判决，再由
FusionDetector 组合成最终结论。

---------------------------------------------------------------------
场景规划（N_STEP 步，每步一个事件）
---------------------------------------------------------------------
  区间        场景      物理量测        网络流量      应被谁发现
  ----------------------------------------------------------------
  0..29     BENIGN    干净            BENIGN         -
  30..54    FDIA      注入 P/Q 偏置   BENIGN         仅物理层
  55..79    DoS       干净            DoS-Hulk       仅网络层
  80..104   DDoS      干净            DDoS           仅网络层
  105..119  COMBINED  注入偏置        DoS-Hulk       两层（协同攻击）

关键验证点：FDIA 段网络流量完全正常，DPNet 检测率应≈0（盲区），
物理层与融合系统应能高检出 —— 证明两层融合相对任一单层的增益。

运行：python run_fusion.py
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
import gc

from _repo import ARTIFACTS

import numpy as np

from icps_detection.pandapower_model import PandapowerGridModel
from icps_detection.physical_layer import WLSResidualDetector
from icps_detection.datasets import load_from_manifest
from icps_detection.fusion import NetworkProbe, FusionDetector

# ---------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------
BURN_IN = 100          # 干净预热步数（让自适应滤波器收敛）
N_STEP = 120
LOAD_SIGMA = 0.01      # 负荷对数随机游走强度
SEED = 2026

# 物理量测噪声标准差：P/Q 段 MW/MVAr，V 段 p.u.
SIG_PQ = 0.5
SIG_V = 0.002

# FDIA 偏置强度
ATT_BUS = [4, 9]
BIAS_P, BIAS_Q = 15.0, 5.0

# 场景段 (起, 止, 场景名)
SEGMENTS = [
    (0, 30, "BENIGN"),
    (30, 55, "FDIA"),
    (55, 80, "DoS"),
    (80, 105, "DDoS"),
    (105, 120, "COMBINED"),
]


def section(t):
    print("\n" + "=" * 68 + f"\n{t}\n" + "=" * 68)


def scenario_at(k):
    for s, e, name in SEGMENTS:
        if s <= k < e:
            return name
    return "BENIGN"


# =====================================================================
# 网络样本池：按类从 CIC-IDS2017 抽原始 77 维特征
# =====================================================================
def build_network_pool(probe, n_per_class=60):
    """加载 CIC 全量后按类抽样，返回 {类: (n,77) float32}，随即释放。"""
    section("准备网络样本池（从 CIC-IDS2017 抽样）")
    df, _m, _ = load_from_manifest()
    labels = df["Label"].to_numpy()
    pool = {}
    rng = np.random.default_rng(SEED)
    for cls in ["BENIGN", "DoS-Hulk", "DDoS"]:
        idx = np.where(labels == cls)[0]
        take = min(n_per_class, len(idx))
        pick = rng.choice(idx, size=take, replace=False)
        arr = df.loc[pick, probe.feature_names].to_numpy(dtype=np.float32)
        # 兜底：个别格子可能 NaN/Inf
        arr = np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)
        pool[cls] = arr
        print(f"  {cls:10s} 抽到 {take} 条，形状 {arr.shape}")
    del df, labels
    gc.collect()
    return pool


# =====================================================================
# 主流程
# =====================================================================
def main():
    rng = np.random.default_rng(SEED)

    # ---------- 1. 物理电网 ----------
    section("1. 构建 pandapower IEEE14 物理电网")
    grid = PandapowerGridModel("case14")
    n_state, n_meas = grid.n_state, grid.n_meas
    n_p, n_q = len(grid.p_idx), len(grid.q_idx)
    print(f"状态维数 {n_state}（{grid.n_state_bus} 非平衡节点 × 相角/幅值）")
    print(f"量测维数 {n_meas}（P {n_p} + Q {n_q} + V "
          f"{n_meas-n_p-n_q}）")

    # 量测噪声协方差（生成真值时使用，物理层也以它为 R 初值）
    sig = np.zeros(n_meas)
    sig[:n_p + n_q] = SIG_PQ
    sig[n_p + n_q:] = SIG_V
    R_meas = np.diag(sig ** 2)

    # ---------- 2. 物理层检测器（WLS 参考 + 标准化残差卡方） ----------
    section("2. 构建物理层残差检测器（WLS 参考 + 标准化残差卡方）")
    # 先让负荷随机游走一段，使电网工况进入自然波动的稳态
    for _ in range(BURN_IN):
        grid.step_load_walk(LOAD_SIGMA, rng=rng)
    detector = WLSResidualDetector(grid.h, R_meas, alpha=0.05)
    print(f"物理层检测器无状态、对每断面独立检测；卡方门限 "
          f"chi2(0.95,{detector.m}) = {detector.chi_threshold:.2f}")
    print("参考状态 x_ref 由冗余量测 WLS（坏数据剔除）给出，"
          "攻击量测不污染参考。")

    # ---------- 3. 网络探针 + 样本池 ----------
    probe = NetworkProbe(os.path.join(ARTIFACTS, "dpnet_cicids.pt"))
    print(f"\n网络探针就绪：DPNet {len(probe.classes)} 类，"
          f"序列 (T={probe.T},F={probe.F})")
    pool = build_network_pool(probe)

    # ---------- 4. 融合检测器 ----------
    fusion = FusionDetector(detector, probe, w_phys=0.5, w_net=0.5,
                            mode="weighted", score_threshold=0.5)

    # FDIA 攻击向量（量测域恒定偏置）
    a_phys = grid.fdia_vector(ATT_BUS, bias_p=BIAS_P, bias_q=BIAS_Q)

    # ---------- 5. 联合场景逐时间步 ----------
    section(f"5. 联合场景运行 {N_STEP} 步")
    rows = []
    detail_print = set()      # 每段取代表步详细打印
    for s, _e, _n in SEGMENTS:
        detail_print.add(s + 1)

    for k in range(N_STEP):
        scen = scenario_at(k)
        x_true = grid.step_load_walk(LOAD_SIGMA, rng=rng)
        z = grid.h(x_true) + rng.normal(0.0, sig)

        # 物理攻击：FDIA 段 与 COMBINED 段 注入偏置
        if scen in ("FDIA", "COMBINED"):
            z = z + a_phys
        # 参考状态取干净真值 x_true（理想 WLS 参考，不被攻击量测污染）
        phys_info = detector.step(z, x_ref=x_true)

        # 网络样本：BENIGN/FDIA 段配正常流量；DoS/COMBINED 配 DoS；
        # DDoS 段配 DDoS
        net_cls = {"BENIGN": "BENIGN", "FDIA": "BENIGN",
                   "DoS": "DoS-Hulk", "COMBINED": "DoS-Hulk",
                   "DDoS": "DDoS"}[scen]
        cand = pool[net_cls]
        x77 = cand[rng.integers(len(cand))]
        net_info = probe.predict(x77)

        fused = fusion.fuse(phys_info, net_info)
        rows.append((scen, phys_info, net_info, fused))

        if k in detail_print:
            print(f"\n  k={k:3d} 真值={scen:8s} -> 融合={fused['label']}")
            print(f"        物理 p={fused['p_phys']:.3f} "
                  f"alarm={fused['phys_alarm']}  |  "
                  f"网络 p={fused['p_net']:.3f} "
                  f"alarm={fused['net_alarm']}({fused['net_label']})")
            print(f"        fused_score={fused['fused_score']:.3f} "
                  f"来源={fused['source']}")

    # ---------- 6. 分段统计：检测率 / 误报率 ----------
    section("6. 分段检测结果（BENIGN 段为误报率，其余为检测率）")
    print(f"{'场景':10s}{'物理层':>12}{'网络层':>12}{'融合层':>12}")
    seg_rates = {}
    for s, e, name in SEGMENTS:
        pa = np.mean([rows[k][1]["alarm"] for k in range(s, e)])
        na = np.mean([rows[k][2]["alarm"] for k in range(s, e)])
        fa = np.mean([rows[k][3]["alarm"] for k in range(s, e)])
        seg_rates[name] = (pa, na, fa)
        print(f"{name:10s}{pa:12.3f}{na:12.3f}{fa:12.3f}")

    # ---------- 7. 单层 vs 融合 对比 ----------
    section("7. 单层 vs 融合：总体检测率 / 误报率 / 准确率")

    def metrics(layer):
        """layer: 0物理 1网络 2融合。报警口径指标。"""
        tp = fp = tn = fn = 0
        for scen, pi, ni, fu in rows:
            attack = scen != "BENIGN"
            al = (pi["alarm"], ni["alarm"], fu["alarm"])[layer]
            if attack and al:
                tp += 1
            elif attack and not al:
                fn += 1
            elif (not attack) and al:
                fp += 1
            else:
                tn += 1
        det = tp / max(tp + fn, 1)
        far = fp / max(fp + tn, 1)
        acc = (tp + tn) / max(tp + tn + fp + fn, 1)
        return det, far, acc

    print(f"{'方案':16s}{'攻击检测率':>12}{'正常误报率':>12}"
          f"{'准确率':>10}")
    for i, nm in enumerate(["仅物理层", "仅网络层(DPNet)", "两层融合"]):
        d, f, a = metrics(i)
        print(f"{nm:16s}{d:12.3f}{f:12.3f}{a:10.3f}")

    # ---------- 8. 融合标签类别匹配 ----------
    section("8. 融合判决类别与真值匹配")
    expect = {"BENIGN": "BENIGN", "FDIA": "FDIA", "DoS": "DoS-Hulk",
              "DDoS": "DDoS", "COMBINED": "DoS-Hulk+FDIA"}
    print(f"{'真值场景':10s}{'期望融合标签':>18s}{'实际匹配率':>12}")
    for s, e, name in SEGMENTS:
        ok = np.mean([rows[k][3]["label"] == expect[name]
                      for k in range(s, e)])
        print(f"{name:10s}{expect[name]:>18s}{ok:12.3f}")

    print("\n" + "#" * 68)
    print("# 两层融合 ICPS 联合仿真完成")
    print("#" * 68)


if __name__ == "__main__":
    main()
