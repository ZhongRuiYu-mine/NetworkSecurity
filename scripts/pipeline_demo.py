# -*- coding: utf-8 -*-
"""
复现管线集成演示（按三条复现建议组织）
=====================================================================
A. pandapower IEEE-14 总线模型生成真实 f(x)/h(x)，负荷随机游走
   生成 200 个时间断面的真值轨迹，叠加 SCADA 量测噪声并注入 FDIA；
B. 使用 Sage-Husa 自适应 EKF（IVB-NCA-NLKF 的平替）跑通物理层
   检测流程：干净数据标定阈值 -> 在线自适应 -> 攻击门控；
C. 用 CIC 结构的小样本 DataFrame 验证 prepare_cicids（切分 +
   Web Attack/Infiltration 等少数类 ADASYN + 训练统计量标准化），
   并探测官方下载站点连通性（真实数据通过 download_cicids2017 拉取）。

运行：python pipeline_demo.py
"""


# --- 合并仓库引导（scripts/_bootstrap） ---
import os as _os, sys as _sys
REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_os.path.join(REPO_ROOT, "src"), REPO_ROOT):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
del _os, _sys, _p
# --- 合并仓库引导结束 ---

import socket
import urllib.request
import urllib.error

import numpy as np
import pandas as pd

from icps_detection.physical_layer import SageHusaAEKF
from icps_detection.pandapower_model import PandapowerGridModel
from icps_detection.datasets import prepare_cicids, CIC_URLS


def section(title):
    print("\n" + "=" * 66)
    print(title)
    print("=" * 66)


# =====================================================================
# A + B. pandapower 物理模型 + Sage-Husa AEKF
# =====================================================================
def demo_power_sagehusa():
    section("A/B. IEEE-14 + Sage-Husa 自适应 EKF（IVB 平替）")
    rng = np.random.default_rng(2026)

    grid = PandapowerGridModel("case14")
    print(f"状态维数 n = {grid.n_state}（{grid.n_state_bus} 个非平衡节点 "
          f"的 theta/V），量测维数 m = {grid.n_meas}")

    # ---------- 1) 生成真值轨迹（负荷随机游走 + 潮流）----------
    N = 200
    grid.reset()
    states = [grid.state_from_results()]
    for _ in range(N - 1):
        states.append(grid.step_load_walk(sigma=0.004, rng=rng))
    states = np.asarray(states)

    # ---------- 2) 量测：h(x) + 高斯噪声 ----------
    # 量测段 [P(MW), Q(MVAr), V(pu)]，按尺度分别设定噪声
    n_p = len(grid.p_idx)
    n_q = len(grid.q_idx)
    n_v = len(grid.v_idx)
    std = np.concatenate([
        np.full(n_p, 0.3),     # P: 0.3 MW
        np.full(n_q, 0.3),     # Q: 0.3 MVAr
        np.full(n_v, 2e-3),    # V: 0.002 p.u.
    ])
    R0 = np.diag(std ** 2)
    zs = np.asarray([grid.h(x) for x in states])
    zs = zs + rng.normal(0, std, size=zs.shape)

    # ---------- 3) FDIA：k=150 对前 3 个非平衡节点注入偏置 ----------
    attack_step = 150
    targets = grid.state_buses[:3]
    a = grid.fdia_vector(targets, bias_p=8.0, bias_q=4.0, bias_v=0.02)
    zs[attack_step] += a

    # ---------- 4) Sage-Husa：前 100 步干净数据标定阈值 ----------
    # Q 必须真实覆盖负荷随机游走引起的状态漂移：
    #   theta std ~ 2e-3 rad, V std ~ 1e-4 pu
    n_theta = grid.n_state_bus
    Q_diag = np.concatenate([
        np.full(n_theta, (2e-3) ** 2),
        np.full(n_theta, (1e-4) ** 2),
    ])
    Q0 = np.diag(Q_diag)
    # P0 必须与 Q 同尺度（P=I 会使首步 HPH^T 项失真）
    P0 = 10.0 * Q0
    filt = SageHusaAEKF(
        grid.f, grid.h, Q0, R0, x0=states[0], P0=P0,
        F_fun=grid.F, forgetting=0.97, gate_adaptation=True,
        warmup=10,
    )
    tau = filt.calibrate_threshold(zs[:100], refs=states[:100])
    print(f"标定残差阈值 tau = {tau:.4f}")

    # ---------- 5) 在线检测 ----------
    alarms = []
    for k in range(100, N):
        info = filt.step(zs[k], x_ref=states[k])
        alarms.append(info["alarm"])
    alarms = np.asarray(alarms)
    local = attack_step - 100

    pre = alarms[:local]
    print(f"攻击点(k=150) 报警 = {alarms[local]}")
    print(f"攻击前干净区间误报 = {pre.sum()}/{pre.size}")
    print(f"估计 R 特征值范围 = "
          f"{np.linalg.eigvalsh(filt.R).min():.2e} ~ "
          f"{np.linalg.eigvalsh(filt.R).max():.2e}")
    assert alarms[local]
    print("[OK] Sage-Husa 平替流程跑通，攻击点成功检出")


# =====================================================================
# C. CIC 数据准备（结构验证）+ 官方站点连通性
# =====================================================================
def _make_cic_like_frame(seed=0):
    """构造一个结构与 CIC-IDS2017 一致的小样本 DataFrame。"""
    rng = np.random.default_rng(seed)
    dist = [
        ("BENIGN", 600), ("DoS-Hulk", 120), ("PortScan", 80),
        ("WebAttack-BruteForce", 15), ("Infiltration", 9),
        ("Heartbleed", 3),
    ]
    labels = np.concatenate([np.full(n, c) for c, n in dist])
    n = len(labels)
    data = {f"feat_{i}": rng.normal(0, 1, n) for i in range(12)}
    # 标识符/时间列（应在 to_xy 中被丢弃）
    data["Flow ID"] = [f"flow_{i}" for i in range(n)]
    data["Source IP"] = ["10.0.0.1"] * n
    data["Destination IP"] = ["192.168.0.1"] * n
    data["Timestamp"] = pd.date_range("2026-01-01", periods=n, freq="s")
    data["Label"] = labels
    return pd.DataFrame(data)


def demo_cic_prepare():
    section("C. CIC-IDS2017 数据准备管线（结构验证）")
    df = _make_cic_like_frame()

    rare = ["WebAttack-BruteForce", "Infiltration", "Heartbleed"]
    out = prepare_cicids(df, test_size=0.2, rare_classes=rare,
                         beta=1.0, random_state=42)

    # 标识符列不能出现在特征中
    assert "Flow ID" not in out["feature_names"]
    # 稀有类应被过采样到接近主流类量级（受最大类上限约束）
    train_counts = out["train_counts"]
    print(f"过采样后 Infiltration 训练样本 = "
          f"{train_counts.get('Infiltration', 0)}（原始训练集约 7）")
    assert train_counts.get("Infiltration", 0) > 7
    print("[OK] 切分 -> 少数类 ADASYN -> 标准化流程无泄漏")


def check_dataset_host():
    section("C-附. CIC-IDS2017 官方站点连通性探测")
    socket.setdefaulttimeout(15)
    reachable = False
    for url in CIC_URLS:
        host = url.split("/")[2]
        try:
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req, timeout=15) as resp:
                size = resp.headers.get("Content-Length", "未知")
                print(f"[可达] {host}  HTTP {resp.status}  "
                      f"Content-Length={size}")
                reachable = True
                break
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            print(f"[不可达] {host}  原因: {type(e).__name__}: {e}")
    if reachable:
        print("执行以下命令即可开始真实下载（约 200+ MB）：")
        print("    python -c \"from icps_detection.datasets import "
              "download_cicids2017; download_cicids2017()\"")
    else:
        print("官方站点当前不可达。可：1) 检查网络/代理后重试；")
        print("2) 手动下载 MachineLearningCSV.zip 放入 data/；")
        print("3) 使用 Kaggle 页面（需登录）。")


if __name__ == "__main__":
    demo_power_sagehusa()
    demo_cic_prepare()
    check_dataset_host()
    print("\n" + "#" * 66)
    print("# 集成管线演示完成")
    print("#" * 66)
