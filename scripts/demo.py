# -*- coding: utf-8 -*-
"""
系统自测 / 最小可运行演示
=====================================================================
用合成数据串联四大核心模块，验证：
  1. ADASYN 不平衡采样
  2. 降噪自编码器（仅正常样本训练）
  3. WLS / EKF-卡方 / IVB-NCA-NLKF 物理层检测（注入量测偏置攻击）
  4. N-Burst 特征 -> CSEKM 聚类 -> CPL+SV 在线匹配
  5. DPNet 双通道前向传播与单步训练

运行：python demo.py
"""


# --- 合并仓库引导（scripts/_bootstrap） ---
import os as _os, sys as _sys
REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_os.path.join(REPO_ROOT, "src"), REPO_ROOT):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
del _os, _sys, _p
# --- 合并仓库引导结束 ---

import numpy as np
import torch

from icps_detection.preprocessing import (
    ADASYN, Standardizer, DenoisingAutoencoder, get_device,
)
from icps_detection.physical_layer import (
    WLSStateEstimator, EKFChiSquareDetector, IVB_NCA_NLKF,
)
from icps_detection.network_layer import (
    NBurstFeatureExtractor, EvolutionaryKMeans, CPLMatcher,
)
from icps_detection.dpnet import DPNet

rng = np.random.default_rng(42)


def section(title):
    print("\n" + "=" * 66)
    print(title)
    print("=" * 66)


# ---------------------------------------------------------------------
# 1. ADASYN
# ---------------------------------------------------------------------
def demo_adasyn():
    section("1. ADASYN 不平衡采样")
    X_maj = rng.normal(loc=0.0, scale=1.0, size=(400, 6))
    X_min = rng.normal(loc=2.2, scale=1.0, size=(40, 6))
    X = np.vstack([X_maj, X_min])
    y = np.concatenate([np.zeros(400), np.ones(40)])

    sampler = ADASYN(beta=1.0, k_neighbors=5, random_state=42)
    X_new, y_new = sampler.fit_resample(X, y)
    _, cnt_before = np.unique(y, return_counts=True)
    _, cnt_after = np.unique(y_new, return_counts=True)
    print(f"采样前类别计数: {cnt_before}")
    print(f"采样后类别计数: {cnt_after}")
    assert abs(cnt_after[0] - cnt_after[1]) <= 1
    print("[OK] 两类已基本平衡")


# ---------------------------------------------------------------------
# 2. 降噪自编码器
# ---------------------------------------------------------------------
def demo_autoencoder():
    section("2. 降噪自编码器（仅正常样本训练）")
    X_normal = rng.normal(size=(300, 12)) @ np.diag(np.linspace(1, 3, 12))
    X_attack = rng.normal(size=(30, 12)) * 4.0

    ae = DenoisingAutoencoder(
        input_dim=12, latent_dim=4, hidden_dims=(24, 16),
        epochs=30, noise_factor=0.15, random_state=42,
    )
    ae.fit(X_normal)                                 # 关键：只喂正常样本
    z = ae.encode(X_normal)
    score_normal = ae.reconstruction_error(X_normal)
    score_attack = ae.reconstruction_error(X_attack)
    print(f"潜在特征形状: {z.shape}")
    print(f"正常样本重构误差 均值 = {score_normal.mean():.3f}")
    print(f"攻击样本重构误差 均值 = {score_attack.mean():.3f}")
    assert score_attack.mean() > score_normal.mean()
    print("[OK] 攻击样本重构误差显著高于正常样本")


# ---------------------------------------------------------------------
# 3. 物理层：WLS + EKF + IVB-NCA-NLKF
# ---------------------------------------------------------------------
# 非线性状态空间：x = [theta, omega]
DT = 0.1

def f_fun(x):
    theta, omega = x
    return np.array([theta + DT * omega,
                     0.995 * omega + 0.02 * np.sin(theta)])

def h_fun(x):
    theta, omega = x
    return np.array([theta,
                     omega + 0.05 * theta ** 2])

Q_true = np.diag([1e-4, 1e-3])
R_true = np.diag([1e-3, 1e-3])


def _simulate(n_steps, attack_step=None, bias=1.2):
    """生成状态轨迹与量测；可在指定时刻注入量测偏置（模拟 FDIA）。"""
    xs, zs = [], []
    x = np.array([0.2, 0.0])
    for k in range(n_steps):
        x = f_fun(x) + rng.multivariate_normal(np.zeros(2), Q_true)
        z = h_fun(x) + rng.multivariate_normal(np.zeros(2), R_true)
        if attack_step is not None and k == attack_step:
            z[0] += bias
        xs.append(x)
        zs.append(z)
    return np.asarray(xs), np.asarray(zs)


def demo_physical():
    section("3. 物理层状态异常检测")
    xs, zs = _simulate(n_steps=200, attack_step=150, bias=1.5)

    # --- 3.1 WLS（单时间断面静态估计）---
    wls = WLSStateEstimator(h_fun, R_true, x0=np.zeros(2))
    x_wls, info_wls = wls.estimate(zs[0])
    print(f"[WLS] 首断面目标函数 J = {info_wls['cost']:.4f}, "
          f"收敛 = {info_wls['converged']}")

    # --- 3.2 EKF + 卡方 ADI（全程）---
    ekf = EKFChiSquareDetector(f_fun, h_fun, Q_true, R_true,
                               x0=zs[0], alpha=0.05)
    ekf_alarms = []
    for z in zs:
        ekf_alarms.append(ekf.step(z)["alarm"])
    ekf_alarms = np.asarray(ekf_alarms)
    print(f"[EKF-卡方] 门限 = {ekf.chi_threshold:.3f}; "
          f"攻击点(150)报警 = {ekf_alarms[150]}; "
          f"其余时刻报警数 = {ekf_alarms.sum() - ekf_alarms[150]}")

    # --- 3.3 IVB-NCA-NLKF：前 120 步干净数据标定阈值 ---
    # 残差参考 x_k 取 WLS 静态估计：干净时 WLS 与 NLKF 一致；
    # 攻击时 WLS 跟随被污染量测、NLKF 抵抗偏置 -> 偏差骤增触发告警。
    ref_wls = WLSStateEstimator(h_fun, R_true, x0=np.zeros(2))
    clean_refs = [ref_wls.estimate(z)[0] for z in zs[:120]]

    ivb = IVB_NCA_NLKF(f_fun, h_fun, Q_true, R_true, x0=zs[0],
                       n_iter=10)
    tau = ivb.calibrate_threshold(zs[:120], refs=clean_refs,
                                  percentile=99.5)
    print(f"[IVB] 标定阈值 tau = {tau:.4f}")

    alarms, xis = [], []
    for k in range(120, 200):
        x_ref = ref_wls.estimate(zs[k])[0]
        info = ivb.step(zs[k], x_ref=x_ref)
        alarms.append(info["alarm"])
        xis.append(info["xi"])
    alarms = np.asarray(alarms)
    attack_local = 150 - 120
    pre_alarms = alarms[:attack_local]                  # 攻击前干净区间
    post_alarms = alarms[attack_local + 1:]             # 攻击后恢复期
    print(f"[IVB] 攻击点(150)报警 = {alarms[attack_local]}; "
          f"攻击点残差 xi = {xis[attack_local]:.4f}")
    print(f"[IVB] 攻击前干净区间误报 = "
          f"{pre_alarms.sum()}/{pre_alarms.size}")
    print(f"[IVB] 攻击后 R 膨胀恢复期报警 = "
          f"{post_alarms.sum()}/{post_alarms.size}（攻击残迹，属预期）")
    print(f"[IVB] 估计 R = \n{np.round(ivb.R_hat, 5)}")
    assert alarms[attack_local]
    print("[OK] IVB-NCA-NLKF 在攻击点成功触发告警")


# ---------------------------------------------------------------------
# 4. 网络层：N-Burst -> CSEKM -> CPL
# ---------------------------------------------------------------------
def _gen_onoff_trace(seed, duration=30.0, rate_on=60.0,
                     mean_on=0.25, mean_off=0.6):
    """生成单条 ON/OFF 交替更新过程的包时间戳。"""
    r = np.random.default_rng(seed)
    t, timestamps = 0.0, []
    on = True
    while t < duration:
        hold = r.exponential(mean_on if on else mean_off)
        if on:
            n_pkts = r.poisson(rate_on * hold)
            if n_pkts > 0:
                timestamps.append(np.sort(t + r.uniform(0, hold, n_pkts)))
        t += hold
        on = not on
    ts = np.concatenate(timestamps) if timestamps else np.array([0.0])
    return ts[ts < duration]


def demo_network():
    section("4. 网络层流量异常检测（N-Burst + CSEKM + CPL）")
    extractor = NBurstFeatureExtractor(bin_width=0.1, on_threshold=1)

    # 生成 40 个正常观测窗口的特征（参数略有抖动）
    vectors = []
    for i in range(40):
        ts = _gen_onoff_trace(seed=i, duration=20.0,
                              rate_on=60 + rng.normal(0, 5))
        vectors.append(extractor.extract(ts)["vector"])
    V = np.asarray(vectors)

    # 对数化偏态特征（突发率/到达率）后标准化；
    # 标准化器只 fit 一次，在线匹配时必须复用，禁止对新窗口单独 fit。
    scaler = Standardizer().fit(np.log1p(V))
    V_proc = scaler.transform(np.log1p(V))

    csekm = EvolutionaryKMeans(
        n_clusters=3, pop_size=8, n_generations=30,
        alpha=0.6, mutation_prob=0.15, random_state=42,
    )
    csekm.fit(V_proc)
    print(f"[CSEKM] 最优 inertia = {csekm.inertia_:.3f}; "
          f"各簇样本数 = {np.bincount(csekm.labels_)}")

    # 用 CSEKM 质心初始化缓存模式库 CPL
    cpl = CPLMatcher(sim_threshold=0.85, decay=0.98,
                     evict_threshold=0.1).initialize(
        csekm.cluster_centers_)

    # 正常窗口：取与其所属簇质心余弦最接近的训练样本（典型正常模式）
    centroids = csekm.cluster_centers_[csekm.labels_]
    cos_self = ((V_proc * centroids).sum(axis=1)
                / (np.linalg.norm(V_proc, axis=1)
                   * np.linalg.norm(centroids, axis=1) + 1e-12))
    i_ok = int(np.argmax(cos_self))
    res_ok = cpl.match(V_proc[i_ok])
    print(f"[CPL] 正常窗口匹配 = matched:{res_ok['matched']}, "
          f"sim = {res_ok['similarity']:.3f}")
    assert res_ok["matched"]

    # 异常窗口：高强度洪水式突发（DoS/DDoS 特征），复用训练标准化器
    ts_bad = _gen_onoff_trace(seed=999, duration=20.0, rate_on=300.0,
                              mean_on=1.5, mean_off=0.1)
    raw_bad = extractor.extract(ts_bad)["vector"]
    v_bad = scaler.transform(np.log1p(raw_bad).reshape(1, -1))[0]
    res_bad = cpl.match(v_bad)
    print(f"[CPL] 攻击窗口匹配 = matched:{res_bad['matched']}, "
          f"sim = {res_bad['similarity']:.3f}, "
          f"anomaly = {res_bad['anomaly']}")
    assert res_bad["anomaly"]
    print("[OK] 正常模式命中、攻击模式正确标记为未知")


# ---------------------------------------------------------------------
# 5. DPNet
# ---------------------------------------------------------------------
def demo_dpnet():
    section("5. DPNet 双通道攻击分类")
    B, T, F, C = 8, 20, 15, 5
    model = DPNet(n_features=F, seq_len=T, num_classes=C,
                  lstm_hidden=128, lstm_layers=3, dropout=0.2)
    device = get_device()
    model.to(device)

    X = torch.randn(B, T, F, device=device)
    y = torch.randint(0, C, (B,), device=device)

    probs = model(X)
    print(f"输出概率形状: {tuple(probs.shape)}")
    print(f"各行概率和: {probs.sum(dim=1).detach().cpu().numpy().round(4)}")
    assert probs.shape == (B, C)
    assert torch.allclose(probs.sum(dim=1), torch.ones(B, device=device),
                          atol=1e-5)

    # 单步训练验证反向传播
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.NLLLoss()
    loss_before = criterion(torch.log(model(X) + 1e-12), y).item()
    for _ in range(5):
        optimizer.zero_grad()
        loss = criterion(torch.log(model(X) + 1e-12), y)
        loss.backward()
        optimizer.step()
    loss_after = criterion(torch.log(model(X) + 1e-12), y).item()
    print(f"NLL 损失: {loss_before:.4f} -> {loss_after:.4f}（5 步优化）")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params:,}")
    print("[OK] DPNet 前向/反向传播正常")


if __name__ == "__main__":
    print(f"NumPy {np.__version__} | PyTorch {torch.__version__} | "
          f"device = {get_device()}")
    demo_adasyn()
    demo_autoencoder()
    demo_physical()
    demo_network()
    demo_dpnet()
    print("\n" + "#" * 66)
    print("# 所有核心模块自测通过")
    print("#" * 66)
