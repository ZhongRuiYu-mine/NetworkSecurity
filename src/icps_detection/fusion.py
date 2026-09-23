# -*- coding: utf-8 -*-
"""
两层融合决策模块
=====================================================================
把"物理层状态估计"与"网络层深度学习"对同一 ICPS 事件的判决组合为
最终结论。包含两个类：

1. NetworkProbe
   加载已训练的自包含 checkpoint（dpnet_cicids.pt，含模型权重 +
   scaler mu/sigma + 类别 + 序列形状），对 77 维原始流量特征做
   一致的预处理与 DPNet 推理，返回攻击类别、概率与网络层告警。

2. FusionDetector
   输入物理层检测器 step() 的结果（卡方 ADI/告警）与 NetworkProbe
   的结果（攻击概率/类别），通过决策表 + 加权分数给出最终判决。

   ----------------------------------------------------------------
   决策表（两层互补）
   ----------------------------------------------------------------
     物理告警  网络告警 -> 最终判决                     证据来源
     ----------------------------------------------------------
       否       否      BENIGN                          NONE
       是       否      FDIA（隐蔽虚假数据注入）         PHYS
       否       是      <网络攻击类，如 DoS/DDoS>        NET
       是       是      <网络类>+FDIA（协同复合攻击）    BOTH

   核心价值：FDIA 只篡改电网量测、网络流量完全正常，网络层 DPNet
   无法发现，只有物理层卡方检验能检出 —— 融合因此能覆盖任意单层
   都看不见的攻击面。

   ----------------------------------------------------------------
   融合模式 mode
   ----------------------------------------------------------------
     "weighted" : fused = w_phys*p_phys + w_net*p_net，超阈值报警；
     "or"       : 任一层报警即报警（高检测率，误报略高）；
     "and"      : 两层同时报警（高精度，漏报高，一般不用）。
"""

import numpy as np

try:
    import torch
except ImportError as exc:
    raise ImportError("融合模块需要 PyTorch") from exc

from .dpnet import DPNet
from .preprocessing import get_device


# =====================================================================
# 1. 网络层推理探针
# =====================================================================
class NetworkProbe:
    """
    DPNet 网络层推理封装。

    checkpoint 必须包含：
        model_state, classes, seq_T, seq_F,
        scaler_mu, scaler_sigma, feature_names
    """

    def __init__(self, ckpt_path="dpnet_cicids.pt", device=None):
        self.device = device or get_device()
        ck = torch.load(ckpt_path, map_location=self.device,
                        weights_only=False)

        self.classes = list(ck["classes"])
        self.T = int(ck["seq_T"])
        self.F = int(ck["seq_F"])
        self.feature_names = list(ck["feature_names"])
        self.mu = np.asarray(ck["scaler_mu"], dtype=np.float32)
        self.sigma = np.asarray(ck["scaler_sigma"], dtype=np.float32)

        self.model = DPNet(n_features=self.F, seq_len=self.T,
                           num_classes=len(self.classes))
        self.model.load_state_dict(ck["model_state"])
        self.model.to(self.device).eval()

        self.benign_idx = self.classes.index("BENIGN")

    @torch.no_grad()
    def predict(self, x):
        """
        对原始（未标准化）流量特征推理。

        Parameters
        ----------
        x : (77,) 或 (B,77)，特征顺序必须与 self.feature_names 一致。

        Returns
        -------
        dict : label, attack_prob, alarm, probs, class_idx
        """
        x = np.asarray(x, dtype=np.float32)
        one = x.ndim == 1
        if one:
            x = x[None, :]

        # 与训练一致的 Z-Score（常量列 sigma 已在保存时置 1）
        x_std = (x - self.mu) / self.sigma
        x_std = x_std.reshape(-1, self.T, self.F)
        xt = torch.from_numpy(x_std.astype(np.float32)).to(self.device)

        probs = self.model(xt).cpu().numpy()
        idx = probs.argmax(axis=1)
        attack_prob = 1.0 - probs[:, self.benign_idx]

        if one:
            i = int(idx[0])
            return {"label": self.classes[i], "class_idx": i,
                    "attack_prob": float(attack_prob[0]),
                    "alarm": bool(i != self.benign_idx),
                    "probs": probs[0]}
        return {"label": [self.classes[i] for i in idx],
                "class_idx": idx, "attack_prob": attack_prob,
                "alarm": idx != self.benign_idx, "probs": probs}


# =====================================================================
# 2. 物理层异常分：卡方 ADI -> [0,1)
# =====================================================================
def physical_score(phys_info, chi_threshold=None):
    r"""
    把卡方新息检验结果映射为 [0,1) 的物理异常概率。

        r = ADI / chi_threshold     # r=1 恰在卡方门限
        p = r / (1 + r)

    - r<1（未超门限） -> p<0.5；
    - r=1             -> p=0.5；
    - 攻击时 r>>1     -> p 趋近 1。

    因此加权分数以 0.5 为分界，与"卡方是否报警"在物理含义上自洽。
    """
    adi = float(phys_info.get("ADI", 0.0))
    thr = float(chi_threshold if chi_threshold is not None
                else phys_info.get("threshold", 1.0))
    r = adi / max(thr, 1e-12)
    return float(r / (1.0 + r))


# =====================================================================
# 3. 两层融合检测器
# =====================================================================
class FusionDetector:
    """
    组合物理层检测器与网络层探针，输出最终判决。

    Parameters
    ----------
    phys_detector : 物理层对象（EKFChiSquareDetector / SageHusaAEKF
                    等），需提供 chi_threshold；由调用方先 fit/预热。
    network_probe : NetworkProbe 实例。
    w_phys/w_net   : 加权模式下两层权重（默认各 0.5）。
    mode           : "weighted" / "or" / "and"。
    score_threshold: weighted 模式的告警分数门限（默认 0.5）。
    """

    def __init__(self, phys_detector, network_probe,
                 w_phys=0.5, w_net=0.5, mode="weighted",
                 score_threshold=0.5):
        self.phys = phys_detector
        self.net = network_probe
        self.w_phys, self.w_net = w_phys, w_net
        self.mode = mode
        self.score_threshold = score_threshold

    # 两层信息 -> 融合判决
    def fuse(self, phys_info, net_info):
        """
        Parameters
        ----------
        phys_info : 物理层 step() 返回（含 alarm、ADI）。
        net_info  : NetworkProbe.predict() 返回（含 alarm、label、
                    attack_prob）。

        Returns
        -------
        dict : label, alarm, source, fused_score, p_phys, p_net,
               phys_alarm, net_alarm, net_label
        """
        chi_thr = getattr(self.phys, "chi_threshold",
                          phys_info.get("threshold", 1.0))
        p_phys = physical_score(phys_info, chi_thr)
        p_net = float(net_info["attack_prob"])

        phys_alarm = bool(phys_info.get("alarm", False))
        net_alarm = bool(net_info["alarm"])

        # ---- 融合告警 ----
        fused = self.w_phys * p_phys + self.w_net * p_net
        # 关键：任一层的"硬报警"都是确凿证据，不允许被加权分数稀释
        # （否则会出现物理强报警但因网络无声、score 略低于门限而漏报）。
        hard_alarm = phys_alarm or net_alarm
        if self.mode == "and":
            alarm = phys_alarm and net_alarm
        elif self.mode == "or":
            alarm = hard_alarm
        else:  # weighted：保留单层硬证据；软分数仅用于捕捉"两层都没硬报
            # 但综合分可疑"的情况
            alarm = bool(hard_alarm or fused >= self.score_threshold)

        # ---- 决策表：最终类别与证据来源 ----
        net_label = net_info["label"]
        if phys_alarm and net_alarm:
            label = f"{net_label}+FDIA"
            source = "BOTH"
        elif phys_alarm and not net_alarm:
            label = "FDIA"
            source = "PHYS"
        elif net_alarm and not phys_alarm:
            label = net_label
            source = "NET"
        else:
            label = "BENIGN"
            source = "NONE"

        return {"label": label, "alarm": alarm, "source": source,
                "fused_score": float(fused),
                "p_phys": p_phys, "p_net": p_net,
                "phys_alarm": phys_alarm, "net_alarm": net_alarm,
                "net_label": net_label}
