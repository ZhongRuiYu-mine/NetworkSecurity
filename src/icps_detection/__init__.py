"""
ICPS 混合入侵检测系统核心包
================================
融合物理层状态估计 (WLS/EKF/IVB-NCA-NLKF) 与网络层深度学习 (CSEKM/DPNet)。

模块一览
--------
- physical_layer    : 物理层（WLS + EKF-卡方 ADI、IVB-NCA-NLKF、
                      Sage-Husa 自适应 EKF）
- preprocessing     : ADASYN 不平衡采样 + Autoencoder 降维去噪
- network_layer     : N-Burst 特征、CSEKM 进化 K-Means、CPL+SV 余弦匹配
- dpnet             : DPNet 双通道 (CNN + BiLSTM) 攻击分类
- datasets          : CIC-IDS2017 下载/加载/清洗/少数类过采样
- pandapower_model  : pandapower IEEE 总线 / 直流电机物理模型 (f,h)
"""

__all__ = [
    "physical_layer", "preprocessing", "network_layer", "dpnet",
    "datasets", "pandapower_model",
]
__version__ = "1.1.0"
