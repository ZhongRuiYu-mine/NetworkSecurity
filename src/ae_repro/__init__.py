# -*- coding: utf-8 -*-
"""
ae_repro —— 论文自编码器检测机制的复现包（CICIDS2017 / chethuhn 数据集）

本包刻意**不依赖**平台仓库内的任何模块，只依赖 numpy / torch / sklearn(pandas)。
这样组员既可以把单个文件拷进仓库，也可以直接在这里跑完整复现。

模块
----
ae_core.py           纯 numpy 的自编码器核心（训练 / 打分 / 嵌入 / 阈值 / 落盘）
ae_detector.py       ★ 贴进仓库用的 DetectorBase 实现（`from .base import ...`）
_contract.py         平台 base.py 的逐字副本（独立运行自检用，勿改）
data_prep.py         复现平台 data_loader 的 CICIDS2017 数据协议
experiments.py       复现实验 + 消融 + 报告 JSON + 图
train_detector.py    训练并落盘检测器（给组员直接 load 用）
verify_contract.py   平台契约自检（不需要平台仓库即可跑）

对应论文
--------
主复现：AUTO.pdf —— El Emary et al., "Anomaly Detection in Blockchain-Based
        Metaverse Transactions Using Hybrid Autoencoder and Isolation Forest
        Models...", Int. J. Res. Metaverse, vol.3 no.1, pp.46-63, 2026.
        （本包只复现其 **第 1 阶段 Autoencoder**，不实现 IF 与加权融合）
增强项：Autoencoder-Based_Anomaly_Detection.pdf —— Lozano-Paredes et al.,
        "Explainable Autoencoder-Based Anomaly Detection in IEC 61850 GOOSE
        Networks", IEEE Access, vol.14, 2026.
        （仅借鉴其 EVT/GPD 阈值标定与逐特征重构误差归因）
"""

from .ae_core import (
    AEConfig,
    AutoencoderCore,
    EVTThreshold,
    binary_metrics,
    best_f1_threshold,
    fit_evt_threshold,
)

__all__ = [
    "AEConfig",
    "AutoencoderCore",
    "EVTThreshold",
    "fit_evt_threshold",
    "binary_metrics",
    "best_f1_threshold",
]
__version__ = "1.0.0"


def __getattr__(name: str):  # PEP 562：AePaperDetector 延迟导入，避免必须先有 torch
    if name == "AePaperDetector":
        from .ae_detector import AePaperDetector

        return AePaperDetector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
