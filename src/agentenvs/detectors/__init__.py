"""防守方检测器插件包：IF / Kalman / Autoencoder 三种机制 + 自定义扩展点。

合并后新增：
  * `ae_paper` / `ae_cicids` —— 论文复现版自编码器（ae_repro 包），
    逐特征标准化残差 + EVT/GPD 阈值标定，`embed_dim=4`。详见 docs/PROTOCOL.md。
"""
from .base import (
    SIGNAL_FLAG,
    SIGNAL_SCORE,
    SIGNAL_TAIL,
    SIGNAL_TRUST,
    DETECTOR_REGISTRY,
    DetectorBase,
    DetectorOutput,
    coerce_detector,
    list_detectors,
    make_detector,
    register_detector,
)
from .autoencoder_detector import AutoencoderDetector
from .isolation_forest_detector import IsolationForestDetector
from .kalman_detector import KalmanFilterDetector
from .null_detector import NullDetector
from .ae_paper_detector import AePaperDetector
# 模板示例（注册了 zscore_demo / stateful_demo，并把 WrapExistingModel 暴露出来）
from .templates import StatefulDemoDetector, WrapExistingModel, ZScoreDemoDetector

__all__ = [
    "DetectorBase",
    "DetectorOutput",
    "DETECTOR_REGISTRY",
    "make_detector",
    "coerce_detector",
    "register_detector",
    "list_detectors",
    "IsolationForestDetector",
    "KalmanFilterDetector",
    "AutoencoderDetector",
    "AePaperDetector",
    "NullDetector",
    "ZScoreDemoDetector",
    "StatefulDemoDetector",
    "WrapExistingModel",
    "SIGNAL_SCORE",
    "SIGNAL_FLAG",
    "SIGNAL_TRUST",
    "SIGNAL_TAIL",
]
