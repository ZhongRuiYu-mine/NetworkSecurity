# -*- coding: utf-8 -*-
"""仓库路径常量 —— 四包合并后统一的落点约定。

合并前每个包都自带一套相对路径假设（`ae_repro/_cache`、`<repo>/chethuhn/...`、
`data/cicids_sample.npz`），合并后用本模块收敛成一处：

    <repo>/data/raw/cicids2017/         原始 CICIDS2017 CSV（只读，gitignore）
    <repo>/data/cicids_sample.npz       ★ 规范数据包（协议 v1，见 PROTOCOL.md）
    <repo>/data/cache/                  历史/对照用数据包（gitignore）

可用环境变量 `NETSEC_DATA_DIR` 整体重定向 `data/`（CI 或换机器时用）。
"""
from __future__ import annotations

import os

_HERE = os.path.dirname(os.path.abspath(__file__))          # <repo>/src/ae_repro
REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))          # <repo>

DATA_DIR = os.environ.get("NETSEC_DATA_DIR", os.path.join(REPO_ROOT, "data"))
RAW_CSV_DIR = os.path.join(DATA_DIR, "raw", "cicids2017")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")

#: 规范数据包（协议 v1：评估集排除检测器拟合用 Monday 行）
BUNDLE = os.path.join(DATA_DIR, "cicids_sample.npz")
#: 修复前那一版（测试集含检测器训练数据），仅用于新旧口径对照
OLD_BUNDLE = os.path.join(CACHE_DIR, "cicids_sample.old_protocol.npz")

__all__ = ["REPO_ROOT", "DATA_DIR", "RAW_CSV_DIR", "CACHE_DIR", "RESULTS_DIR",
           "BUNDLE", "OLD_BUNDLE"]
