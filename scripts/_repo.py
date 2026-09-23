# -*- coding: utf-8 -*-
"""合并后仓库的路径常量（scripts/ 下的脚本统一从这里取路径）。"""
from __future__ import annotations

import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("NETSEC_DATA_DIR", os.path.join(REPO, "data"))
RAW_CSV = os.path.join(DATA, "raw", "cicids2017")
BUNDLE = os.path.join(DATA, "cicids_sample.npz")
ARTIFACTS = os.path.join(REPO, "artifacts")
RESULTS = os.path.join(REPO, "results")

__all__ = ["REPO", "DATA", "RAW_CSV", "BUNDLE", "ARTIFACTS", "RESULTS"]
