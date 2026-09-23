# -*- coding: utf-8 -*-
"""
data_prep.py —— 复现平台 `agentenvs/data_loader.py` 的数据协议（CICIDS2017 / chethuhn）。

对齐点（逐条照抄平台默认值，保证结果能和 IF / Kalman 横向对比）
--------------------------------------------------------------
| 平台 `DataConfig` / `default_file_specs`            | 本文件 |
|-----------------------------------------------------|--------|
| 良性来源 = Monday，`nrows=60000`                     | `rows_normal=60000` |
| 带标签文件 = 周二~周五 8 个文件，`nrows=20000`        | `default_file_specs()` 同序同种子 |
| `normal_ratio_in_train=0.30`（良性下采样）            | 同 |
| `test_size=0.2`，**按类别分层**                       | 同 |
| `min_attack_rows=200`，少于此数的攻击类并入 `Other`   | 同 |
| `class_preset="default"` = BENIGN + DDoS/PortScan/Bot/DoS/WebAttack/Other | 同 |
| 归一化 = **只用良性训练集的** Min–Max 到 [0,1]（论文 Eq.1） | 同 |
| 特征列 = CSV 全部列去掉 `Label`（本数据没有 Timestamp 列，共 77 维） | 同 |
| `encoding="latin-1"`，表头前导空格保留                      | 同 |

产物 `.npz` 与平台 `save_bundle()` 落盘格式一致，因此既可以用本包跑实验，
也可以用 `agentenvs.data_loader.load_bundle()` 直接读进平台环境。

用法
----
    python -m ae_repro.data_prep --csv-dir "D:/.../versions/1" --out _cache/cicids_sample.npz
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

BENIGN = "BENIGN"
OTHER_CLASS = "Other"
DEFAULT_ATTACK_CLASSES = ["DDoS", "PortScan", "Bot", "DoS", "WebAttack", OTHER_CLASS]

#: 默认 CSV 目录 / 缓存目录 —— 合并后用仓库统一路径（src/ae_repro/paths.py）
from .paths import BUNDLE as _BUNDLE_OUT, CACHE_DIR as _CACHE_DIR, RAW_CSV_DIR as _RAW_CSV_DIR

DEFAULT_CSV_DIR = _RAW_CSV_DIR
DEFAULT_CACHE = _CACHE_DIR

# ---------------------------------------------------------------- 标签归一化（抄平台）
_LABEL_ALIASES: Dict[str, str] = {
    "benign": BENIGN,
    "ddos": "DDoS",
    "dos hulk": "DoS",
    "dos goldeneye": "DoS",
    "dos slowloris": "DoS",
    "dos slowhttptest": "DoS",
    "heartbleed": "Heartbleed",
    "portscan": "PortScan",
    "bot": "Bot",
    "ftp-patator": "FTP-Patator",
    "ssh-patator": "SSH-Patator",
    "infiltration": "Infiltration",
    "web attack - brute force": "WebAttack",
    "web attack - xss": "WebAttack",
    "web attack - sql injection": "WebAttack",
}
_WEB_ATTACK_RE = re.compile(r"web\s*attack", re.IGNORECASE)
_DOS_RE = re.compile(r"^dos[\s\-_]", re.IGNORECASE)


def _clean(raw: str) -> str:
    s = str(raw)
    s = s.replace("\ufeff", "").replace("\ufffd", "-").replace("ï¿½", "-")
    s = "".join(ch if (ch.isascii() and ch.isprintable()) else "-" for ch in s)
    s = re.sub(r"[\s\-_]*\-[\s\-_]*", " - ", s)
    s = re.sub(r"\s+", " ", s).strip(" -")
    return s


def normalize_label(raw: str) -> str:
    s = _clean(raw)
    key = s.lower()
    if key in _LABEL_ALIASES:
        return _LABEL_ALIASES[key]
    if _WEB_ATTACK_RE.search(key):
        return "WebAttack"
    if _DOS_RE.match(key) or key.startswith("dos"):
        return "DoS"
    if key.startswith("ddos"):
        return "DDoS"
    if "patator" in key:
        return "FTP-Patator" if key.startswith("ftp") else "SSH-Patator"
    return s.replace(" ", "") or "UNKNOWN"


# ---------------------------------------------------------------- 配置
@dataclass
class FileSpec:
    filename: str
    nrows: Optional[int] = 20000
    seed: int = 42
    encoding: str = "latin-1"


@dataclass
class CicidsConfig:
    csv_dir: str = DEFAULT_CSV_DIR
    out_path: str = os.path.join(DEFAULT_CACHE, "cicids_sample.npz")
    rows_normal: int = 60000
    rows_per_file: int = 20000
    normal_ratio_in_train: float = 0.30
    test_size: float = 0.20
    min_attack_rows: int = 200
    seed: int = 42
    class_names: List[str] = field(default_factory=lambda: [BENIGN] + list(DEFAULT_ATTACK_CLASSES))

    eval_exclude_normal_source: bool = True
    """**关键开关**：是否把检测器的拟合数据（Monday 良性）排除在评估集之外。

    `normal_source()` 会从 Monday 抽 `rows_normal` 行做检测器的训练集；
    `file_specs()` 里的 Monday 项会抽到**同一批行**（同名文件 + 同 nrows + 同 seed），
    于是这批行会同时出现在 `X_normal_train` 和带标签池里。若不排除，测试集会有
    约 1/3 的样本是检测器**逐位见过**的训练数据 → 评估变成自评（FPR 被低估、
    攻击占比被稀释）。True = 评估集只用非 Monday 行（正确做法）；
    False = 复现修复前的旧口径（仅供对照，写论文不要用）。
    """

    def file_specs(self) -> List[FileSpec]:
        n, s = self.rows_per_file, self.seed
        return [
            FileSpec("Monday-WorkingHours.pcap_ISCX.csv", self.rows_normal, s),
            FileSpec("Tuesday-WorkingHours.pcap_ISCX.csv", n, s + 1),
            FileSpec("Wednesday-workingHours.pcap_ISCX.csv", n, s + 2),
            FileSpec("Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv", n, s + 3),
            FileSpec("Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv", n // 4, s + 4),
            FileSpec("Friday-WorkingHours-Morning.pcap_ISCX.csv", n, s + 5),
            FileSpec("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv", n, s + 6),
            FileSpec("Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv", n, s + 7),
        ]
        # 说明：Monday 的 FileSpec 与 normal_source() 完全一致（同名文件/同 nrows/同 seed），
        # 抽到的是同一批行。这不是笔误：它是平台 data_loader 默认「用全部 8 个文件做带标签池」
        # 的复制。检测器**只在 X_normal_train 上训练**，所以评估集必须排除这批行，
        # 由 `eval_exclude_normal_source` 在切分阶段完成，而不是靠删掉这个 FileSpec。

    def normal_source(self) -> List[FileSpec]:
        return [FileSpec("Monday-WorkingHours.pcap_ISCX.csv", self.rows_normal, self.seed)]


# ---------------------------------------------------------------- CSV 读取（抄平台）
def _read_csv_sampled(path: str, spec: FileSpec, feature_cols: Optional[Sequence[str]] = None):
    import pandas as pd

    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到数据文件：{path}")

    header = pd.read_csv(path, nrows=0, encoding=spec.encoding)
    raw_cols = list(header.columns)
    all_cols = [str(c).strip() for c in raw_cols]
    header.columns = all_cols
    raw_by_clean = {str(c).strip(): c for c in raw_cols}

    want = set(feature_cols) if feature_cols is not None else None
    clean_use = [c for c in all_cols if want is None or c in want]
    if "Label" in all_cols:
        clean_use = clean_use + ["Label"]
    clean_use = list(dict.fromkeys(clean_use))
    usecols = [raw_by_clean[c] for c in clean_use if c in raw_by_clean]

    if spec.nrows is not None and spec.nrows > 0:
        bytes_per_row = max(len(usecols) * 9.0, 60.0)
        est_total = max(int(os.path.getsize(path) / bytes_per_row), 1)
        est_total = min(est_total, 3_000_000)
    else:
        est_total = None

    rng = np.random.default_rng(spec.seed)
    chunks: List = []
    kept = 0
    seen = 0
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=20000,
                             low_memory=False, encoding=spec.encoding):
        chunk.columns = [c.strip() for c in chunk.columns]
        n = len(chunk)
        if spec.nrows is None:
            chunks.append(chunk)
            kept += n
            seen += n
            continue
        remaining_quota = max(spec.nrows - kept, 0)
        remaining_rows = max((est_total or (seen + n)) - seen, n)
        p = min(1.0, remaining_quota / remaining_rows)
        if p > 0:
            mask = rng.random(n) < p
            if mask.any():
                sub = chunk.loc[mask]
                if kept + len(sub) > spec.nrows:
                    sub = sub.sample(n=spec.nrows - kept, random_state=spec.seed)
                chunks.append(sub)
                kept += len(sub)
        seen += n
        if kept >= spec.nrows:
            break

    if not chunks:
        raise RuntimeError(f"{path} 没有采样到任何行，请调大 nrows")
    import pandas as pd

    df = pd.concat(chunks, ignore_index=True)
    if spec.nrows is not None and len(df) > spec.nrows:
        df = df.sample(n=spec.nrows, random_state=spec.seed).reset_index(drop=True)
    return df, all_cols


def _sanitize_features(df, feature_cols: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    import pandas as pd

    X = df.loc[:, feature_cols].copy()
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    valid = ~X.isna().all(axis=1).to_numpy()
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
    return X.to_numpy(dtype=np.float32), valid


def _minmax(X_normal: np.ndarray, *others: np.ndarray):
    lo = np.min(X_normal, axis=0).astype(np.float32)
    hi = np.max(X_normal, axis=0).astype(np.float32)
    rng = np.where((hi - lo) < 1e-12, 1.0, hi - lo).astype(np.float32)

    def _scale(A):
        if A is None:
            return None
        return np.clip((np.asarray(A, dtype=np.float32) - lo) / rng, 0.0, 1.0).astype(np.float32)

    return _scale(X_normal), [_scale(o) for o in others], (lo, rng)


def _split_indices(y: np.ndarray, test_size: float, rng,
                   eligible: Optional[np.ndarray] = None) -> np.ndarray:
    """返回测试集索引（按类别分层，与平台 `_split` 逐行等价）。

    `eligible`（可选 bool 掩码）标记**可以进测试集**的行；False 的行一律留在
    训练集。用于把检测器的拟合数据排除在评估之外，见
    `CicidsConfig.eval_exclude_normal_source`。
    """
    n_eligible = len(y) if eligible is None else int(np.count_nonzero(eligible))
    n_test = int(round(n_eligible * test_size))
    test_idx: List[int] = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        if eligible is not None:
            idx = idx[eligible[idx]]
        if idx.size == 0:
            continue
        rng.shuffle(idx)
        k = int(round(len(idx) * test_size))
        test_idx.extend(idx[:k].tolist())
    test_idx = np.array(sorted(test_idx), dtype=np.int64)
    if len(test_idx) > n_test:
        test_idx = test_idx[:n_test]
    return test_idx


def _split(X, y, test_size: float, rng):
    test_idx = _split_indices(y, test_size, rng)
    mask = np.ones(len(y), dtype=bool)
    mask[test_idx] = False
    return X[mask], y[mask], X[test_idx], y[test_idx]


def _balance_normal(X, y, max_ratio: float, rng):
    n_attack = int((y != 0).sum())
    if n_attack == 0:
        return X, y
    benign = np.where(y == 0)[0]
    limit = int(n_attack * max_ratio / max(1e-6, 1.0 - max_ratio))
    if len(benign) <= limit:
        return X, y
    keep = np.concatenate([rng.choice(benign, size=limit, replace=False), np.where(y != 0)[0]])
    keep.sort()
    return X[keep], y[keep]


_DAY_OF_FILE = {
    "Monday-WorkingHours.pcap_ISCX.csv": "Monday",
    "Tuesday-WorkingHours.pcap_ISCX.csv": "Tuesday",
    "Wednesday-workingHours.pcap_ISCX.csv": "Wednesday",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv": "Thursday",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv": "Thursday",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv": "Friday",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv": "Friday",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv": "Friday",
}


def _merge_rare(y, class_names, min_rows: int, other_name: str = OTHER_CLASS):
    cnt = np.bincount(y, minlength=len(class_names))
    keep = [0]
    drop: Dict[str, int] = {}
    for i in range(1, len(class_names)):
        if cnt[i] >= min_rows or class_names[i] == other_name:
            keep.append(i)
        else:
            drop[class_names[i]] = int(cnt[i])
    if not drop:
        return y, list(class_names), {}
    other_local = keep.index(class_names.index(other_name))
    remap = np.zeros(len(class_names), dtype=np.int64)
    for new_i, old_i in enumerate(keep):
        remap[old_i] = new_i
    y_new = y.copy()
    y_new[np.isin(y, [class_names.index(k) for k in drop])] = other_local
    y_new = remap[y_new]
    return y_new, [class_names[i] for i in keep], drop


# ---------------------------------------------------------------- 主流程
def build(cfg: Optional[CicidsConfig] = None, verbose: bool = True) -> Dict[str, object]:
    """复现平台数据协议，返回 dict（形状同 TrafficDataset）。"""
    cfg = cfg or CicidsConfig()
    t0 = time.perf_counter()
    class_names = list(cfg.class_names)

    # ---------- 1. 良性流量（检测器训练 + 归一化统计） ----------
    normal_frames, all_cols = [], None
    for fs in cfg.normal_source():
        path = os.path.join(cfg.csv_dir, fs.filename)
        df, cols = _read_csv_sampled(path, fs)
        if all_cols is None:
            all_cols = cols
        lab = df["Label"].map(normalize_label)
        normal_frames.append(df.loc[lab == BENIGN])
    import pandas as pd

    normal_df = pd.concat(normal_frames, ignore_index=True)
    feature_cols = [c for c in (all_cols or []) if c != "Label"]

    # ---------- 2. 带标签流量 ----------
    frames, label_map = [], {}
    day_frames: List[np.ndarray] = []
    for fs in cfg.file_specs():
        path = os.path.join(cfg.csv_dir, fs.filename)
        df, _ = _read_csv_sampled(path, fs, feature_cols=feature_cols)
        raw = df["Label"].astype(str)
        canon = raw.map(normalize_label)
        label_map.update(dict(zip(raw.head(5000).tolist(), canon.head(5000).tolist())))
        df = df.copy()
        df["_canon"] = canon.values
        frames.append(df)
        day_frames.append(np.full(len(df), _DAY_OF_FILE.get(fs.filename, fs.filename), dtype=object))
    labeled = pd.concat(frames, ignore_index=True)
    day_all = np.concatenate(day_frames)

    y_canon = labeled["_canon"].astype(str).to_numpy()
    X_raw, valid = _sanitize_features(labeled, feature_cols)
    y_canon = y_canon[valid]
    day_all = day_all[valid]
    Xn_raw, n_valid = _sanitize_features(normal_df, feature_cols)
    Xn_raw = Xn_raw[n_valid]

    # ---------- 3. 只用良性统计做 Min–Max（论文 Eq.1） ----------
    Xn, (Xall,), (lo, rng_) = _minmax(Xn_raw, X_raw)
    Xall = np.asarray(Xall, dtype=np.float32)

    # ---------- 4. 类别索引 + 稀有类并入 Other ----------
    spec_index = {n: i for i, n in enumerate(class_names)}

    def _index_of(c: str) -> int:
        if c == BENIGN:
            return 0
        if c in spec_index:
            return spec_index[c]
        return spec_index.get(OTHER_CLASS, len(class_names) - 1)

    y = np.array([_index_of(c) for c in y_canon], dtype=np.int64)
    y, class_names, merge_info = _merge_rare(y, class_names, cfg.min_attack_rows)

    # ---------- 5. 分层切分 + 良性下采样 ----------
    if len(Xall) != len(day_all):
        raise RuntimeError(f"内部错误：X({len(Xall)}) 与 day({len(day_all)}) 长度不一致")
    rng = np.random.default_rng(cfg.seed)
    # 检测器的拟合数据（Monday 良性）不许进评估集：它们既在 X_normal_train 里，
    # 又被 file_specs() 的 Monday 项抽进了带标签池，不排除就是自己评自己。
    eligible = np.ones(len(y), dtype=bool)
    if cfg.eval_exclude_normal_source:
        eligible &= np.asarray([str(d) != "Monday" for d in day_all])
    test_idx = _split_indices(y, cfg.test_size, rng, eligible=eligible)
    train_mask = np.ones(len(y), dtype=bool)
    train_mask[test_idx] = False
    X_tr, y_tr = Xall[train_mask], y[train_mask]
    X_te, y_te = Xall[test_idx], y[test_idx]
    day_test = day_all[test_idx]
    X_tr, y_tr = _balance_normal(X_tr, y_tr, cfg.normal_ratio_in_train, rng)

    out = {
        "X_normal_train": np.asarray(Xn, dtype=np.float32),
        "X_train": X_tr, "y_train": y_tr,
        "X_test": X_te, "y_test": y_te,
        "day_test": np.asarray(day_test, dtype=object),
        "feature_names": np.array(feature_cols, dtype=object),
        "class_names": np.array(class_names, dtype=object),
        "source": "cicids2017",
    }

    # ---------- 6. 落盘 ----------
    os.makedirs(os.path.dirname(os.path.abspath(cfg.out_path)), exist_ok=True)
    np.savez_compressed(cfg.out_path, **out)

    def _dist(yv: np.ndarray) -> Dict[str, int]:
        c = np.bincount(yv, minlength=len(class_names))
        return {class_names[i]: int(c[i]) for i in range(len(class_names))}

    meta = {
        "csv_dir": cfg.csv_dir,
        "out_path": cfg.out_path,
        "n_features": int(len(feature_cols)),
        "feature_names": list(feature_cols),
        "class_names": list(class_names),
        "merged_into_other": merge_info,
        "shapes": {
            "X_normal_train": list(out["X_normal_train"].shape),
            "X_train": list(out["X_train"].shape),
            "X_test": list(out["X_test"].shape),
        },
        "distribution_train": _dist(y_tr),
        "distribution_test": _dist(y_te),
        "distribution_test_by_day": {
            d: int(v) for d, v in zip(*np.unique(np.asarray(day_test, dtype=str),
                                                 return_counts=True))
        },
        "distribution_normal_source": {"BENIGN": int(len(Xn))},
        "normalization": "min-max，统计量仅来自 Monday 良性流量（对齐平台与论文 Eq.1）",
        "config": {
            "rows_normal": cfg.rows_normal,
            "rows_per_file": cfg.rows_per_file,
            "normal_ratio_in_train": cfg.normal_ratio_in_train,
            "test_size": cfg.test_size,
            "min_attack_rows": cfg.min_attack_rows,
            "seed": cfg.seed,
            "eval_exclude_normal_source": cfg.eval_exclude_normal_source,
        },
        "eval_note": (
            "评估集已排除检测器拟合数据（Monday 良性），避免自评"
            if cfg.eval_exclude_normal_source else
            "⚠ 旧口径：Monday 行允许进测试集，其中绝大部分是检测器的训练数据"
        ),
        "label_map_sample": dict(list(label_map.items())[:40]),
        "seconds": float(time.perf_counter() - t0),
    }
    with open(os.path.splitext(cfg.out_path)[0] + ".meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)

    if verbose:
        print(f"[data] 特征维数        : {len(feature_cols)}")
        print(f"[data] 良性训练集      : {out['X_normal_train'].shape}")
        print(f"[data] 训练/测试       : {out['X_train'].shape} / {out['X_test'].shape}")
        print(f"[data] 训练集类别分布  : {meta['distribution_train']}")
        print(f"[data] 测试集类别分布  : {meta['distribution_test']}")
        print(f"[data] 测试集按天      : {meta['distribution_test_by_day']}")
        print(f"[data] 评估口径        : {meta['eval_note']}")
        print(f"[data] 稀有类归并      : {merge_info}")
        print(f"[data] 落盘            : {cfg.out_path}")
        print(f"[data] 用时            : {meta['seconds']:.1f}s")
    return out


def load_npz(path: str) -> Dict[str, object]:
    """读回落盘结果（返回 dict，字段名与 TrafficDataset 对齐）。"""
    blob = np.load(path, allow_pickle=True)
    return {k: blob[k] for k in blob.files}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="复现平台数据协议：CICIDS2017 -> .npz")
    ap.add_argument("--csv-dir", default=DEFAULT_CSV_DIR, help="CICIDS2017 原始 CSV 目录")
    ap.add_argument("--out", default=_BUNDLE_OUT,
                    help="默认写到仓库规范数据包 data/cicids_sample.npz（协议 v1）")
    ap.add_argument("--rows-normal", type=int, default=60000)
    ap.add_argument("--rows-per-file", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--keep-monday-in-test", action="store_true",
                    help="旧口径：允许 Monday 行（= 检测器的训练数据）进测试集，仅用于对照")
    args = ap.parse_args(argv)

    build(CicidsConfig(csv_dir=args.csv_dir, out_path=args.out,
                       rows_normal=args.rows_normal, rows_per_file=args.rows_per_file,
                       seed=args.seed,
                       eval_exclude_normal_source=not args.keep_monday_in_test))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
