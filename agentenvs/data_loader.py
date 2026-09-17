"""数据加载：把 chethuhn/n网络流量 CSV 或合成数据变成环境可用的数组。

三种数据来源
------------
1. `load_cicids(...)`  读取 `chethuhn/network-intrusion-dataset/versions/1/*.csv`
2. `load_npz(...)`     读取 `save_bundle()` 落盘的 .npz（推荐：只解析一次 CSV）
3. `make_synthetic(...)` 合成数据，用于快速跑通 / CI 冒烟测试

统一产物 `TrafficDataset`：
    X_normal_train  良性流量（只给检测器 fit / 归一化统计用）
    X_train, y_train  带标签训练集（MADDPG 交互用）
    X_test,  y_test   带标签测试集（评估用）
    feature_names, class_names
    X 全部已按 X_normal_train 做 Min-Max 归一化到 [0,1]（对齐论文 Eq.1）
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .taxonomy import BENIGN, ClassSpec, normalize_label

#: 默认的数据目录（仓库内）
DEFAULT_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "chethuhn", "network-intrusion-dataset", "versions", "1",
)


# ----------------------------------------------------------------------------- 配置
@dataclass
class FileSpec:
    """一个 CSV 文件的采样配置。"""

    filename: str
    nrows: Optional[int] = 20000          # 该文件最多采样多少行（None = 全量）
    seed: int = 42
    encoding: str = "latin-1"             # CICIDS2017 官方 CSV 兼容性最好


@dataclass
class DataConfig:
    data_dir: str = DEFAULT_DATA_DIR
    files: Optional[List[FileSpec]] = None      # None -> 用 default_file_specs()
    normal_source: Optional[List[FileSpec]] = None  # 良性流量（检测器训练）来源
    class_preset: str = "default"
    normal_ratio_in_train: float = 0.30   # 训练集中良性占比上限（避免极度不均衡）
    test_size: float = 0.2
    val_size: float = 0.0                 # 从训练集再切一份验证集（可选）
    min_attack_rows: int = 200            # 少于该样本数的攻击类并入 Other
    drop_duplicate_labels: bool = True
    max_features: Optional[int] = None    # 可选的方差筛选（None = 全用）
    seed: int = 42

    def resolved_files(self) -> List[FileSpec]:
        return list(self.files) if self.files else default_file_specs()

    def resolved_normal_source(self) -> List[FileSpec]:
        if self.normal_source is not None:
            return list(self.normal_source)
        # 默认用 Monday（纯良性）来训练检测器
        return [FileSpec("Monday-WorkingHours.pcap_ISCX.csv", nrows=60000, seed=self.seed)]


@dataclass
class TrafficDataset:
    """环境与检测器共享的数据容器。"""

    X_normal_train: np.ndarray
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    X_val: Optional[np.ndarray] = None
    y_val: Optional[np.ndarray] = None
    feature_names: List[str] = field(default_factory=list)
    class_names: List[str] = field(default_factory=list)
    spec: Optional[ClassSpec] = None
    meta: Dict = field(default_factory=dict)

    @property
    def feat_dim(self) -> int:
        return int(self.X_train.shape[1])

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def summary(self) -> str:
        lines = [
            f"特征维度      : {self.feat_dim}",
            f"类别          : {self.class_names}",
            f"检测器良性集  : {len(self.X_normal_train)} 行（全 benign）",
            f"训练集        : {len(self.X_train)} 行",
            f"测试集        : {len(self.X_test)} 行",
        ]
        if self.X_val is not None:
            lines.append(f"验证集        : {len(self.X_val)} 行")
        cnt = np.bincount(self.y_train, minlength=self.num_classes)
        lines.append("训练集类别分布: " + ", ".join(
            f"{self.class_names[i]}={int(cnt[i])}" for i in range(self.num_classes)
        ))
        cnt_t = np.bincount(self.y_test, minlength=self.num_classes)
        lines.append("测试集类别分布: " + ", ".join(
            f"{self.class_names[i]}={int(cnt_t[i])}" for i in range(self.num_classes)
        ))
        if self.meta.get("label_map"):
            lines.append("原始标签映射  : " + str(self.meta["label_map"]))
        return "\n".join(lines)


def default_file_specs(nrows: int = 20000, seed: int = 42) -> List[FileSpec]:
    """覆盖 DEFAULT_ATTACK_CLASSES 里全部类别的最小文件集合。"""
    return [
        FileSpec("Monday-WorkingHours.pcap_ISCX.csv", nrows, seed),                          # 纯 benign
        FileSpec("Tuesday-WorkingHours.pcap_ISCX.csv", nrows, seed + 1),                     # Patator
        FileSpec("Wednesday-workingHours.pcap_ISCX.csv", nrows, seed + 2),                   # DoS 族
        FileSpec("Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv", nrows, seed + 3), # WebAttack
        FileSpec("Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv", nrows // 4, seed + 4),
        FileSpec("Friday-WorkingHours-Morning.pcap_ISCX.csv", nrows, seed + 5),              # Bot
        FileSpec("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv", nrows, seed + 6),       # DDoS
        FileSpec("Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv", nrows, seed + 7),   # PortScan
    ]


# ----------------------------------------------------------------------------- CSV 读取
def _read_csv_sampled(path: str, spec: FileSpec, feature_cols: Optional[Sequence[str]] = None):
    """分块读取一个 CSV，按 spec.nrows 随机保留行；返回 (DataFrame, 全部列名)。"""
    import pandas as pd

    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到数据文件：{path}")

    header = pd.read_csv(path, nrows=0, encoding=spec.encoding)
    raw_cols = list(header.columns)                 # CICIDS2017 的表头带前后空格
    all_cols = [str(c).strip() for c in raw_cols]
    header.columns = all_cols
    raw_by_clean = {str(c).strip(): c for c in raw_cols}

    want = set(feature_cols) if feature_cols is not None else None
    # 注意：usecols 必须用 CSV 里的原始列名（带空格），读出来之后再 strip
    clean_use = [c for c in all_cols if want is None or c in want]
    if "Label" in all_cols:
        clean_use = clean_use + ["Label"]
    clean_use = list(dict.fromkeys(clean_use))
    usecols = [raw_by_clean[c] for c in clean_use if c in raw_by_clean]

    # 预估总行数：用平均行长估算，够用即可（早停判据，不要求精确）
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
    for chunk in pd.read_csv(
        path, usecols=usecols, chunksize=20000, low_memory=False, encoding=spec.encoding
    ):
        chunk.columns = [c.strip() for c in chunk.columns]
        n = len(chunk)
        if spec.nrows is None:
            chunks.append(chunk)
            kept += n
            seen += n
            continue
        # 按"剩余配额 / 剩余行数"的比例做无偏采样
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
    df = pd.concat(chunks, ignore_index=True)
    if spec.nrows is not None and len(df) > spec.nrows:
        df = df.sample(n=spec.nrows, random_state=spec.seed).reset_index(drop=True)
    return df, all_cols


def _sanitize_features(df, feature_cols: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    """清洗特征：inf/NaN -> 列中位数；返回 (X_raw, 有效行 mask)。"""
    import pandas as pd

    X = df.loc[:, feature_cols].copy()
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    valid = ~X.isna().all(axis=1).to_numpy()
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
    return X.to_numpy(dtype=np.float32), valid


def normalize_features(
    X_train_norm: np.ndarray, *others: np.ndarray
) -> Tuple[np.ndarray, List[np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Min-Max 归一化到 [0,1]（论文 Eq.1），统计量只用良性训练集估计。
    返回 (X_normal_scaled, [others_scaled...], (min, range))
    """
    lo = np.min(X_train_norm, axis=0).astype(np.float32)
    hi = np.max(X_train_norm, axis=0).astype(np.float32)
    rng = np.where((hi - lo) < 1e-12, 1.0, hi - lo).astype(np.float32)

    def _scale(A):
        if A is None:
            return None
        return np.clip((np.asarray(A, dtype=np.float32) - lo) / rng, 0.0, 1.0).astype(np.float32)

    return _scale(X_train_norm), [_scale(o) for o in others], (lo, rng)


# ----------------------------------------------------------------------------- 主流程
def build_dataset(cfg: Optional[DataConfig] = None) -> TrafficDataset:
    """按 DataConfig 读 CSV → 归一化 → 划分 → 返回 TrafficDataset。"""
    cfg = cfg or DataConfig()
    spec = ClassSpec.from_preset(cfg.class_preset)

    # ---------- 1. 良性流量（检测器 fit 用） ----------
    normal_frames = []
    all_cols: Optional[List[str]] = None
    for fs in cfg.resolved_normal_source():
        path = os.path.join(cfg.data_dir, fs.filename)
        df, cols = _read_csv_sampled(path, fs)
        if all_cols is None:
            all_cols = cols
        lab = df["Label"].map(normalize_label)
        normal_frames.append(df.loc[lab == BENIGN])
    normal_df = _concat(normal_frames)
    feature_cols = [c for c in (all_cols or []) if c != "Label"]

    # ---------- 2. 带标签流量（训练/测试用） ----------
    frames, label_map = [], {}
    for fs in cfg.resolved_files():
        path = os.path.join(cfg.data_dir, fs.filename)
        df, _ = _read_csv_sampled(path, fs, feature_cols=feature_cols)
        raw = df["Label"].astype(str)
        canon = raw.map(normalize_label)
        # 记录原始 -> 规范 的映射（抽样即可，避免对几百万行做 Python 级循环）
        label_map.update(dict(zip(raw.head(5000).tolist(), canon.head(5000).tolist())))
        df = df.copy()
        df["_canon"] = canon.values
        frames.append(df)

    labeled = _concat(frames)
    y_canon = labeled["_canon"].astype(str).to_numpy()
    X_raw, valid = _sanitize_features(labeled, feature_cols)
    y_canon = y_canon[valid]

    Xn_raw, n_valid = _sanitize_features(normal_df, feature_cols)
    Xn_raw = Xn_raw[n_valid]

    # ---------- 3. 归一化（统计量只来自良性集） ----------
    Xn, (Xall,), _ = normalize_features(Xn_raw, X_raw)
    Xall = np.asarray(Xall, dtype=np.float32)

    # ---------- 4. 类别索引 + 稀有类并入 Other ----------
    y = np.array([spec.index_of(c) for c in y_canon], dtype=np.int64)
    class_names = spec.names
    y, class_names, merge_info = _merge_rare(
        y, class_names, Xall, min_rows=cfg.min_attack_rows, other_name=class_names[-1]
    )
    spec_out = ClassSpec(benign_name=class_names[0], attack_classes=class_names[1:])

    # ---------- 5. 划分 ----------
    rng = np.random.default_rng(cfg.seed)
    X_tr, y_tr, X_te, y_te = _split(Xall, y, cfg.test_size, rng)
    X_val = y_val = None
    if cfg.val_size > 0:
        X_tr, y_tr, X_val, y_val = _split(X_tr, y_tr, cfg.val_size, rng)
    X_tr, y_tr = _balance_normal(X_tr, y_tr, cfg.normal_ratio_in_train, rng)

    # ---------- 6. 可选方差筛选 ----------
    if cfg.max_features and cfg.max_features < X_tr.shape[1]:
        keep_idx = np.argsort(-X_tr.var(axis=0))[: cfg.max_features]
        keep_idx.sort()
        X_tr, X_te, Xn = X_tr[:, keep_idx], X_te[:, keep_idx], Xn[:, keep_idx]
        if X_val is not None:
            X_val = X_val[:, keep_idx]
        feature_cols = [feature_cols[i] for i in keep_idx]

    return TrafficDataset(
        X_normal_train=Xn,
        X_train=X_tr, y_train=y_tr,
        X_test=X_te, y_test=y_te,
        X_val=X_val, y_val=y_val,
        feature_names=list(feature_cols),
        class_names=class_names,
        spec=spec_out,
        meta={"label_map": label_map, "merged": merge_info,
              "source": "cicids2017", "data_dir": cfg.data_dir},
    )


def _concat(frames: List):
    import pandas as pd

    if not frames:
        raise RuntimeError("没有可用的数据帧")
    return pd.concat(frames, ignore_index=True)


def _merge_rare(y, class_names, X, min_rows: int, other_name: str):
    """样本数 < min_rows 的攻击类并入 other_name，并把索引重排连续。"""
    cnt = np.bincount(y, minlength=len(class_names))
    keep = [0]  # BENIGN
    drop: Dict[str, int] = {}
    for i in range(1, len(class_names)):
        if cnt[i] >= min_rows or class_names[i] == other_name:
            keep.append(i)
        else:
            drop[class_names[i]] = int(cnt[i])
    if not drop:
        return y, list(class_names), {}
    # 把被丢掉的类映射到 other
    other_local = keep.index(class_names.index(other_name)) if other_name in class_names else len(keep) - 1
    remap = np.zeros(len(class_names), dtype=np.int64)
    for new_i, old_i in enumerate(keep):
        remap[old_i] = new_i
    y_new = y.copy()
    if other_name in class_names:
        y_new[np.isin(y, [class_names.index(k) for k in drop])] = other_local
    y_new = remap[y_new]
    return y_new, [class_names[i] for i in keep], drop


def _split(X, y, test_size: float, rng):
    """按类别分层切分。"""
    n = len(y)
    n_test = int(round(n * test_size))
    test_idx: List[int] = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        k = int(round(len(idx) * test_size))
        test_idx.extend(idx[:k].tolist())
    test_idx = np.array(sorted(test_idx), dtype=np.int64)
    if len(test_idx) > n_test:
        test_idx = test_idx[:n_test]
    train_mask = np.ones(n, dtype=bool)
    train_mask[test_idx] = False
    return X[train_mask], y[train_mask], X[test_idx], y[test_idx]


def _balance_normal(X, y, max_ratio: float, rng):
    """下采样训练集中的良性样本，避免类别极度不均衡。"""
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


# ----------------------------------------------------------------------------- 合成数据
def make_synthetic(
    n_normal: int = 3000,
    n_per_class: int = 600,
    feat_dim: int = 26,
    num_attack_classes: int = 5,
    seed: int = 0,
) -> TrafficDataset:
    """合成数据：每类攻击偏移一组专属特征，方便快速验证学习是否真的发生。"""
    rng = np.random.default_rng(seed)
    spec = ClassSpec(attack_classes=[f"Atk{i + 1}" for i in range(num_attack_classes)])
    names = spec.names

    Xn = np.abs(rng.normal(0.3, 0.08, size=(n_normal, feat_dim))).astype(np.float32)
    Xs, ys = [Xn], [np.zeros(len(Xn), dtype=np.int64)]
    block = max(1, feat_dim // (num_attack_classes + 2))
    for c in range(1, num_attack_classes + 1):
        lo = ((c - 1) * block) % feat_dim
        idx = [(lo + k) % feat_dim for k in range(block)]
        Xc = np.abs(rng.normal(0.3, 0.08, size=(n_per_class, feat_dim))).astype(np.float32)
        Xc[:, idx] += 0.35
        Xs.append(Xc)
        ys.append(np.full(n_per_class, c, dtype=np.int64))
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)

    # 切分
    rng2 = np.random.default_rng(seed + 1)
    X_tr, y_tr, X_te, y_te = _split(X, y, 0.25, rng2)
    X_tr, y_tr = _balance_normal(X_tr, y_tr, 0.5, rng2)
    Xn_scaled, (a, b), _ = normalize_features(Xn, X_tr, X_te)

    return TrafficDataset(
        X_normal_train=np.asarray(Xn_scaled, dtype=np.float32),
        X_train=np.asarray(a, dtype=np.float32), y_train=y_tr,
        X_test=np.asarray(b, dtype=np.float32), y_test=y_te,
        feature_names=[f"f{i}" for i in range(feat_dim)],
        class_names=names, spec=spec,
        meta={"label_map": {}, "source": "synthetic"},
    )


# ----------------------------------------------------------------------------- 落盘
def save_bundle(ds: TrafficDataset, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = dict(
        X_normal_train=ds.X_normal_train, X_train=ds.X_train, y_train=ds.y_train,
        X_test=ds.X_test, y_test=ds.y_test,
        feature_names=np.array(ds.feature_names, dtype=object),
        class_names=np.array(ds.class_names, dtype=object),
        source=str(ds.meta.get("source", "unknown")),
    )
    if ds.X_val is not None:
        payload["X_val"] = ds.X_val
        payload["y_val"] = ds.y_val
    np.savez_compressed(path, **payload)


def load_bundle(path: str) -> TrafficDataset:
    if not path.endswith(".npz"):
        path = path + ".npz"
    blob = np.load(path, allow_pickle=True)
    class_names = [str(x) for x in blob["class_names"]]
    spec = ClassSpec(benign_name=class_names[0], attack_classes=class_names[1:])
    return TrafficDataset(
        X_normal_train=blob["X_normal_train"].astype(np.float32),
        X_train=blob["X_train"].astype(np.float32), y_train=blob["y_train"].astype(np.int64),
        X_test=blob["X_test"].astype(np.float32), y_test=blob["y_test"].astype(np.int64),
        X_val=blob["X_val"].astype(np.float32) if "X_val" in blob else None,
        y_val=blob["y_val"].astype(np.int64) if "y_val" in blob else None,
        feature_names=[str(x) for x in blob["feature_names"]],
        class_names=class_names, spec=spec,
        meta={"source": str(blob["source"]), "bundle": path},
    )
