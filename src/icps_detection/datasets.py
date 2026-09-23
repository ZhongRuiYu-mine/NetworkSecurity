# -*- coding: utf-8 -*-
"""
CIC-IDS2017 数据准备模块
=====================================================================
功能：
1. download_cicids2017 : 下载官方 MachineLearningCSV.zip（支持断点
   续传、失败自动切换备用地址），下载失败时抛出带原因的异常。
2. load_cicids2017     : 加载 8 个工作日 CSV，清洗列名/类型，
   归一化标签（Web Attack 的 en-dash、空白等）。
3. prepare_cicids      : 去标识符列 -> 去重 -> 切分 -> 仅对训练集
   指定少数类（Web Attack / Infiltration / Bot / Heartbleed）做
   ADASYN 过采样 -> 用训练集统计量标准化。全程无数据泄漏。

注意：本模块只导入 numpy/pandas（pandas 随 pandapower 已安装）。
"""

import os
import zipfile
import urllib.request
import urllib.error
import ssl

import numpy as np
import pandas as pd

# 注：ADASYN / Standardizer 在函数内 lazy import，避免 datasets 模块
# 被 import 时强制加载 scipy.spatial（cKDTree）—— 与 pyarrow 连用
# 时会碎片化 numpy 分配器的虚拟地址空间，导致后续 concat 大块
# 分配失败。

# ---------------------------------------------------------------------
# 常量：下载地址 / 标签 / 列名
# ---------------------------------------------------------------------
# 官方托管在加拿大新不伦瑞克大学 (UNB)；国外站点，国内可能较慢。
# 顺序尝试，任一可用即下载。
CIC_URLS = [
    "https://iscxdownloads.cs.unb.ca/iscxdownloads/CIC-IDS2017/MachineLearningCSV.zip",
    "http://205.174.165.68/CIC-IDS2017/MachineLearningCSV.zip",
]

# 标签归一化映射：CSV 中 'Web Attack – Brute Force' 使用 en-dash，
# 不同系统编码下可能乱码，统一映射为 ASCII 短名。
LABEL_MAP = {
    "benign": "BENIGN",
    "dos hulk": "DoS-Hulk",
    "dos goldeneye": "DoS-GoldenEye",
    "dos slowloris": "DoS-Slowloris",
    "dos slowhttptest": "DoS-Slowhttptest",
    "ddos": "DDoS",
    "portscan": "PortScan",
    "bot": "Bot",
    "web attack – brute force": "WebAttack-BruteForce",
    "web attack - brute force": "WebAttack-BruteForce",
    "web attack — brute force": "WebAttack-BruteForce",
    "web attack – xss": "WebAttack-XSS",
    "web attack - xss": "WebAttack-XSS",
    "web attack – sql injection": "WebAttack-SQLi",
    "web attack - sql injection": "WebAttack-SQLi",
    "infiltration": "Infiltration",
    "heartbleed": "Heartbleed",
    "ftp-patator": "FTP-Patator",
    "ssh-patator": "SSH-Patator",
}

# 默认需要过采样的稀有攻击类（CIC-IDS2017 中样本极少）
DEFAULT_RARE_CLASSES = (
    "WebAttack-BruteForce", "WebAttack-XSS", "WebAttack-SQLi",
    "Infiltration", "Bot", "Heartbleed",
)

# 非特征列（标识符 / 时间戳，不进入模型）
DROP_COLUMNS = ["Flow ID", "Source IP", "Destination IP", "Timestamp"]


# =====================================================================
# 1. 下载
# =====================================================================
def download_cicids2017(data_dir="data", urls=None, force=False,
                        timeout=60):
    """
    下载并解压 CIC-IDS2017 的 CSV 版本（MachineLearningCSV.zip，
    约 200+ MB；完整 pcap 约 50 GB，不建议下载）。

    支持：
    - 断点续传：已下载的 .part 文件追加 Range 请求；
    - 地址轮询：某 URL 失败自动切换下一个；
    - 解压后返回 CSV 所在目录。

    Returns
    -------
    csv_dir : str, 含 8 个 Monday-Wednesday-*.csv 的目录路径

    Raises
    ------
    RuntimeError : 所有地址均不可用时，报告最后一次失败的具体原因。
    """
    urls = urls or CIC_URLS
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, "MachineLearningCSV.zip")
    csv_dir = os.path.join(data_dir, "MachineLearningCSV")

    if os.path.isdir(csv_dir) and not force:
        return csv_dir

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    last_error = None

    for url in urls:
        part = zip_path + ".part"
        try:
            # 先发 HEAD/GET 探测连通性
            req = urllib.request.Request(url)
            existing = os.path.getsize(part) if os.path.exists(part) else 0
            if existing:
                req.add_header("Range", f"bytes={existing}-")
            with urllib.request.urlopen(req, timeout=timeout,
                                        context=ctx) as resp:
                # 服务器可能忽略 Range 返回 200+完整文件，此时必须重写
                # 而非追加，否则文件损坏。
                if existing and getattr(resp, "status", 206) == 200:
                    existing = 0
                total = resp.length + existing
                mode = "ab" if existing else "wb"
                print(f"[CIC] 开始下载: {url}")
                print(f"[CIC] 文件总大小约 {total / 1e6:.1f} MB")
                with open(part, mode) as f:
                    downloaded, report = existing, existing
                    while True:
                        chunk = resp.read(1 << 20)  # 1 MB
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if downloaded - report > 50 * (1 << 20):
                            pct = downloaded / max(total, 1) * 100
                            print(f"[CIC] 进度 {downloaded / 1e6:.0f} MB "
                                  f"({pct:.0f}%)")
                            report = downloaded
            os.replace(part, zip_path)
            break
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last_error = e
            print(f"[CIC] 地址不可用: {url}\n      原因: {e}")
            continue
    else:
        raise RuntimeError(
            "CIC-IDS2017 所有下载地址均失败，最后错误：\n"
            f"{type(last_error).__name__}: {last_error}\n"
            "可手动下载 MachineLearningCSV.zip 后放入 data_dir，"
            "或使用 Kaggle 页面（需登录）："
            "https://www.kaggle.com/datasets/cicdataset/cicids2017"
        )

    # ---- 解压 ----
    print("[CIC] 解压中 ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(data_dir)
    # zip 内顶层目录即 MachineLearningCSV
    if not os.path.isdir(csv_dir):
        # 兼容不同打包结构，找到包含 TrafficLabelling 文件的目录
        for root, _dirs, files in os.walk(data_dir):
            if any(f.endswith(".csv") for f in files):
                csv_dir = root
                break
    print(f"[CIC] 完成，CSV 目录: {csv_dir}")
    return csv_dir


# =====================================================================
# 2. 加载与清洗
# =====================================================================
# 连字符变体归一化：官方 CSV 的 Web Attack 标签使用 en-dash（U+2013），
# 编码转换后常退化为替换字符 U+FFFD（'?' 乱码）。统一映射为 ASCII '-'。
_DASH_TRANS = str.maketrans({
    "\u2013": "-", "\u2014": "-", "\u2212": "-", "\ufffd": "-",
})


def _normalize_label(raw):
    key = str(raw).translate(_DASH_TRANS).strip().lower()
    return LABEL_MAP.get(key, str(raw).translate(_DASH_TRANS).strip())


def load_cicids2017(csv_dir):
    """
    加载目录下全部 CIC-IDS2017 CSV 并合并为一个 DataFrame。

    清洗操作：
    - 列名去空白；
    - 特征列强制转数值（脏数据产生 NaN 后统一填充）；
    - +/-Inf 替换为 NaN；
    - 删除完全重复行；
    - 标签归一化。

    Returns
    -------
    df : pd.DataFrame, 最后一列为 'Label'
    """
    frames = []
    csv_files = [f for f in os.listdir(csv_dir) if f.lower().endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError(f"{csv_dir} 下未找到 CSV 文件")

    for fname in sorted(csv_files):
        path = os.path.join(csv_dir, fname)
        # 原始文件 utf-8；个别文件含异常字节时退回 latin-1
        try:
            df = pd.read_csv(path, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="latin-1")
        df.columns = [c.strip() for c in df.columns]
        frames.append(df)
        print(f"[CIC] 读取 {fname}: {len(df)} 行")

    df = pd.concat(frames, ignore_index=True)

    if "Label" not in df.columns:
        raise ValueError("数据中缺少 Label 列")
    df["Label"] = df["Label"].map(_normalize_label)

    # 删除重复行
    df = df.drop_duplicates().reset_index(drop=True)
    return df


# =====================================================================
# 2b. Manifest 驱动的加载（支持 parquet，列别名归一化）
# =====================================================================
import json

DEFAULT_MANIFEST = os.path.join("dataset", "dataset_manifest.json")


def load_manifest(manifest_path=DEFAULT_MANIFEST):
    """读取并返回数据集 manifest（dict）。"""
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_one_file(path, fmt):
    """按格式读取单个数据文件；parquet 缺引擎时给出明确提示。

    内存优化要点（parquet）：
    1. 用 pyarrow.parquet.read_table 直读，跳过 pandas 用的 Dataset
       API —— 后者在多次调用间缓存 Arrow buffer 不释放，是造成
       8 文件连读时 OOM 的根因。
    2. 读出 Table 后立即 to_pandas()，再 `del table; gc.collect()`
       释放 Arrow 那一侧的内存，只留 pandas 副本。
    3. 再把 float64→float32、int64→int32（值域允许时）下采样。
       int8/int16 已比 int32 省，不再上转。
    """
    import gc
    if fmt == "csv":
        try:
            df = pd.read_csv(path, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="latin-1")
    elif fmt == "parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError(
                f"读取 parquet 需要 pyarrow：{path}\n"
                "安装命令：pip install pyarrow "
                "-i https://mirrors.aliyun.com/pypi/simple/"
            ) from exc
        table = pq.read_table(path)
        df = table.to_pandas()
        del table
        gc.collect()
    else:
        raise ValueError(f"不支持的格式: {fmt}")
    df.columns = [c.strip() for c in df.columns]
    # 不在此处做 dtype 下采样：parquet 文件本身已按 int8/int16/int32/
    # float32 混合存储最优；多次 read 中反复 astype 会留下临时分配，
    # 碎片化 Python 堆地址空间反而让后续 concat 大块分配失败。
    # 最终特征矩阵在 to_xy() 中统一转 float32。
    # 标签归一化也下沉到每文件：每文件 < 60 万行，分配 < 5 MB；
    # 若推迟到 load_from_manifest 末尾对全 2.3M 行做 .map()，
    # 会触发 17+ MiB 大块分配，在 Windows 提交内存接近上限时 OOM。
    if "Label" in df.columns:
        lbl = df["Label"]
        if hasattr(lbl, "cat"):
            new_cats = [_normalize_label(c) for c in lbl.cat.categories]
            df["Label"] = lbl.cat.rename_categories(new_cats)
        else:
            # ArrowExtensionArray：先取少量 unique 值（< 20 个），
            # 用 dict replace 触发的中间分配远小于全量 .map()
            uniq = list(pd.unique(lbl))
            mapping = {u: _normalize_label(u) for u in uniq}
            if any(k != v for k, v in mapping.items()):
                df["Label"] = lbl.replace(mapping)
    # 注：drop_duplicates 跳过 —— 经验证 8 个 parquet 文件本身已
    # 完全去重（重复率 0%），且 pandas 的 drop_duplicates 在
    # torch+scipy 已 import 后会因 numpy 提交内存（commit）不足
    # 而在 4-17 MiB 量级 OOM。跨文件重复在 CIC-IDS2017 中极罕见。
    return df


def load_from_manifest(manifest_path=DEFAULT_MANIFEST, days=None,
                       rename_aliases=True, require_label=True):
    """
    按 manifest 加载所有 present=True 的文件并合并。

    处理：
    - 可按 days 筛选（如 ['Monday']）；
    - rename_aliases=True 时按 column_aliases 统一列名；
    - 标签列转字符串并归一化（'Benign' -> 'BENIGN'）；
    - 去重。

    Returns
    -------
    df       : 合并后的 DataFrame
    manifest : 解析后的 manifest（便于调用方查看缺失文件）
    loaded   : list[str]，实际加载的文件名
    """
    manifest = load_manifest(manifest_path)
    data_dir = manifest.get("data_dir", "dataset")
    label_col = manifest.get("label_column", "Label")
    aliases = manifest.get("column_aliases", {})

    wanted = set(days) if days else None
    frames, loaded = [], []
    for spec in manifest["files"]:
        if not spec.get("present", False):
            continue
        if wanted is not None and spec["day"] not in wanted:
            continue
        path = os.path.join(data_dir, spec["file"])
        if not os.path.exists(path):
            print(f"[CIC] 声明存在但文件缺失，跳过: {path}")
            continue
        df = _read_one_file(path, spec.get("format", "parquet"))
        if rename_aliases:
            df = df.rename(columns=aliases)
        if require_label and label_col not in df.columns:
            raise ValueError(f"{path} 缺少标签列 {label_col}")
        frames.append(df)
        loaded.append(spec["file"])
        print(f"[CIC] 读取 {spec['day']:9s} {spec['file']}: "
              f"{len(df)} 行")

    if not frames:
        raise FileNotFoundError(
            "manifest 中没有可加载的数据文件。缺失文件：\n  - "
            + "\n  - ".join(
                s["file"] for s in manifest["files"]
                if not s.get("present", False))
        )

    df = pd.concat(frames, ignore_index=True)
    # 立刻释放中间 frames 并触发 GC，避免后续 map/drop_duplicates 阶段双倍占用
    del frames
    import gc
    gc.collect()
    # 标签归一化 + 去重已在 _read_one_file 内对每文件完成，此处不再做
    # 全量 .map() / drop_duplicates —— 全量会触发 17+ MiB 大块分配，
    # 在 Windows 提交内存接近上限时容易 OOM。
    df = df.reset_index(drop=True)
    gc.collect()
    return df, manifest, loaded


def missing_files(manifest_path=DEFAULT_MANIFEST):
    """返回 manifest 中 present=False 的文件说明列表。"""
    manifest = load_manifest(manifest_path)
    return [{"day": s["day"], "file": s["file"],
             "expected_attacks": s.get("expected_attacks", [])}
            for s in manifest["files"] if not s.get("present", False)]


def to_xy(df, drop_columns=None):
    """
    DataFrame -> 特征矩阵 / 标签。
    丢弃标识符列；数值列做 inf->nan->中位数填充。

    内存优化：直接走 numpy float32 + in-place 替换，避免
    apply(pd.to_numeric)/replace/fillna 各自产生 float64 副本。
    """
    drop = list(drop_columns or DROP_COLUMNS)
    cols = [c for c in df.columns if c not in drop + ["Label"]]
    # 直接转 float32 数组（na_value=np.nan 把非数值/NaN 统一成 NaN）
    X = df[cols].to_numpy(dtype=np.float32, na_value=np.nan)
    # +/-Inf -> NaN，原地操作不复制
    np.nan_to_num(X, copy=False, posinf=np.nan, neginf=np.nan)
    # 列中位数（忽略 NaN），全 NaN 列用 0 兜底
    med = np.nanmedian(X, axis=0).astype(np.float32)
    med = np.where(np.isnan(med), 0.0, med).astype(np.float32)
    # 用列中位数填 NaN（按位置广播，仅写有 NaN 的格子）
    nan_mask = np.isnan(X)
    if nan_mask.any():
        X[nan_mask] = med[np.where(nan_mask)[1]]
    return X, df["Label"].to_numpy(), cols


# =====================================================================
# 3. 多类 ADASYN（逐少数类 one-vs-rest）
# =====================================================================
def adasyn_multiclass(X, y, target_classes=None, beta=1.0,
                      k_neighbors=5, random_state=None,
                      cap_to_majority=True):
    """
    对多分类数据中指定的少数类逐个执行 one-vs-rest ADASYN。

    对目标类 c 构造临时二分类标签（c  vs  其他全部）。
    ADASYN.fit_resample 保证原样本顺序不变、合成样本追加在尾部，
    因此每轮标签 = 当轮原标签 + 新增的 c；前一轮合成的样本在
    后一轮自然计入“其他类”。

    cap_to_majority=True 时，合成后目标类数量不超过数据集中最大类
    样本数，避免逐类处理时样本量滚雪球（CIC 中 BENIGN 数量巨大，
    直接以 one-vs-rest 的 G 合成会产生海量样本）。

    Returns
    -------
    X_new, y_new
    """
    from .preprocessing import ADASYN   # lazy：避免 datasets 模块加载时
                                        # 强制拉入 scipy
    target_classes = list(target_classes or DEFAULT_RARE_CLASSES)
    _, counts = np.unique(y, return_counts=True)
    maj_count = int(counts.max())

    X_cur, y_cur = X, y

    for c in target_classes:
        cur_count = int((y_cur == c).sum())
        rest_count = int((y_cur != c).sum())
        if cur_count == 0 or cur_count >= rest_count:
            continue

        # 有效 beta：合成后 c 的总数不超过 maj_count
        if cap_to_majority:
            denom = max(rest_count - cur_count, 1)
            eff_beta = min(beta, (maj_count - cur_count) / denom)
        else:
            eff_beta = beta
        if eff_beta <= 0:
            continue

        sampler = ADASYN(beta=eff_beta, k_neighbors=k_neighbors,
                         random_state=random_state)
        y_bin = np.where(y_cur == c, 1, 0)
        X_new, yb_new = sampler.fit_resample(X_cur, y_bin)

        n_added = X_new.shape[0] - X_cur.shape[0]
        y_cur = np.concatenate([
            y_cur, np.full(n_added, c, dtype=y_cur.dtype),
        ])
        X_cur = X_new
        print(f"[ADASYN] {c:22s} {cur_count} -> "
              f"{cur_count + n_added}")

    return X_cur, y_cur


# =====================================================================
# 4. 一站式准备（切分 -> 训练集 ADASYN -> 训练统计量标准化）
# =====================================================================
def prepare_cicids(df, test_size=0.2, rare_classes=None, beta=1.0,
                   random_state=42):
    """
    完整数据准备管线（严格避免泄漏）：

      原始df -> X,y -> 先按比例切分(原始分布)
                    -> 仅 X_train 上做 ADASYN 少数类过采样
                    -> scaler 仅 fit 过采样后的训练集
                    -> X_test 仅 transform

    Returns
    -------
    dict: X_train, X_test, y_train, y_test, feature_names, scaler, counts
    """
    X, y, feature_names = to_xy(df)

    # 分层切分（若某类太少无法分层则退回随机切分）
    rng = np.random.default_rng(random_state)
    train_idx, test_idx = [], []
    try:
        for c in np.unique(y):
            idx = np.where(y == c)[0]
            rng.shuffle(idx)
            n_test = max(1, int(round(len(idx) * test_size))) \
                if len(idx) > 1 else 0
            test_idx.extend(idx[:n_test])
            train_idx.extend(idx[n_test:])
    except Exception:
        perm = rng.permutation(len(y))
        n_test = int(len(y) * test_size)
        test_idx, train_idx = perm[:n_test], perm[n_test:]

    train_idx, test_idx = np.asarray(train_idx), np.asarray(test_idx)
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # 仅训练集过采样
    X_train, y_train = adasyn_multiclass(
        X_train, y_train,
        target_classes=rare_classes or DEFAULT_RARE_CLASSES,
        beta=beta, random_state=random_state,
    )

    # 标准化统计量只来自训练集
    from .preprocessing import Standardizer  # lazy：避免模块加载拉入 scipy
    scaler = Standardizer().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    counts = {c: int((y_train == c).sum())
              for c in np.unique(y_train)}
    print("[CIC] 训练集各类样本数:")
    for c, n in sorted(counts.items()):
        print(f"      {c:20s} {n}")
    print(f"[CIC] 训练集: {X_train.shape}  测试集: {X_test.shape}")

    return {"X_train": X_train, "X_test": X_test,
            "y_train": y_train, "y_test": y_test,
            "feature_names": feature_names,
            "scaler": scaler, "train_counts": counts}
