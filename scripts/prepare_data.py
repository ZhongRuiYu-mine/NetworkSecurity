"""
把 CICIDS2017 原始 CSV 预处理成环境直接可用的数据包（.npz），一次解析、反复使用。

用法（在仓库根运行）：
    python scripts/prepare_data.py --out data/cicids_sample.npz

    注意：另有更完整的协议 v1 生成器 `python -m ae_repro.data_prep`
    （默认就写到 data/cicids_sample.npz，且评估集排除 Monday 行，
    见 PROTOCOL.md）。本脚本是平台的原始构建器，保留用于对照。

产物 TrafficDataset 内容：
    X_normal_train   良性流量（检测器 fit + 归一化统计）
    X_train/y_train  带标签训练集（MADDPG 交互）
    X_test/y_test    带标签测试集（评估）
    feature_names / class_names

注意：
  * 所有特征按良性训练集做 Min-Max 归一化到 [0,1]（论文 Eq.1）；
  * 原始 Label 里的破损编码（Web Attack ? Brute Force 等）会被归一化（见 taxonomy.py）；
  * 样本数不足以进入类别空间的攻击族会被并入 "Other" 兜底类。
"""
from __future__ import annotations


# --- 合并仓库引导（scripts/_bootstrap） ---
import os as _os, sys as _sys
REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_os.path.join(REPO_ROOT, "src"), REPO_ROOT):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
del _os, _sys, _p
# --- 合并仓库引导结束 ---

import argparse
import os
import sys
import time

from agentenvs import (
    DEFAULT_DATA_DIR,
    DataConfig,
    FileSpec,
    build_dataset,
    default_file_specs,
    save_bundle,
)


def main() -> int:
    ap = argparse.ArgumentParser(description="预处理 CICIDS2017 -> .npz 数据包")
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "data", "cicids_sample.npz"))
    ap.add_argument("--rows-per-file", type=int, default=20000,
                    help="每个 CSV 采样行数（越大类别越全，解析也越慢）")
    ap.add_argument("--normal-rows", type=int, default=60000,
                    help="用于训练检测器的良性流量条数（取 Monday）")
    ap.add_argument("--class-preset", default="default", choices=["default", "full", "binary"])
    ap.add_argument("--normal-ratio", type=float, default=0.35,
                    help="训练集中良性流量占比上限（下采样得到）")
    ap.add_argument("--min-attack-rows", type=int, default=200,
                    help="少于该样本数的攻击类并入 Other")
    ap.add_argument("--max-features", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    files = [
        FileSpec(f.filename, min(f.nrows, args.rows_per_file), f.seed)
        for f in default_file_specs(nrows=args.rows_per_file)
    ]
    cfg = DataConfig(
        data_dir=args.data_dir,
        files=files,
        normal_source=[FileSpec("Monday-WorkingHours.pcap_ISCX.csv", args.normal_rows, args.seed)],
        class_preset=args.class_preset,
        normal_ratio_in_train=args.normal_ratio,
        min_attack_rows=args.min_attack_rows,
        max_features=args.max_features,
        seed=args.seed,
    )

    t0 = time.time()
    print(f"[1/2] 解析 CSV（每个文件最多 {args.rows_per_file} 行）...")
    ds = build_dataset(cfg)
    print(f"      用时 {time.time() - t0:.1f}s\n")
    print(ds.summary())

    print(f"\n[2/2] 保存到 {args.out} ...")
    save_bundle(ds, args.out)
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"      完成：{args.out} ({size_mb:.1f} MB)")
    print(
        "\n下一步：\n"
        f"  python train.py --data bundle --bundle {args.out} --detector if --episodes 200 --plots\n"
        f"  python train.py --data bundle --bundle {args.out} --detector kalman --episodes 200 --plots\n"
        f"  python train.py --data bundle --bundle {args.out} --detector autoencoder --episodes 200 --plots"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
