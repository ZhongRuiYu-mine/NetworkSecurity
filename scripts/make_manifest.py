# -*- coding: utf-8 -*-
"""生成 data/manifest.json + data/checksums.sha256（记录来源与校验和，便于复现）。"""
import hashlib
import json
import os
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "data")
RAW_CSV = os.path.join(DATA, "raw", "cicids2017")
RAW_PQ = os.path.join(DATA, "raw", "cicids2017-parquet")


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def entries(folder, exts):
    out = []
    if not os.path.isdir(folder):
        return out
    for name in sorted(os.listdir(folder)):
        p = os.path.join(folder, name)
        if not os.path.isfile(p) or not name.lower().endswith(exts):
            continue
        out.append({"file": name, "bytes": os.path.getsize(p), "sha256": sha256(p)})
    return out


def main():
    csvs = entries(RAW_CSV, (".csv",))
    pqs = entries(RAW_PQ, (".parquet",))
    bundle = os.path.join(DATA, "cicids_sample.npz")
    meta = os.path.join(DATA, "cicids_sample.meta.json")

    import numpy as np
    shapes = {}
    if os.path.exists(bundle):
        d = np.load(bundle, allow_pickle=True)
        shapes = {k: list(d[k].shape) for k in
                  ("X_normal_train", "X_train", "y_train", "X_test", "y_test")}
        class_names = [str(c) for c in d["class_names"]]
        d.close()
    else:
        class_names = []

    def link(p):
        try:
            return os.path.realpath(p) if os.path.islink(p) or os.path.isdir(p) else p
        except Exception:
            return p

    manifest = {
        "dataset": "CICIDS2017",
        "repo": REPO,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "protocol": "v1（见 PROTOCOL.md）",
        "layout": {
            "raw_csv_dir": {"path": "data/raw/cicids2017",
                            "mounted_as": "junction",
                            "target": link(RAW_CSV),
                            "note": "合并时用 junction 挂载，零额外磁盘占用；原目录未改动"},
            "raw_parquet_dir": {"path": "data/raw/cicids2017-parquet",
                                "mounted_as": "junction",
                                "target": link(RAW_PQ),
                                "note": "network（ICPS）方向自建的 77 维 parquet，与 78 维 CSV 版本不同"},
            "canonical_bundle": "data/cicids_sample.npz",
            "canonical_meta": "data/cicids_sample.meta.json",
        },
        "regenerate": {
            "bundle": "python -m ae_repro.data_prep",
            "checksums": "python scripts/make_manifest.py",
            "env_override": "NETSEC_DATA_DIR（整体重定向 data/）、NETSEC_RAW_DIR（只重定向原始 CSV）",
        },
        "canonical_bundle": {
            "path": "data/cicids_sample.npz",
            "bytes": os.path.getsize(bundle) if os.path.exists(bundle) else None,
            "sha256": sha256(bundle) if os.path.exists(bundle) else None,
            "meta_sha256": sha256(meta) if os.path.exists(meta) else None,
            "shapes": shapes,
            "class_names": class_names,
            "normalization": "Min-Max，统计量仅来自 Monday 良性流量（论文 Eq.1）",
            "eval_protocol": "评估集已排除 Monday 行（检测器拟合数据），避免自评",
        },
        "raw_csv": {"n_files": len(csvs), "total_bytes": sum(e["bytes"] for e in csvs),
                    "files": csvs},
        "raw_parquet": {"n_files": len(pqs), "total_bytes": sum(e["bytes"] for e in pqs),
                        "files": pqs,
                        "known_issues": [
                            "PortScan 仅 1956 条（官方 158930，manifest 自述疑似被抽样）",
                            "Heartbleed 11 条、Infiltration 36 条、Bot 稀疏",
                            "77 维（含 Protocol），与 78 维 CSV 版本不可混用",
                        ]},
    }

    with open(os.path.join(DATA, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    lines = []
    for e in csvs:
        lines.append(f"{e['sha256']}  raw/cicids2017/{e['file']}")
    for e in pqs:
        lines.append(f"{e['sha256']}  raw/cicids2017-parquet/{e['file']}")
    if manifest["canonical_bundle"]["sha256"]:
        lines.append(f"{manifest['canonical_bundle']['sha256']}  cicids_sample.npz")
    with open(os.path.join(DATA, "checksums.sha256"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"CSV {len(csvs)} 个 / {manifest['raw_csv']['total_bytes']/2**20:.1f} MB")
    print(f"parquet {len(pqs)} 个 / {manifest['raw_parquet']['total_bytes']/2**20:.1f} MB")
    print("规范数据包 sha256:", manifest["canonical_bundle"]["sha256"])
    print("已写入 data/manifest.json 与 data/checksums.sha256")


if __name__ == "__main__":
    main()
