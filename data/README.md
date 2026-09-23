# 数据层说明

本目录只保留**可复现所需的元信息**，大文件不入 git（见根 `.gitignore`）。

```
data/
├── README.md               ← 本文件
├── manifest.json           来源、维度、行数、sha256、生成脚本
├── cicids_sample.npz       ★ 规范数据包（协议 v1）—— 由脚本生成，不入 git
├── cicids_sample.meta.json 生成它的完整配置
├── raw/
│   ├── cicids2017/         原始 CICIDS2017 CSV（8 个，约 844 MB）
│   └── cicids2017-parquet/ network 方向自建的 77 维 parquet（约 258 MB）
└── cache/                  历史/对照用数据包（旧口径，仅供新旧对照）
```

## `raw/` 是 junction（目录联接），不是真实副本

合并时用 `New-Item -ItemType Junction` 把原始数据挂进来，**零额外磁盘占用**，
原目录（`大三/网安/chethuhn/...` 与 `大三/网安/network/dataset/`）保持不动。

换机器 / 重新获取数据时，二选一：

```powershell
# ① 重新下载官方数据（需要网络）
python scripts/prepare_data.py --data-dir data/raw/cicids2017 --out data/cicids_sample.npz

# ② 把已有数据挂进来
New-Item -ItemType Junction -Path data\raw\cicids2017 -Target <你的 CICIDS2017 CSV 目录>
```

也可以用环境变量整体重定向数据目录（CI / 多机器时用）：

```powershell
$env:NETSEC_DATA_DIR = "E:\datasets\netsec"
```

## 规范数据包（协议 v1）怎么生成

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m ae_repro.data_prep          # 默认就写到 data/cicids_sample.npz
```

产物形状（与 `data/cicids_sample.meta.json` 逐项对应）：

| 字段 | 形状 | 说明 |
|---|---|---|
| `X_normal_train` | (60000, 78) | 良性流量，**只有它**用于检测器 fit 与 min-max 统计 |
| `X_train` / `y_train` | (36875, 78) | 带标签训练集（MADDPG 交互），良性下采样到 30% |
| `X_test` / `y_test` | (25000, 78) | 评估集，**已排除 Monday 行**（协议 v1 的关键修复） |
| `feature_names` | (78,) | 与平台 `taxonomy` / `INTERFACE.md` 一致 |
| `class_names` | (6,) | BENIGN / DDoS / PortScan / DoS / WebAttack / Other |

> ⚠️ 旧数据包（平台 `data/cicids_sample.npz`，29000 行）里有 **4066 行与
> `X_normal_train` 逐位重复**（14.02%），即检测器自己拟合过的样本进了评估集。
> 协议 v1 已修复到 0.70%。详见 `PROTOCOL.md` 与根目录 `GAP_ANALYSIS.md`。
