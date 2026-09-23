# PR：四方向合并为统一仓库 + 评测协议 v1

> 直接用这个文件当 PR 描述（标题用下面第一个 H1 的下一行）。
> 分支：`merge/netsec-protocol-v1`　目标：`main`

**建议标题**：`merge: 四方向合并为统一仓库（src/ 布局）+ 评测协议 v1`

---

## 这个 PR 做了什么

把四个平行目录（`chethuhn` / `network` / `safenetwork` / `ae_repro`）合并成一个仓库，
并落实**统一评测协议 v1**。原四个本地目录保持不动。

### 布局调整（纯迁移，无逻辑改动）

| 原路径 | 新路径 |
|---|---|
| `agentenvs/`、`algorithms/`、`utils/` | `src/` 下同名包 |
| `train.py`、`prepare_data.py`、`verify_pipeline.py` | `scripts/` |
| `README.md` | 项目级 README（平台 README 移到 `docs/README_platform.md`） |
| `docs/requirements.txt` | `docs/requirements_platform.txt`（标注为历史文件） |

`agentenvs/` → `src/agentenvs/` 等文件在 diff 里是 **100% 相似度的 rename**，可直接跳过审阅。

### 新增

| 文件 | 作用 |
|---|---|
| `PROTOCOL.md` | **唯一评测口径来源**（数据/阈值/维度/指标/未裁决议题） |
| `src/ae_repro/` | 论文复现版 AE（`AUTO.pdf`）+ EVT/GPD 阈值标定 + 契约自检 |
| `src/icps_detection/` | ICPS 两层融合（物理层状态估计 + DPNet 网络层） |
| `src/agentenvs/detectors/ae_paper_detector.py` | `ae_paper` 正式注册进 `DetectorBase` 契约 |
| `scripts/eval_all_detectors.py` | 一条命令产出同口径对比表 |
| `scripts/make_manifest.py` | 数据清单 + sha256 校验和 |
| `scripts/run_all.ps1` | 一键自检 + 对比 |
| `tests/test_smoke.py` | 10 项冒烟测试（契约/协议/脚本） |
| `docs/DECISIONS.md`、`docs/ACCEPTANCE.md` | 合并决策与非平凡决定的记录、验收证据 |
| `data/README.md`、`results/README.md` | 数据层与产物说明 |

## 协议 v1 的两个关键修复

1. **评估集不再包含检测器拟合过的样本。** 旧数据包测试集 29000 行里有
   **4066 行（14.02%）**与检测器拟合集逐位重复；协议 v1 降到 **176/25000 = 0.70%**。
2. **主口径 = 良性 95% 分位阈值**（`fit()` 只喂良性、不传 `y_test`）；
   F1 扫描（用测试标签选阈值）只作**乐观上界**并列报告。
   同一个 IF 在两套口径下 F1 = **0.591 vs 0.465**，混用会让结论失真。

### 主口径结果（`results/detector_comparison.md`）

| 机制 | P | R | F1 | ROC-AUC | FPR | FP‰ | P@1% |
|---|---|---|---|---|---|---|---|
| `ae_paper` | 0.7638 | 0.5689 | **0.6521** | **0.8474** | 6.12% | 61.2 | 0.0858 |
| `autoencoder` | 0.7168 | 0.5800 | 0.6412 | 0.8021 | 7.97% | 79.7 | 0.0684 |
| `kalman_frozen` | 0.6541 | 0.5205 | 0.5797 | 0.7841 | 9.58% | 95.8 | 0.0521 |
| `kalman_decay` | 0.6515 | 0.5168 | 0.5764 | 0.7795 | 9.62% | 96.2 | 0.0515 |
| `if` | 0.6745 | 0.3445 | 0.4560 | 0.7327 | 5.79% | 57.9 | 0.0567 |
| `null`（消融） | 0.2581 | 1.0000 | 0.4103 | 0.5000 | 100% | 1000.0 | 0.0100 |

口径变化**直接改变排名**：旧的 F1 扫描口径给的是 `AE 0.6689 > Kalman 0.6348 > IF 0.5910`，
主口径下 IF 掉到 0.4560、被 `kalman_frozen` 反超（IF 的旧数字受益于标签选阈）。
`ae_paper` 的 ROC-AUC 0.8474 与 `ae_repro` 独立报告的 0.8451 一致，说明合并没改变模型行为。

## 顺手修掉的缺陷

| 缺陷 | 影响 |
|---|---|
| `templates.py` 在 Windows GBK 控制台打印 `✅` 抛 `UnicodeEncodeError` | 自检**假失败**（功能全过但退出码 1） |
| `train_cicids.py` 落盘缺 `scaler_mu`/`scaler_sigma` | `fusion.NetworkProbe` 读不了；重跑训练会覆盖成坏 checkpoint |
| 产物路径散落（`scripts/outputs/`、硬编码数据包路径） | 统一到 `results/`、`data/` |
| `.gitignore` 用了行尾注释 → 规则失效 | 差点把 1.1GB 原始数据提交进去（1139MB → 2.46MB） |

## 验收证据（`docs/ACCEPTANCE.md`）

```
python scripts/verify_pipeline.py          → exit 0「全部自检通过」
python -m agentenvs.detectors.templates    → exit 0「模板自检通过」
python -m ae_repro.verify_contract         → 35/35 通过（契约来源=真 agentenvs.detectors.base）
python -m pytest tests -q                  → 10 passed
python -m ae_repro.data_prep               → 生成规范数据包（19.7s）
python scripts/eval_all_detectors.py       → 产出 results/detector_comparison.{md,json}
```

## ⚠️ 与 `main` 的 2 处分叉需要作者确认

逐文件 SHA256 比对：24 个文件中 13 个逐字节一致（只差换行符）、9 个差异全部来自本次布局调整，
**2 个在合并前就已分叉**：

| 文件 | `main`（09-17） | 本 PR（来自 09-21 快照） |
|---|---|---|
| `agentenvs/detectors/autoencoder_detector.py` | `self.embed_dim = 0` | `int(embed_dim) if embed_dim is not None else 0` |
| `algorithms/maddpg.py` | 无调试输出 | 多 5 行 `[DEBUG] q_det …` |

`main` 的那两个版本已**逐字节备份**在 `legacy/diverged-from-remote/`
（含完整 diff 与 3 种恢复方式，其中一种不需要懂 git）。

**需要作者确认 `embed_dim` 那一行哪个为准**：`main` 的写法会忽略传入的 `embed_dim`，
统一 `embed_dim=4` 时 AE 的 `signal_dim` 会掉到 3（其余机制为 7），热替换会报 `ValueError`。
若 `main` 版本为准，在本分支补一个 commit 改回去即可。

## 建议的审阅方式

1. 先看 `PROTOCOL.md`（口径）和 `results/detector_comparison.md`（结果），再看代码；
2. diff 里 `agentenvs/` → `src/agentenvs/` 之类是 100% rename，可跳过；
3. 真正需要读的新增逻辑只有 `scripts/eval_all_detectors.py`、`src/ae_repro/`、
   `src/agentenvs/detectors/ae_paper_detector.py`；
4. `docs/DECISIONS.md` 记录了每个非平凡决定，以及**刻意没做**的事。

## 本 PR 未覆盖

- Kalman / autoencoder / null 的端到端策略训练（仍只有 IF 有策略权重）
- ICPS 融合层接入契约、oracle 泄漏修复（`run_fusion.py` 仍用干净真值作参考状态）
- 「带符号 embedding vs `normalize_obs` 裁剪」的裁决（需组内决定）
- ICPS 全链路运行（缺 `pandapower`/`pandera`/`geojson`）
- Kalman / AE / ICPS / MADDPG 报告；LICENSE
