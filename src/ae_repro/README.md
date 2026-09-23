# 自编码器检测机制复现包（`ae_repro`）

> 小组任务 · 复现论文里的 **Autoencoder** 部分，接到 `NetworkSecurity` 平台的
> `DetectorBase` 契约上，数据集只用 **chethuhn / CICIDS2017**。

---

## 0. 先看这个：我做了什么

| 交付物 | 位置 | 说明 |
|---|---|---|
| **复现报告** | `results/REPORT.md` | 主结果 + 消融 + 与论文差异，可直接贴进小论文 |
| **原始指标** | `results/repro_report.json` | REPORT.md 的机器可读版，所有数字的来源 |
| **4 张图** | `results/fig1..4*.png` | 训练曲线 / ROC+PR / 消融对比 / 逐特征归因 |
| **检测器（贴进仓库）** | `ae_detector.py` + `ae_core.py` | 拷两个文件进 `agentenvs/detectors/` 即可用 |
| **已训练好的检测器** | `artifacts/ae_paper.npz|.json` | 直接 `load()`，不用重训 |
| **契约自检** | `verify_contract.py` | 34 项全过，对着 `INTERFACE.md` 第 8 章清单逐条测 |
| **实验设计探针** | `results/design_probe/*.json` | 选型依据（为什么这么定超参） |
| **基率校正指标** | `results/REPORT.md` §2/§3 | 提案第 4 节要求的 **FP‰** 与 **Precision@1%** |
| **新旧评估口径对照** | `results/repro_report.old_protocol.json` | 修复前那一版（测试集含检测器训练数据），供对照 |

### 0.5 本次修复（code review 结论）

| # | 问题 | 修法 | 影响 |
|---|---|---|---|
| 1 | `DEFAULT_CSV_DIR` 多了一层 `chethuhn`，默认路径不存在 | 改成 `chethuhn/network-intrusion-dataset/...` | `python -m ae_repro.data_prep` 现在能直接跑（默认路径实测存在） |
| 2 | **测试集里混着检测器的训练数据**：`file_specs()` 的 Monday 项与 `normal_source()` 抽到同一批行，其中 1/5 进测试集 → 旧测试集 **32.68%** 的样本是检测器逐位见过的 | 新增 `CicidsConfig.eval_exclude_normal_source=True`，切分时把 Monday 行排除在测试集外（可用 `--keep-monday-in-test` 复现旧口径） | 测试集 37000 → **25000**，攻击占比 17.4% → 25.8%；FPR 被低估的问题消失（详见 `results/REPORT.md` §1.1） |
| 3 | `trust` 用 std 标定，被极端良性样本撑大（std=22.49 vs 中位数 0.25），trust 被压成常数 ~0.5，观测通道没信息量 | 改用稳健尺度 **IQR/1.349**（`score_robust_scale`），中心与语义不变 | trust 的 5%~95% 区间从 **[0.49, 0.66]** 变成 **[0.10, 1.00]**；告警样本 trust 均值 0.9999、未告警 0.7971 |
| 4 | 报告缺提案第 4 节要求的 FP‰ 与 Precision@1% | `binary_metrics()` 增加 `fp_per_1000_flows` / `precision_at_1pct` / `precision_at_0.1pct`，报告与 `train_detector` 落盘都会输出 | 主配置：FP‰ = **63.2**、P@1% = **0.0834**（旧口径 60.8 / 0.0865） |

> 顺带纠正了三处写错的表述：① FPR 是**只在良性样本上算**的指标，与攻击占比无关，被基率压垮的是 precision（不是 FPR）；② "每天重标定后误报率 5%" 是阈值定义的构造结果，不构成"阈值可校准"的证据；③ 消融里引用的 `probe2_training_length.json` 数字其实在 `probe1_transform_x_score.json`，且"原始 MSE 极不稳定"这个结论与最终表矛盾（现在按实测改写）。

### 0.6 交接须知（给组长/集成同学）

**这份交付已经能做什么**：独立复现 + 契约适配 + 可解释性归因 + 已训练好可直接 `load()` 的检测器
+ 34 项契约自检。指标经过独立复核（用纯 numpy 重写前向，复算结果与落盘 `metrics.json` 逐位一致）。

**怎么跑**（只需要 numpy/pandas/scikit-learn/torch/matplotlib）：

```powershell
python -m ae_repro.verify_contract        # 34 项契约自检（~30 秒，不需要平台仓库）
python -m ae_repro.experiments --quick    # 只跑主配置（~6 分钟）
python -m ae_repro.experiments            # 全套 9 个实验（~35 分钟）
python -m ae_repro.make_report --report results/repro_report.json
```

数据包 `_cache/cicids_sample.npz` 已经带在包里，**不需要原始 CSV 也能跑**；
若要重建数据包，需要 `ae_repro/` 与 `chethuhn/`（CICIDS2017 原始 CSV）同级，
否则用 `--csv-dir` 指定路径。

**交接时必须一起说明的三个"未验证 / 待对齐"事项**（否则组里拿去接平台会踩坑）：

1. **还没在真实的 `agentenvs` 仓库里跑过**。"拷两个文件 + 加一行 import 就能被
   `make_detector` / CLI 认"这条路径**未经实测**；`_contract.py` 是 `base.py` 的副本，
   也无法核对是否同步。拿到仓库后第一件事就是跑一次
   `python -m agentenvs.detectors.templates` 或平台的 `verify_pipeline.py`。
2. **阈值口径和平台约定不一致**：本检测器拿到 `y_test` 后仍走**良性分位数**阈值，
   而 `INTERFACE.md` §2.8 的约定是"给了标签就走 F1 扫描"。三方对比必须统一，
   否则 AE 与 IF/Kalman 是在不同工作点上比。
3. **三方对比要指定用哪个 AE**：平台内置的 `ae`（64-32 / latent 8 / 原始 MSE /
   embedding 78 维）与本包的 `ae_paper`（32-16-8 / latent 4 / 逐特征标准化残差 /
   embedding 4 维）是两个不同模型。另外，**平台的分层切分是否也把 Monday 排除在
   评估集之外**需要数据组确认——如果不排除，IF/Kalman 的数字也是自评口径。

依赖环境：本包在 Python 3.12.6 + torch 2.14.0+cpu 上重跑并验证过；
更早那一版报告是在 Python 3.13.7 + torch 2.13.0+cpu 上跑的。
换 torch 版本可能带来微小数值差异（见报告里的 environment 表）。

---

## 1. 选哪篇论文？为什么

两篇候选里我选 **`AUTO.pdf`** 作为主复现对象：

| | AUTO.pdf | Autoencoder-Based_Anomaly_Detection.pdf |
|---|---|---|
| 主题 | 区块链元宇宙交易 → Hybrid AE–IF | IEC 61850 GOOSE 变电站 → 双视图 AE |
| AE 的输入 | **通用表格特征**（选出 `F'` 后喂网络） | **GOOSE 协议专有特征**：`stNum`/`sqNum` 序列语义 + 重传时序 |
| 能不能落到 CICIDS2017 | ✅ 能：CICIDS 就是一张 78 维流特征表，直接当 `F'` 用 | ❌ 不能：CICIDS 的流特征里**根本不存在** stNum/sqNum/GOOSE 重传间隔，强行"复现"等于自己造一套特征 |
| 阈值口径 | 固定 0.52（作用于自己 min–max 后的融合分数） | EVT/GPD 尾部建模（更严谨） |
| 报告指标 | Table 1：AE 单独 0.949/0.82/0.79/0.80 | 按攻击类型的 F1 |

**结论**：只有 AUTO.pdf 的 AE 能**原样**落到 CICIDS2017 上，所以主复现它。
但 B 论文有一个东西明显更好——**EVT/GPD 阈值标定**和**逐特征重构误差归因**。
这两点不依赖 GOOSE 特征，可以无损移植，所以作为**增强项**一起实现了（见 §4）。

顺带说明：平台仓库 `agentenvs/detectors/autoencoder_detector.py` 的 docstring 里
本来就引用了这两篇 pdf，所以这个选型也和平台既有方向一致。

---

## 2. 复现协议（和平台完全对齐）

严格照 `agentenvs/data_loader.py` 的 `DataConfig` 默认值做，这样结果能和
IF / Kalman 三方横向对比：

| 项 | 取值 |
|---|---|
| 良性训练集 | `Monday-WorkingHours` 采样 60000 行 |
| 带标签数据 | 8 个文件（含 Monday），每文件 20000 行（Infiltration 文件 5000 行） |
| **评估集** | **排除 Monday 行**（`eval_exclude_normal_source=True`）：Monday 既是检测器的拟合数据，就不能再进测试集 |
| 归一化 | **只用良性训练集**的 min–max → `[0,1]`（论文 Eq.1） |
| 类别空间 | `BENIGN / DDoS / PortScan / Bot / DoS / WebAttack / Other`（Bot 样本 <200 并入 Other） |
| 划分 | 按类别分层，`test_size=0.2` → 测试集 **25000 行**（Friday 12156 / Thursday 4868 / Tuesday 3982 / Wednesday 3994） |
| **检测器约束** | **只在良性流量上训练**；阈值只用**良性分数**标定，测试标签只用于评估 |

> ⚠ **口径别搞混**：测试集是分层抽样的，攻击占比 25.8%，远高于线上。
> 受基率影响的是 **precision**，所以报告里的 P=0.758 会**高估可部署性**，
> 要引用 `P@1% = 0.0834`。而 **FPR / FP‰ 只在良性样本上计算，与攻击占比无关**，
> 可以直接外推（63.2 FP / 1000 条流）。修复前的旧口径把这两件事说反了，已在报告里改正。

---

## 3. 主结果（主口径：良性 95% 分位阈值；**测试集已排除检测器训练数据**）

| 指标 | 本复现（CICIDS2017） | 论文 AUTO.pdf Table 1（AE 行） |
|---|---|---|
| ROC-AUC | **0.8451** | 0.949 |
| PR-AUC | **0.6617** | 未报告 |
| Precision（测试集内 25.8% 攻击占比） | **0.7580** | 0.82 |
| Recall | **0.5689** | 0.79 |
| F1 | **0.6500** | 0.80 |
| **FP‰（每千条误报）** | **63.2** | 未报告 |
| **Precision@1%（基率校正）** | **0.0834** | 未报告 |
| 阈值 | 良性 95% 分位 = 2.3092 | 0.52（其自身归一化分数上） |
| 参数量 / 训练 / 推理 | 6274 参数 / ~360 s (CPU) / 0.0016 ms 每条 | 未报告 |

**新旧口径对照**（同一模型、同一阈值口径，唯一差别是测试集是否包含检测器的训练数据）：

| 口径 | 测试集 | 与训练集逐位重复 | ROC-AUC | PR-AUC | P | F1 | FPR | FP‰ | P@1% |
|---|---|---|---|---|---|---|---|---|---|
| 旧（自评） | 37000 | 32.68% | 0.8514 | 0.6014 | 0.6646 | 0.6137 | 0.0608 | 60.8 | 0.0865 |
| **新（已修复）** | 25000 | 0.70% | 0.8451 | 0.6617 | 0.7580 | 0.6500 | 0.0632 | 63.2 | 0.0834 |

自评口径会把 precision 抬高、把 FPR 压低；**论文请引用新口径**。

**为什么不能直接比数字**（报告第 5 节有完整列表）：
数据集不同、特征空间不同、阈值口径不同、评估协议不同（论文 10 折交叉验证 vs
本复现跨天测试）。CICIDS2017 有很强的**跨天分布漂移**，跨天测试天然比同分布
交叉验证低一截。**正确做法**是把本复现当作"该 AE 在 CICIDS 跨天协议下的基线"，
而不是"没复现出论文的 0.949"。

### 关键诊断：瓶颈在阈值，不在模型

| 阈值口径 | 阈值 | P | R | F1 | FPR | FP‰ | P@1% |
|---|---|---|---|---|---|---|---|
| `percentile`（主口径，同平台内置 AE） | 2.3092 | 0.7580 | 0.5689 | 0.6500 | 0.0632 | 63.2 | 0.0834 |
| `evt`（B 论文的 GPD 标定） | 4.9539 | 0.8570 | 0.5497 | 0.6698 | 0.0319 | 31.9 | **0.1482** |
| `best_f1`（测试集扫描，**乐观上界**） | 3.6312 | 0.8295 | 0.5649 | **0.6721** | 0.0404 | 40.4 | 0.1238 |
| `per_day_calib`（每天用当天良性重标定，仅诊断） | — | 0.7911 | 0.5458 | 0.6459 | 0.0501 | 50.1 | 0.0991 |

- **EVT 阈值把 F1 从 0.650 提到 0.670、FPR 从 6.3% 降到 3.2%，P@1% 从 0.083 提到 0.148**，
  且它只用良性分数，可上线——这是移植 B 论文 EVT 的直接收益。建议**用 EVT 口径作为上线口径**。
- `best_f1` 和 `evt` 很接近（F1 0.6721 vs 0.6698），说明 EVT 标定已经接近最优阈值，
  **阈值不是主要矛盾**。
- 跨天漂移的直接证据是**阈值本身差多少**：当天重标定阈值 2.11~6.47（相差 **3.1 倍**）。
  注意"当天误报率≈5%"是构造出来的（阈值就是当天良性分数的 95% 分位），
  **不构成"阈值可校准"的证据**；真正的结论是"一天标定、多天使用"会让
  Wednesday 这类漂移大的日子实际误报率冲到 10.7%。

### 按攻击类别（主口径阈值）

| 类别 | 样本数 | 检出率 |
|---|---|---|
| DoS | 2929 | **0.6859** |
| DDoS | 2483 | **0.6677** |
| PortScan | 743 | 0.0054 |
| WebAttack | 99 | 0.0000 |
| Other（**实测全部是周二 FTP/SSH-Patator 暴力破解**） | 199 | 0.0000 |
| BENIGN | 18547 | 误报率 0.0632 |

**弱项要说清楚**：PortScan / WebAttack 检出率≈0，而 `Other` 那一类实际是**暴力破解**
（Bot 因样本数 <200 被并入 Other，Infiltration 没进测试集），也就是说
**暴力破解 100% 漏检**。
原因是这些流在特征空间里离 Monday 良性太近，自编码器"能重构它们"。
这是**重构式无监督方法的固有短板**，正好是"AE 需要和 IF / Kalman 互补"的论据，
建议如实写进论文，别藏。

### 逐特征归因 Top5（论文第 7 步）

`Init_Win_bytes_forward` > `URG Flag Count` > `Destination Port` >
`Idle Min` > `SYN Flag Count`

### 送进环境的 embedding（`embed_dim=4`）

取标准化残差最大的 4 维：`Destination Port, URG Flag Count,
Init_Win_bytes_forward, Idle Min`（带符号，让 critic 知道**偏离方向**而不只是幅度）。
注意：如果平台 `normalize_obs=True` 会把整个观测裁剪到 `[0,1]`（`INTERFACE.md` §3.3），
带符号的 embedding 会被压掉负号——这一条要向集成同学确认。

---

## 4. 消融结论（`results/REPORT.md` 第 6 节）

| 变体 | ROC-AUC | PR-AUC | F1 | 结论 |
|---|---|---|---|---|
| **主配置**（min–max 直入 + 逐特征标准化打分 + 浅层解码 + BN + dropout0.1） | 0.8451 | 0.6617 | 0.6500 | 默认 |
| 论文字面实现（min–max 直入 + **原始 MSE** 打分） | **0.8618** | **0.7495** | 0.6299 | **排序能力更强**，但归因/embedding 失真、且对训练轮数敏感 |
| 额外 z-score 输入（B 论文做法） | 0.7478 | 0.6452 | 0.6477 | ❌ AUC 掉 0.097 |
| log1p + z-score 输入 | 0.6461 | 0.5966 | 0.6436 | ❌ AUC 掉 0.199 |
| 去掉 dropout | 0.8417 | 0.6705 | **0.6676** | dropout 主要帮 PR-AUC |
| 对称解码器（论文原话） | 0.8483 | 0.6632 | 0.6500 | 与默认无差别，可用 |
| 去掉 BatchNorm | 0.7968 | 0.6970 | 0.6467 | BN 有用（AUC -0.048） |
| 输出层加 sigmoid | 0.8397 | 0.7406 | **0.6710** | 排序更好但 FPR 升到 7.6% |
| ReLU + 30 轮（贴近平台内置 AE） | 0.8032 | 0.6287 | 0.6459 | 交叉对照 |

**三条硬结论**：

1. **输入不要再标准化一次**。平台已按论文 Eq.1 做过 min–max，核心内部再加
   z-score / log1p 会让 ROC-AUC 掉 0.10~0.20。攻击流量的**幅度**信息被压掉了
   （`design_probe/probe1_transform_x_score.json` 有完整 12 格交叉表，
   但那是**旧口径的设计阶段记录**，只表趋势、不可引用，见 `design_probe/PROVENANCE.md`）。
2. **打分口径是"排序能力 ↔ 可解释性/鲁棒性"的取舍**。论文字面实现（原始 MSE）
   的 AUC/PR-AUC 更高（0.8618 / 0.7495 vs 0.8451 / 0.6617），但它的逐特征残差
   没有可比量纲（归因和 embedding 都会失真），而且对训练预算敏感
   （`design_probe/probe1_transform_x_score.json`：同一配置 76→181 轮时
   PR-AUC 从 0.621 掉到 0.556——同样是**旧口径的设计阶段记录**）。
   逐特征标准化残差在 15~250 轮之间几乎不漂移。
   **只比 AUC/PR-AUC 就该用字面实现；要让 critic 拿到有方向的残差就用主配置。**
   （旧版 README 把这条写成了"原始 MSE 极不稳定"，与最终表矛盾，已按实测改写。）
3. **对称解码器也完全可以**（论文原话就是对称），实测与默认几乎一致。

---

## 5. 组员怎么接（3 步）

### 方案 A：拷进仓库（推荐，之后 CLI 直接认）

```powershell
# 1) 拷两个文件
copy ae_repro\ae_core.py     <repo>\agentenvs\detectors\ae_core.py
copy ae_repro\ae_detector.py <repo>\agentenvs\detectors\ae_paper_detector.py

# 2) 在 <repo>\agentenvs\detectors\__init__.py 加一行
#    from .ae_paper_detector import AePaperDetector

# 3) 跑（注册名 ae_paper / ae_cicids）
& $PY train.py --data bundle --detector ae_paper --embed-dim 4 --out outputs/ae_paper
```

`--embed-dim 4` 是为了和 IF / Kalman 统一观测维度，三方对比时才"只换检测器、
其它全不动"。**别用默认的 78**，那会把 `obs_dim` 撑大、策略权重不通用。

### 方案 B：不改仓库，直接传对象

```python
from ae_repro.ae_detector import AePaperDetector
det = AePaperDetector(embed_dim=4).load("ae_repro/artifacts/ae_paper")   # 已训练好
env = make_env(ds, detector=det, detector_embed_dim=4)
```

### 方案 C：只要指标，不碰代码

直接用 `results/repro_report.json` 里的数字写论文；引 `results/fig1..4*.png` 配图。

### 想自己复跑

```powershell
# 数据包已经生成好了：ae_repro/_cache/cicids_sample.npz（78 维，60000 良性 + 25000 测试）
python -m ae_repro.verify_contract                    # 契约自检（34 项，~30 秒）
python -m ae_repro.experiments --quick                # 只跑主配置（~6 分钟）
python -m ae_repro.experiments                        # 全套 9 个实验（~35 分钟）
python -m ae_repro.make_report --report results/repro_report.json
python -m ae_repro.train_detector                     # 重新训练并落盘 artifacts/
# 想重建数据包（默认已排除 Monday，可用 --keep-monday-in-test 复现旧口径）
python -m ae_repro.data_prep
```

---

## 6. 文件清单

```
ae_repro/
├── README.md               ← 本文件
├── ae_core.py              ★ 纯 numpy/torch 的自编码器核心（可独立使用）
├── ae_detector.py          ★ DetectorBase 适配层（拷进仓库的就是这两个）
├── _contract.py            平台 base.py 的逐字副本（独立自检用，勿改）
├── data_prep.py            复现平台 data_loader 的 CICIDS2017 协议
├── experiments.py          9 个实验 + 消融 + 报告 + 画图
├── train_detector.py       训练并落盘检测器
├── make_report.py          repro_report.json → REPORT.md
├── verify_contract.py      契约自检（对照 INTERFACE.md 第 8 章）
├── results/
│   ├── REPORT.md           ★ 复现报告（给人看的）
│   ├── repro_report.json   原始指标（机器可读）
│   ├── repro_report.old_protocol.json   修复前那一版（新旧口径对照用）
│   ├── fig1_training_curve.png / fig2_roc_pr.png
│   ├── fig3_ablation.png / fig4_attribution.png
│   └── design_probe/       选型探针原始数据
├── artifacts/              ★ 训练好的检测器（.npz 权重 + .json 元信息）
└── _cache/                 数据包 cicids_sample.npz + meta.json
                            以及 cicids_sample.old_protocol.npz（旧口径，对照用）
```

---

## 7. 已知限制 / 下一阶段可做

1. **跨天漂移是主要矛盾**：周一定标的阈值在周三会明显偏松（当天重标定阈值是全局的
   2.8 倍，周三实际误报率 10.7% vs 设计值 5%）。可做的方向：滑动窗口重标定、
   EVT 在线上滚动更新、或按天做域自适应。本包已经把诊断工具和数字准备好了。
2. **要和组长确认平台是否也把 Monday 放进带标签池**。本包默认在**评估集**里排除它
   （`eval_exclude_normal_source=True`）。如果平台的分层切分不做这个排除，
   那 IF / Kalman 的数字同样是自评口径，三方对比之前必须统一——
   这是数据协议层面的事，不是 AE 自己的事。
3. **PortScan / WebAttack 检出率≈0、暴力破解（Other）100% 漏检**，
   这是重构式方法的固有短板。建议论文里把这些类别单独列出来，
   作为"为什么需要多机制互补"的证据。
4. **没做 10 折交叉验证**（论文口径）。如果评审要求"同分布下的复现忠实度"，
   `experiments.py` 里加一个 benign-only 的 K 折即可，协议代码都已就绪。
5. **没实现论文的 Isolation Forest 与 0.6/0.4 加权融合**——按小组分工那部分
   由 IF 负责；融合时注意论文 Table 1 里的融合阈值 0.50 和 AE 单独阈值 0.52
   是两套口径，别混用。
6. **Bot 类别被并进 Other**：20000 行采样下 Bot 只有 40 条，低于平台的
   `min_attack_rows=200`。要单独评估 Bot 得把 `--rows-per-file` 调大重跑。
7. **阈值口径还没和平台内置检测器对齐**：`AePaperDetector.fit()` 收到 `y_test` 后
   仍走良性分位数阈值，而 `INTERFACE.md` §2.8 的约定是"给了标签就走 F1 扫描"。
   如果平台里 IF / Kalman 走的是 F1 扫描，三方的工作点就不可比。
   建议加 `threshold_mode="auto"` 跟随平台口径，或统一都用 `alpha` 分位数。
8. **退化特征是一个尚未引爆的风险**：Monday 里恒为 0 的 10 个特征
   （`Bwd PSH Flags`、`Fwd URG Flags`、`CWE Flag Count`、一批 `* Avg */Bulk`）
   残差标准差被打到 `1e-8` 下限，z 分数因此在数学上无界。新测试集里刚好
   **没有**样本在这些维度上越界（`|z|>1` 的样本 0 条），指标完全不受影响
   （剔除这 10 维后 AUC/P/R/F1/FP 一位不差）；但旧测试集里有 408 条
   `Fwd URG Flags=1` 的样本，单条分数冲到 1e14、把测试集平均分数抬到 6.9e9。
   换数据集或换特征子集后可能重演。稳妥的修法：训练前丢掉常数列，
   或把下限从 `1e-8` 提到稳健下界（如 `max(σ, 1e-3)`），或对 z 做 clip。
   注意 `trust` 的问题**不是**它引起的——用稳健尺度前，剔除这 10 维毫无改善，
   因为病根在良性分数本身的极端尾部（中位数 0.25、最大值 3842）。
