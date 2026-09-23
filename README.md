# netsec —— 网络安全大作业（四方向合并仓库）

CICIDS2017 多机制入侵检测 + 可插拔检测器契约 + MADDPG 多智能体响应 + ICPS 两层融合。

> 本仓库由四个平行目录（`chethuhn` / `network` / `safenetwork` / `ae_repro`）合并而成。
> 合并范围、目录映射、每个非平凡决定见 **`docs/DECISIONS.md`**；
> 曾经的缺口清单与证据见 `GAP_ANALYSIS.md`（在原工作区 `大三/网安/`）。

---

## 1. 30 秒上手

```powershell
$env:PYTHONPATH = "$PWD\src"      # 脚本自带引导块，这一步其实可省
pip install -r requirements.txt

# ① 平台 + 契约自检（约 1 分钟；退出码 0 = 环境可用）
python scripts/verify_pipeline.py
python -m agentenvs.detectors.templates

# ② 生成规范数据包（协议 v1，约 20 秒；需要 data/raw/cicids2017 有原始 CSV）
python -m ae_repro.data_prep

# ③ ★ 四种检测机制同口径对比（一条命令产出主表）
python scripts/eval_all_detectors.py

# ④ 端到端策略训练（MADDPG + 指定检测器）
python scripts/train.py --data bundle --detector if --episodes 200 --out results/if
```

## 2. 目录结构

```
netsec/
├── README.md / PROTOCOL.md / docs/        文档；PROTOCOL.md 是唯一评测口径来源
├── src/
│   ├── agentenvs/        ★ MADDPG 多智能体环境 + 可插拔检测器
│   │   └── detectors/        base(契约) / if / kalman / autoencoder / ae_paper / null / templates
│   ├── algorithms/       MADDPG（CTDE）、Actor/Critic、回放池
│   ├── utils/            指标与分层抽样
│   ├── ae_repro/         论文复现版 AE（AUTO.pdf）+ EVT/GPD 阈值标定 + 契约自检
│   └── icps_detection/   ICPS 两层融合（物理层状态估计 + DPNet 网络层）
├── scripts/              所有可执行入口（自带 sys.path 引导）
├── data/                 数据层（大文件不入 git，见 data/README.md）
├── results/              产物：对比表、图表、策略权重
├── docs/reports/         各方向报告（IF docx / ae_repro 报告 …）
├── legacy/chethuhn-IF/   历史独立脚本，仅供参考，不纳入主路径
└── tests/                冒烟测试（pytest）
```

## 3. 四种检测机制，一个契约

任何检测器只要实现 `DetectorBase.fit()` / `score_batch()`，就能换进环境而**不改**
MADDPG 与环境代码。四种机制统一 `embed_dim=4`（`signal_dim=7`），观测维度一致：

| 注册 key | 机制 | 打分 | 阈值（主口径） | 备注 |
|---|---|---|---|---|
| `if` | Isolation Forest | `-score_samples` | 良性分位数 / F1 扫描 | 轻量、可解释 |
| `kalman` | Kalman 滤波 | 逐维 NIS 新息 | 同上 | 有状态、流式；`mode=frozen/decay/forget` |
| `autoencoder` | 平台内置 AE | 平均重构误差 | 同上 | 64-32 / latent 8 / 原始 MSE |
| `ae_paper` | 论文复现版 AE | 逐特征标准化残差 | 良性分位数 / **EVT-GPD** | 32-16-8 / latent 4；`ae_repro` 交付 |
| `null` | 空检测器 | 恒 0 | — | 消融 baseline |

> `ae_paper` 的接入方式见 `docs/DECISIONS.md` §3.2：它同时存在于
> `src/agentenvs/detectors/ae_paper_detector.py`（供平台使用）与
> `src/ae_repro/ae_detector.py`（供独立复现使用），契约检查优先用**真契约**。

## 4. 评测口径（★ 引用数字前必读）

**`PROTOCOL.md` 是唯一口径来源。** 两条硬规则：

1. **评估集不得包含检测器拟合过的样本。** 协议 v1 数据包里
   `X_test ∩ X_normal_train` 逐位重复率 **0.70%**；合并前平台数据包是 **14.02%**
   （4066/29000）——旧的"三方对比表"就是在这个自评口径下得到的。
2. **主口径 = 良性分位数（α=0.05）**，`fit()` 只喂良性流量、不传 `y_test`。
   F1 扫描（用测试标签选阈值）只能作为**乐观上界**并列报告。
   同一个 IF 在两套口径下 F1 是 **0.591 vs 0.465**。

主表：`results/detector_comparison.md`（`python scripts/eval_all_detectors.py` 生成）。

## 5. 数据

`data/raw/` 用 junction 挂载原始 CICIDS2017（约 1.1 GB，零额外磁盘占用，不入 git）。
换机器 / 重新获取见 `data/README.md`；`data/manifest.json` 记录来源、形状、sha256。

## 6. 已知边界（诚实清单）

| 项 | 状态 |
|---|---|
| Kalman / AE / null 的**端到端策略训练** | ❌ 未跑（只有 IF 有策略权重），见 `PROTOCOL.md` §6 |
| ICPS 融合层接入平台契约 | ❌ 未做（物理量测与 78 维流特征语义不同，需专门设计） |
| ICPS 融合的 oracle 泄漏 | ❌ 未修（`run_fusion.py` 仍用干净真值作参考状态） |
| 带符号 embedding vs `normalize_obs` 裁剪 | ❌ 未裁决（`PROTOCOL.md` §6 第 1 条） |
| ICPS 全链路可跑 | ❌ 本机缺 `pandapower`/`pandera`/`geojson` |
| Kalman / AE / ICPS / MADDPG 报告 | ❌ 只有检测器对比表 + IF docx + ae_repro 报告 |
| LICENSE | ❌ 待组内决定 |

## 7. 依赖

见 `requirements.txt`（统一锁定）。核心：numpy / pandas / scikit-learn / scipy /
matplotlib / joblib / torch；ICPS 方向额外需要 pyarrow + pandapower。
