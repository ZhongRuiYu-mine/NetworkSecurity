# 合并记录与工程决策

本文件记录四包合并（`chethuhn` / `network` / `safenetwork` / `ae_repro` → `netsec`）
时做的每一个非平凡决定，以及**刻意没做**的事。

---

## 1. 合并范围

| 决策 | 内容 |
|---|---|
| 范围 | 骨架合并 + 统一评测协议（**不重训 MADDPG**，不改实验结论） |
| 原目录 | **保持不动**（`D:\higher\大三\网安\*`）。本仓库是副本，可随时回退 |
| 数据 | 用 **junction（目录联接）** 挂到 `data/raw/`，零额外磁盘占用；原数据仍在原处 |
| git 历史 | 原四目录都没有 `.git`，因此本仓库是**全新历史**，不保留"哪个方向什么时候加的"信息 |

## 2. 目录映射

| 原路径 | 新路径 |
|---|---|
| `safenetwork/NetworkSecurity-main/agentenvs` | `src/agentenvs` |
| `safenetwork/NetworkSecurity-main/{algorithms,utils}` | `src/{algorithms,utils}` |
| `safenetwork/NetworkSecurity-main/{prepare_data,train,eval_if_only,verify_pipeline}.py` | `scripts/` |
| `safenetwork/NetworkSecurity-main/{README.md,docs/INTERFACE.md}` | `docs/README_platform.md`、`docs/INTERFACE.md` |
| `safenetwork/Isolation Forest.docx` | `docs/reports/Isolation Forest.docx` |
| `ae_repro/*` | `src/ae_repro/*`（`_cache/` → `data/cache/`） |
| `network/icps_detection` | `src/icps_detection` |
| `network/{demo,pipeline_demo,train_cicids,run_fusion}.py` | `scripts/` |
| `network/dpnet_cicids.pt` | `artifacts/dpnet_cicids.pt`（不入 git） |
| `chethuhn/.../IF.py` | `legacy/chethuhn-IF/IF.py`（**仅作历史参照**，见下） |
| `chethuhn/.../*.csv` | 保留原处，`data/raw/cicids2017` junction 指向它 |

## 3. 关键决策

### 3.1 `chethuhn/IF.py` 只做历史参照，不纳入主路径
它是 Monday→Tuesday 的独立脚本（`StandardScaler`、`n_jobs=-1`、无 `DetectorBase`），
与平台口径（60000 良性、min-max、`embed_dim=4`）是**两套实验**。
平台的 `agentenvs/detectors/isolation_forest_detector.py` 才是纳入对比的那一份。
→ 放在 `legacy/` 并在其 README 里写明差异，避免验收时被问"IF 到底交的是哪个"。

### 3.2 `ae_repro/_contract.py` 保留，但自检优先用真契约
`_contract.py` 是平台 `base.py` 的逐字副本。合并后 `verify_contract.py` 改为
**优先 `agentenvs.detectors.base`**，并在开头做一次 AST 漂移检查
（去掉 docstring 后代码等价）；只有脱离仓库单独运行时才退回副本。
→ 消除了 GAP_ANALYSIS §3 第 1 条"副本无法核对是否同步"的风险。

### 3.3 路径收敛成三处
| 位置 | 作用 |
|---|---|
| `src/ae_repro/paths.py` | 仓库路径常量（`REPO_ROOT` / `DATA_DIR` / `RAW_CSV_DIR` / `BUNDLE` …） |
| `scripts/_repo.py` | 脚本侧的同一组常量 |
| 脚本头部的"合并仓库引导"块 | 免设 `PYTHONPATH`，自动把 `src/` 与仓库根加入 `sys.path` |

`agentenvs/data_loader.py` 的 `DEFAULT_DATA_DIR` 改为 `<repo>/data/raw/cicids2017`，
可用环境变量 `NETSEC_RAW_DIR` 覆盖；`ae_repro` 侧可用 `NETSEC_DATA_DIR` 整体重定向。

### 3.4 `algorithms` 与 `utils` 保留原名（已知隐患，刻意未改名）
这两个顶层包名过于通用，`pip install -e .` 后可能与别的项目冲突。改名需要
改动平台内大量 import，属于高风险低收益，**留给下一阶段**；目前用 `src/` 布局
+ 不安装（只靠 `sys.path`）的方式规避。

### 3.5 `torch` 不写死 CUDA 版本
合并前三份 requirements 互相冲突（cu128 / cpu / 未给）。`requirements.txt` 统一为
`torch==2.10.0`（CPU/CUDA 任意），CUDA 安装命令写在文件注释里。

## 4. 顺手修掉的真实缺陷（不在原计划内，但改动小、风险低）

| # | 缺陷 | 修法 | 是否已验证 |
|---|---|---|---|
| 1 | `templates.py` 在 Windows GBK 控制台打印 `✅` 抛 `UnicodeEncodeError`，导致自检**假失败**（退出码 1） | 入口处 `sys.stdout.reconfigure(encoding="utf-8", errors="replace")` | ✅ 实跑，退出码 0 |
| 2 | `train_cicids.py` 落盘的 checkpoint **缺** `scaler_mu/scaler_sigma`，而 `fusion.NetworkProbe` 需要它们 → 重跑训练会覆盖成读不了的权重 | 落盘时补上两个键 + 配置；指标同时写 `artifacts/dpnet_cicids.metrics.json` | ⚠️ 改代码已验证语法/契约，**未重训**（需 231 万行 + GPU） |
| 3 | 产物路径散落（`scripts/outputs/`、`data/cicids_sample.npz` 硬编码） | 统一到 `results/` 与 `data/`，并加 `.gitignore` | ✅ 实跑 |
| 4 | `per_class_report()` 只 print 不返回，指标无法落盘 | 返回 `(acc, macro_f1, per_class)`，由调用方写 JSON | ⚠️ 语法/调用点已验证，未重训 |
| 5 | 训练脚本无 `json`/`os` 导入却被新代码使用 | 补齐导入 | ✅ 语法检查 |

## 5. 与线上仓库 `ZhongRuiYu-mine/NetworkSecurity` 的关系（合并后补记）

线上仓库当时的状态（用**完整** fetch 核对；`--depth 1` 浅克隆会漏掉历史，我第一次就误判成"只有 1 个提交"）：

| 项 | 值 |
|---|---|
| 远端 `main` | `b4cfc87`（2026-09-17 12:11，"Update README.md"） |
| 提交总数 | **3 个**：`c03af5d` Initial commit → `cef1425` Add files via upload → `b4cfc87` Update README.md |
| 文件数 / 体积 | 24 个 / 0.23 MB |
| 布局 | 旧布局：`agentenvs/ algorithms/ utils/ docs/` + 顶层 `train.py prepare_data.py verify_pipeline.py README.md` |

### 5.1 推送策略：只进新分支，不动 `main`

本次合并的提交全部落在新分支 **`merge/netsec-protocol-v1`**（父提交 = `origin/main`）：

- **`main` 一个字节都不动** —— 线上那 24 个文件、3 个提交原样保留；
- 分支的父提交就是 `origin/main`，所以 PR / `git diff origin/main...merge/netsec-protocol-v1`
  能一眼看清"改了哪些、删了哪些、移到哪了"；
- 要回退就删掉这个分支，`main` 不受影响。

### 5.2 逐文件比对结果（远端 24 个文件 vs 本仓库对应文件，SHA256）

- **13 个内容完全一致**（远端因 Windows checkout 是 CRLF、本仓库是 LF，仅换行符不同）；
- **9 个不同，差异全部来自本次合并的改动**（路径常量、`ae_paper` 注册、编码修复、文档改写）——
  即"远端 == 合并前的本地 `safenetwork` 快照"，可安全覆盖；
- **2 个在合并前就已分叉**（两边内容不同，属那位开发者自己的本地改动）：

| 文件 | 线上 `main` | `safenetwork` 快照（本分支采纳） | 影响 |
|---|---|---|---|
| `agentenvs/detectors/autoencoder_detector.py` | `self.embed_dim = 0` | `int(embed_dim) if embed_dim is not None else 0` | 线上写法忽略传入的 `embed_dim`，与 `INTERFACE.md` §2.4「环境把统一维度应用到检测器上」冲突；统一 `embed_dim=4` 时 AE 的 `signal_dim` 会掉到 3，热替换报 `ValueError` |
| `algorithms/maddpg.py` | 无调试输出 | 每 100 次更新打印 `[DEBUG] q_det …` | 纯调试输出，无功能影响 |

**为了让分叉版本不会被覆盖后找不回来**，线上那 2 个文件已**逐字节**备份到
`legacy/diverged-from-remote/`（含完整差异与 3 种恢复方式，其中一种是"不碰 git 直接拷文件"，
见该目录 README）。

**结论：推新分支不会丢任何远端独有改动**（`main` 保持不动，且分叉版本已另有备份）。
分叉的 2 个文件按"快照为准"处理，但**需要作者确认**——若线上版本为准，
在新分支上补一个 commit 改回去即可，不要直接改 `main`。

## 6. 本次**没有**做的事（避免误解）

- ❌ 没有重训 MADDPG 策略（Kalman / AE / null 的策略对比仍缺，见 `PROTOCOL.md` §6 第 3 条）
- ❌ 没有去掉 ICPS 融合的 oracle 泄漏（`run_fusion.py` 仍用干净真值作参考状态）
- ❌ 没有裁决"带符号 embedding vs `normalize_obs` 裁剪"（`PROTOCOL.md` §6 第 1 条，需组内决定）
- ❌ 没有跑 ICPS 全链路（本机缺 `pandapower`/`pandera`/`geojson`）
- ❌ 没有补 Kalman / AE / ICPS / MADDPG 的报告（只产出了检测器对比表）
- ❌ 没有加 LICENSE（许可证由组内决定，未擅自指定）
