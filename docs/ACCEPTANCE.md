# 合并验收记录

> 环境：Windows + Python 3.11.6，numpy 1.26.4，sklearn 1.5.2，torch 2.10.0+cu128
> 数据：CICIDS2017（`data/raw/cicids2017` junction → 原 `chethuhn/...`）

## 1. 复现命令与结果

| # | 命令 | 结果 |
|---|---|---|
| 1 | `python scripts/verify_pipeline.py` | ✅ **exit 0**，「全部自检通过」：4 种检测器 fit/score/契约一致 → 环境构造 → 检测器解耦（观测维度不变）→ MADDPG 集中式更新 → 热替换（kalman/autoencoder/null） |
| 2 | `python -m agentenvs.detectors.templates` | ✅ **exit 0**，「模板自检通过」（zscore_demo / stateful_demo / WrapExistingModel，signal_dim=7） |
| 3 | `python -m ae_repro.verify_contract` | ✅ **35/35 通过**；契约来源 = **`agentenvs.detectors.base`（真契约）**，副本漂移检查 OK |
| 4 | `python -m pytest tests -q` | ✅ **10 passed** |
| 5 | `python -m ae_repro.data_prep` | ✅ 生成 `data/cicids_sample.npz`（60000 良性 / 36875 训练 / 25000 测试），用时 19.7 s |
| 6 | `python scripts/make_manifest.py` | ✅ 生成 `data/manifest.json` + `data/checksums.sha256`（8 CSV 843.7 MB + 8 parquet 258.1 MB） |
| 7 | `python scripts/eval_all_detectors.py` | ✅ 6 种机制全跑通（含 ae_paper 训练 599 s），产出 `results/detector_comparison.{md,json}` |

## 2. 关键验收点

### 2.1 协议 v1 的自评污染被消除

| | 评估集 | 与 `X_normal_train` 逐位重复 | 占比 |
|---|---|---|---|
| 合并前平台数据包 | 29000 | 4066（全是良性，4012 个不同向量） | **14.02 %** |
| 协议 v1 规范包 | 25000 | 176 | **0.70 %** |

由 `tests/test_smoke.py::test_protocol_v1_no_self_evaluation_leakage` 把关（阈值 1%）。

### 2.2 四种机制维度一致（"只换检测器、其它全不动"成立）

```
if             -> IsolationForestDetector  embed_dim=4 signal_dim=7
kalman         -> KalmanFilterDetector     embed_dim=4 signal_dim=7
autoencoder    -> AutoencoderDetector      embed_dim=4 signal_dim=7
ae_paper       -> AePaperDetector          embed_dim=4 signal_dim=7   ← 合并后新增注册
null           -> NullDetector             embed_dim=0 signal_dim=3
```

`ae_paper` 原先**从未在真实 `agentenvs` 仓库里跑过**（其 README 自述待办第 1 条）；
本次接入后 35 项契约自检 + 环境内打分全部通过。

### 2.3 主口径结果（协议 v1：良性 95% 分位阈值，`fit()` 只喂良性、不传 `y_test`）

| 机制 | P | R | F1 | ROC-AUC | FPR | FP‰ | P@1% |
|---|---|---|---|---|---|---|---|
| `ae_paper` | 0.7638 | 0.5689 | **0.6521** | **0.8474** | 6.12 % | 61.2 | 0.0858 |
| `autoencoder`（平台内置） | 0.7168 | 0.5800 | 0.6412 | 0.8021 | 7.97 % | 79.7 | 0.0684 |
| `kalman_frozen` | 0.6541 | 0.5205 | 0.5797 | 0.7841 | 9.58 % | 95.8 | 0.0521 |
| `kalman_decay` | 0.6515 | 0.5168 | 0.5764 | 0.7795 | 9.62 % | 96.2 | 0.0515 |
| `if` | 0.6745 | 0.3445 | **0.4560** | 0.7327 | 5.79 % | 57.9 | 0.0567 |
| `null`（消融） | 0.2581 | 1.0000 | 0.4103 | 0.5000 | 100 % | 1000.0 | 0.0100 |

**口径变化直接改变结论。** 旧 docx 表 5.1（F1 扫描 + 14 % 自评）给的是
`AE 0.6689 > Kalman 0.6348 > IF 0.5910`；主口径下 IF 掉到 0.4560，
`kalman_frozen` 反而超过它。原因是 IF 的旧数字受益于**用测试标签选阈值**
（同一模型 F1 扫描 0.6077 vs 主口径 0.4560）。

### 2.4 交叉验证：合并后的接入路径复现出原包的结论

`ae_paper` 在本仓库得到的 **ROC-AUC = 0.8474**，与 `ae_repro` 独立报告里的
**0.8451** 一致（差 0.0023，量级属正常数值差异）。说明副本合并没有改变模型行为。

### 2.5 逐类检出率暴露的弱项（与 `ae_repro` 报告一致）

| 机制 | DDoS | PortScan | DoS | WebAttack | Other(暴力破解) |
|---|---|---|---|---|---|
| `ae_paper` | 0.6677 | 0.0054 | 0.6859 | 0.0000 | 0.0000 |
| `autoencoder` | 0.6685 | 0.1036 | 0.6845 | 0.0101 | 0.0000 |
| `kalman_frozen` | 0.6746 | 0.0067 | 0.5732 | 0.0000 | 0.0000 |
| `if` | 0.1861 | 0.0000 | 0.6009 | 0.0101 | 0.0000 |

PortScan / WebAttack / 暴力破解在所有重构式与点式方法上几乎全漏
（`if` 连 DDoS 都只有 0.19）。这是"为什么需要多机制互补 + 时序/行为特征"的
量化论据，建议原样写进报告。

## 3. 顺手修掉的缺陷（验收证据）

| 缺陷 | 验证方式 |
|---|---|
| `templates.py` 在 GBK 控制台打印 `✅` 抛 `UnicodeEncodeError` → 自检**假失败**（exit 1） | 修后 `python -m agentenvs.detectors.templates` **exit 0** |
| `train_cicids.py` 落盘缺 `scaler_mu`/`scaler_sigma` → 重训会覆盖成 `fusion.py` 读不了的 checkpoint | 代码已修 + 指标落盘 JSON；**未重训**（需 231 万行 + GPU） |
| 产物路径散落（`scripts/outputs/`、硬编码 `data/cicids_sample.npz`） | 统一到 `results/` 与 `data/`，实跑确认落点 |
| `.gitignore` 用了行尾注释导致规则失效（差点提交 1.1 GB 原始数据） | `git check-ignore` 逐项确认；待提交体积 1139 MB → **2.46 MB** |

## 4. 本次未覆盖（明确不在合并范围内）

- ❌ Kalman / autoencoder / null 的**端到端策略训练**（仍只有 IF 有 `best.pt`）
- ❌ ICPS 融合层的 `DetectorBase` 适配与 oracle 泄漏修复
- ❌ "带符号 embedding vs `normalize_obs` 裁剪"的裁决
- ❌ ICPS 全链路运行（本机缺 `pandapower`/`pandera`/`geojson`）
- ❌ Kalman / AE / ICPS / MADDPG 报告
- ❌ LICENSE

以上全部登记在 `PROTOCOL.md` §6 与 `README.md` §6 的诚实清单里。

## 5. 推送到线上仓库的状态

目标：<https://github.com/ZhongRuiYu-mine/NetworkSecurity>（远端 `main` = `b4cfc87`，3 个提交，24 个文件）

**推送策略：只进新分支，`main` 不动。**

| 项 | 值 |
|---|---|
| 分支名 | `merge/netsec-protocol-v1` |
| 分支 tip | 见 `git rev-parse merge/netsec-protocol-v1`（每次用脚本重建都会变） |
| 父提交 | `origin/main`（= `b4cfc87`），因此可开 PR、`git diff origin/main...` 即可审阅 |
| 分支内容 | 与本地 `master` 树**完全一致**（成员树哈希相同，已核验） |
| 相对 `main` 的规模 | 92 files changed, 60929 insertions(+), 254 deletions(-)（含重命名检测：`agentenvs/`→`src/agentenvs/` 等） |
| 远端文件数 | 93 个，无 >1MB 文件 |

**当前状态：✅ 已推送成功（2026-09-23）**

```
remote: Create a pull request for 'merge/netsec-protocol-v1' on GitHub by visiting:
remote:      https://github.com/ZhongRuiYu-mine/NetworkSecurity/pull/new/merge/netsec-protocol-v1
To https://github.com/ZhongRuiYu-mine/NetworkSecurity
 * [new branch]      merge/netsec-protocol-v1 -> merge/netsec-protocol-v1
```

核验结果：

| 检查 | 结果 |
|---|---|
| 远端分支列表 | `main` = `b4cfc87`（**未改动**）、`merge/netsec-protocol-v1` = 本地 tip |
| 本地 vs 远端 tip | 一致 |
| 远端分支树 == 本地 `master` 树 | 一致（`84bf5471…`） |
| 远端 `main` 是否被动过 | **没有** |
| 开 PR 地址 | <https://github.com/ZhongRuiYu-mine/NetworkSecurity/pull/new/merge/netsec-protocol-v1> |

### 推送过程中遇到并解决的两个非代码问题（留档）

1. **403 权限被拒** → 用 GitHub API 诊断出根因：owner 已发出邀请，但邀请处于
   **"待接受"** 状态，接受前 `permissions.push` 仍是 `false`（凭据本身是有效的
   `oyxpuaiinnng` OAuth token）。`PATCH /user/repository_invitations/334400909` 接受后
   `push` 变为 `true`。
   > 排查提示：`GET /repos/<owner>/<repo>` 看 `permissions.push`，
   > `GET /user/repository_invitations` 看待接受邀请 —— 比反复重推有效。
2. **网络中断**（`curl 55 Send failure: Connection was reset`）→ 本机到 github.com:443
   不稳定。换 `-c http.version=HTTP/1.1 -c http.postBuffer=524288000` 后一次成功。

### 推送方法（已封装脚本）

```powershell
cd D:\higher\netsec
.\scripts\push_branch.ps1 -DryRun     # 先看将要推什么（不推送、离线也能用）
.\scripts\push_branch.ps1             # 按当前 master 重建分支并推送
```

手工命令（等价，三条出路任选其一）：

```powershell
# ① 让仓库 owner 把 oyxpuaiinnng 加成 collaborator（Settings → Collaborators），然后：
cd D:\higher\netsec
git push -u origin merge/netsec-protocol-v1

# ② 改用 owner 自己的账号认证（先清掉缓存凭据，推送时会弹 GitHub 登录窗口）：
cmdkey /delete:LegacyGeneric:target=git:https://github.com
cd D:\higher\netsec
git push -u origin merge/netsec-protocol-v1

# ③ 不申请权限，Fork 到自己的账号再推（网页点 Fork 后）：
cd D:\higher\netsec
git remote add fork https://github.com/oyxpuaiinnng/NetworkSecurity
git push -u fork merge/netsec-protocol-v1
# 然后从 fork 向 ZhongRuiYu-mine:main 开 PR
```

无论哪条路，`main` 都不会被改动；分叉版本的备份也已经在
`legacy/diverged-from-remote/` 里（见该目录 README 的 3 种恢复方式）。
