# 结果目录说明

```
results/
├── detector_comparison.md      ★ 机制层主表（协议 v1 主口径 + 乐观上界 + 逐类检出率）
├── detector_comparison.json    上表的机器可读原始数字（含全部口径与配置）
├── method_analysis.md          ★★ 报告用章节：成功率/整体表现/逐攻击反应/横向对比/成败归因/场景选型
├── policy_sweep.md             逐攻击反应 + 失败归因 + MADDPG 产物对照（method_analysis 的数据源）
├── policy_sweep.json           上表的机器可读原始数字（含混淆矩阵、逐智能体分解）
├── reports/                    各方向报告（IF docx、ae_repro 报告）
├── <detector>_only/            单检测器评估（ROC 图、分数分布、落盘检测器）
├── <detector>/                 MADDPG 策略训练产物（best.pt / history.json / eval_*.json）
├── _legacy_platform/           合并前旧口径产物（逐字节拷贝，仅供对照，见下）
└── _smoke/                     自检与冒烟产物（不入 git）
```

## 怎么生成

```powershell
$env:PYTHONPATH = "$PWD\src"
python scripts/eval_all_detectors.py                      # 机制层主表
python scripts/eval_all_detectors.py --smoke              # 冒烟版（小样本）
python scripts/eval_policy_sweep.py                       # 逐攻击反应 + 失败归因
python scripts/eval_policy_sweep.py --smoke               # 冒烟版（自动写到 results/_smoke/）
python scripts/dump_policy_sweep.py                       # 把 policy_sweep.json 导成纯文本便于核对
python scripts/eval_if_only.py if                         # 单机制评估
python scripts/train.py --data bundle --detector if --episodes 200 --out results/if
```

> `--smoke` 模式的产物会自动落到 `results/_smoke/`，**不会覆盖正式产物**
> （早期版本会覆盖，3000 行的数字被当成正式结果引用过一次）。

## `_legacy_platform/` 是什么

从合并前的 `safenetwork/.../outputs/` **逐字节**拷来的 MADDPG 策略产物：

| 文件 | 内容 |
|---|---|
| `if/eval_if.json` | IF 策略**最后一轮**（ep200）评估 |
| `if/eval_best.json` | IF 策略按 **macro-F1** 选出的最优轮（实际是 **ep10**） |
| `if/history.json` | 200 episode 训练曲线（含每 10 轮一次的 `eval`） |
| `if_only/`、`kalman_only/`、`autoencoder_only/` | 旧口径的单检测器评估 |

⚠️ 这些数字用的是**旧口径**（29000 行评估集，含 14.02% 自评行 + F1 扫描阈值），
**只能和旧 docx 表 5.1 比，不能和 `detector_comparison.md` / `policy_sweep.md` 比**。

⚠️ `eval_*.json` **不记录 episode 号**，只能靠 `history.json` 里的 `eval` 字段反推。
这是产物本身的一个缺陷，已在 `method_analysis.md` §5.9 说明。

## 引用数字前请先读 `PROTOCOL.md`

- **主口径**（良性分位数 α=0.05，`fit()` 不传 `y_test`）→ 报告的主数字，可外推；
- **辅口径**（F1 扫描，用评估集标签选阈值）→ 只能作为乐观上界，必须标注；
- 两套口径的数字**不得**放进同一列比较。

引用 precision 时务必同时给 `P@1%`（`results/detector_comparison.md` 表 1 已含）：
本评估集是分层抽样的，攻击占比远高于线上基率。

`results/method_analysis.md` §5.1.2 给出了本仓库**此前不存在的「成功率」定义**
（机制层 = 该类被告警的比例；端到端 = 告警且判对，仅作下界）。
引用「成功率」时请一并引用该定义，否则数字不可比。

## 尚未生成的关键产物

| 产物 | 状态 |
|---|---|
| `results/kalman/`、`results/autoencoder/`、`results/null/` | ❌ 未跑（只有 IF 有策略权重），见 `PROTOCOL.md` §6 第 3 条 |
| `results/<detector>/` 下的端到端对比分析 | ❌ 待补 |
| 逐智能体的**响应动作分布**（监控/限流/阻断三档） | ❌ 从未评估过，见 `method_analysis.md` §5.9 |
