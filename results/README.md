# 结果目录说明

```
results/
├── detector_comparison.md      ★ 主表（协议 v1 主口径 + 乐观上界 + 逐类检出率）
├── detector_comparison.json    上表的机器可读原始数字（含全部口径与配置）
├── reports/                    各方向报告（IF docx、ae_repro 报告）
├── <detector>_only/            单检测器评估（ROC 图、分数分布、落盘检测器）
├── <detector>/                 MADDPG 策略训练产物（best.pt / history.json / eval_*.json）
└── _smoke/                     自检与冒烟产物（不入 git）
```

## 怎么生成

```powershell
$env:PYTHONPATH = "$PWD\src"
python scripts/eval_all_detectors.py                      # 主表
python scripts/eval_all_detectors.py --smoke              # 冒烟版（小样本）
python scripts/eval_if_only.py if                         # 单机制评估
python scripts/train.py --data bundle --detector if --episodes 200 --out results/if
```

## 引用数字前请先读 `PROTOCOL.md`

- **主口径**（良性分位数 α=0.05，`fit()` 不传 `y_test`）→ 报告的主数字，可外推；
- **辅口径**（F1 扫描，用评估集标签选阈值）→ 只能作为乐观上界，必须标注；
- 两套口径的数字**不得**放进同一列比较。

引用 precision 时务必同时给 `P@1%`（`results/detector_comparison.md` 表 1 已含）：
本评估集是分层抽样的，攻击占比远高于线上基率。

## 尚未生成的关键产物

| 产物 | 状态 |
|---|---|
| `results/kalman/`、`results/autoencoder/`、`results/null/` | ❌ 未跑（只有 IF 有策略权重），见 `PROTOCOL.md` §6 第 3 条 |
| `results/<detector>/` 下的端到端对比分析 | ❌ 待补 |
