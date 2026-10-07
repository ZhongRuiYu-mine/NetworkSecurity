# 各方向报告

| 文件 | 来源 | 状态 |
|---|---|---|
| `Isolation Forest.docx` | `safenetwork/Isolation Forest.docx`（原作者交付的 IF 英文报告） | ⚠️ **其表 5.1 的三方对比表已作废**，见下 |
| `ae_repro_REPORT.md` | `ae_repro/results/REPORT.md`（AE 复现报告） | ✅ 有效；其"下一阶段"清单与本仓库 `PROTOCOL.md` §6 对应 |
| `../results/method_analysis.md` / `.docx` | 由 `scripts/eval_policy_sweep.py` + `scripts/md_to_docx.py` 生成 | ✅ **本项目报告的第 5 章**（三方机制结果与分析） |

## 关于 `Isolation Forest.docx` 表 5.1（三方对比表）

该表是在**旧口径**下得到的，不能继续引用：

1. 评估集 29000 行中 **4066 行（14.02%）与检测器拟合集 `X_normal_train` 逐位重复**
   —— 检测器在自己见过的样本上被评估；
2. 三种机制都走 **F1 扫描**（把 `y_test` 传进 `fit()`，用测试标签选阈值），
   而 `ae_repro` 的报告走良性分位数 —— 两张表的口径不同，不能放在一起比较。

**替换物**：`results/detector_comparison.md`（由 `python scripts/eval_all_detectors.py`
按 `PROTOCOL.md` v1 生成）。它包含：

- 表 1：主口径（良性分位数 α=0.05）下 5 种机制 + null 消融的完整指标
  （P / R / F1 / ROC-AUC / PR-AUC / FPR / FP‰ / P@1% / P@0.1%）；
- 表 2：逐攻击类别检出率；
- 表 3：F1 扫描乐观上界（明确标注用了测试标签）；
- 表 4：配置与耗时（含 Kalman 的 `mode` —— 旧表缺这一项，导致结果不可复现）。

建议：把 docx 里的表 5.1 直接替换为 `results/detector_comparison.md` 的表 1 + 表 2，
并在正文里注明口径变化。docx 本体保留为历史记录。

**更完整的替换物**：`results/method_analysis.md`（+ `.docx`）。它不只给数字，还回答
「成功率怎么定义」「每一次攻击是什么反应」「失败是机制层还是判定层的锅」
「各机制适合什么场景」，是旧 docx 表 5.1 那三行数字**应该配套的整章正文**。

## 仍缺的报告

| 方向 | 缺失内容 |
|---|---|
| Kalman | 无独立报告（原 docx 的三方对比里只有一行数字）；机制层数字见 `results/method_analysis.md` §5.4 |
| Autoencoder（平台内置） | 同上，见 `results/method_analysis.md` §5.5（`ae_paper` 有独立报告） |
| ICPS（`network`） | **完全没有文档**：无 README、无报告、无指标落盘 |
| MADDPG 平台 | 无训练报告（只有 `history.json` 与 `eval_*.json`）；且已有 IF 策略**已崩塌**，见 `results/method_analysis.md` §5.9 |
| 策略层报告（三方机制各跑一遍 MADDPG） | ❌ 未跑，`PROTOCOL.md` §6 第 3 条 |
| 项目级 | 无总结报告 / 演示材料 |
