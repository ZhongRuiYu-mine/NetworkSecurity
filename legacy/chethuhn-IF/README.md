# `legacy/chethuhn-IF/` —— 历史独立脚本（不纳入主路径）

## 这里面是什么

`IF.py`（232 行）是合并前 `chethuhn` 目录里的 Isolation Forest 实验脚本，也是
`docs/reports/Isolation Forest.docx` 提到的那份"IF 交付物"的**来源之一**。

## 为什么放在 legacy 而不是主路径

它和平台里纳入对比的那份 IF 是**两套不同的实验**：

| | `legacy/chethuhn-IF/IF.py` | `src/agentenvs/detectors/isolation_forest_detector.py` |
|---|---|---|
| 定位 | 单文件独立脚本（跑完即弃） | 平台 `DetectorBase` 契约实现 |
| 数据 | Monday 训练 → **Tuesday** 测试（单天） | 协议 v1 数据包（跨天分层 + 排除 Monday） |
| 预处理 | `StandardScaler` | Min-Max（仅良性统计量，论文 Eq.1） |
| 特征 | 78 维，含 Timestamp 剔除 | 78 维，与 `taxonomy`/`INTERFACE.md` 对齐 |
| 阈值 | 先验异常率分位 + Best-F1 扫描（**用测试标签**） | 主口径良性分位数 / 辅口径 F1 扫描 |
| `embed_dim` | 无此概念 | 4（与三种机制统一） |
| 并行 | `n_jobs=-1`（受限沙箱里会 `PermissionError`） | `n_jobs=1` |
| 产物 | 无（连 `if_analysis.png` 都未生成） | `results/if_only/`、对比表 |

两套数字**不可互相引用**，也不可与 `results/detector_comparison.md` 混引。

## 它还有什么价值

1. 里面的 **KS 分布漂移检验**（Monday 良性 vs Tuesday 良性，逐特征）是有用的诊断，
   协议 v1 的数据包里没有这个功能 —— 想复用就把它挪进 `scripts/` 并接到规范数据包上。
2. 它是"IF 用 `-score_samples` 作异常分"这条实现约定的原始出处
   （平台检测器的 docstring 就写了"对齐 chethuhn/.../IF.py 里的做法"）。

## 原始 CSV 在哪

`data/raw/cicids2017/`（junction，指向原 `chethuhn/network-intrusion-dataset/versions/1/`）。
本脚本用相对路径读当前目录的 CSV，直接跑需要 `cd data/raw/cicids2017` 后执行，
输出会写在那个（只读用途的）目录里 —— **不建议**；要复现请改用主路径的
`scripts/eval_all_detectors.py --detectors if`。
