# artifacts/ —— 模型产物（不入 git，可由脚本重建）

| 文件 | 来源 | 重建方式 |
|---|---|---|
| `dpnet_cicids.pt` | ICPS 网络层 DPNet（CIC-IDS2017，77 维 parquet 训练） | `python scripts/train_cicids.py`（需 pyarrow + GPU，约数十万行） |
| `dpnet_cicids.metrics.json` | DPNet 测试集指标 | 同上（落盘在训练脚本第 9 节） |

## 已知问题（合并时修掉，但未重训验证）

合并前的 `train_cicids.py` 落盘时**没有**写 `scaler_mu` / `scaler_sigma`，
而 `src/icps_detection/fusion.py` 的 `NetworkProbe` 需要这两个键 ——
也就是说**重跑训练会覆盖掉一个 `run_fusion.py` 读不了的 checkpoint**。

合并时已在训练脚本里补上这两个键（外加 `test_acc` / `config` / 指标 JSON），
但**没有重训验证**（需要 231 万行数据 + GPU）。见 `docs/DECISIONS.md` §4 第 2 条。

另外 `run_fusion.py` 的物理层仍有 oracle 泄漏（用干净真值作参考状态），
修它属于算法改动，不在本次合并范围内（`PROTOCOL.md` §6 第 4 条）。
