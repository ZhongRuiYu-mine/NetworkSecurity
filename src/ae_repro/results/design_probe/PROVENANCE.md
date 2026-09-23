# design_probe 数据来源说明（**引用前必读**）

这三个 JSON 是**超参选型阶段**的搜索记录，**不是**最终报告的指标，引用时请注意：

1. **口径是修复前的旧口径**：跑这些探针时测试集里仍含检测器的训练数据
   （Monday 行，占测试集 32.68%），所以这里的 AUC / PR-AUC / F1 与
   `results/REPORT.md` §2 的表**不可直接比**，只能看**趋势**（谁比谁好、往哪个方向调）。
2. **生成脚本未随本包提供**：探针是用当时的临时脚本跑的一次性搜索，不在 `ae_repro/`
   里，因此**不可复现**。要引用具体数字，请在当前协议下重跑。
3. 修复后在**新口径**下重做的对照见 `results/REPORT.md` §6（那里的数字才是可引用的）：
   例如"原始 MSE 排序更强"这个结论在新口径下依然成立
   （AUC 0.8618 vs 0.8451），且已在报告里如实写出。

| 文件 | 内容 |
|---|---|
| `probe1_transform_x_score.json` | 输入变换（none / zscore / log1p）× 打分口径（原始 MSE / 逐特征标准化）的 12 格交叉表 |
| `probe2_training_length.json` | 训练轮数（patience 15/40/80/200）、weight decay、dropout、lr 的单变量对照 |
| `probe3_final_combo.json` | 收敛前的 5 组组合方案（final_A..E），final_D 即最终主配置 |
