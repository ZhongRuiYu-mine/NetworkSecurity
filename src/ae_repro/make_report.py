# -*- coding: utf-8 -*-
"""
make_report.py —— 把 `experiments.py` 产出的 repro_report.json 变成可读的 Markdown。

用法
----
    python -m ae_repro.make_report --report results/repro_report.json --out results/REPORT.md
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Sequence


def _md_table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    out = ["| " + " | ".join(str(h) for h in header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        out.append("| " + " | ".join("" if c is None else str(c) for c in r) + " |")
    return "\n".join(out)


def _f(x: Any, nd: int = 4) -> str:
    if x is None:
        return "-"
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def build_markdown(report: Dict[str, Any]) -> str:
    env = report.get("environment", {})
    proto = report.get("protocol", {})
    head = report.get("headline", {})
    results: List[Dict[str, Any]] = report.get("results", [])
    main = next((r for r in results if r["name"] == "ae_paper_main"), results[0] if results else None)

    L: List[str] = []
    L.append(f"# {report.get('title', '复现报告')}")
    L.append("")
    L.append(f"生成时间：{report.get('generated_at', '-')}")
    L.append("")
    L.append("## 0. 一句话结论")
    L.append("")
    _lat = (main or {}).get("latency") or {}
    L.append(
        f"在 chethuhn CICIDS2017 上按平台协议复现 AUTO.pdf 第 1 阶段自编码器检测器："
        f"**ROC-AUC {_f(head.get('roc_auc'))} / PR-AUC {_f(head.get('pr_auc'))} / "
        f"P {_f(head.get('precision'))} / R {_f(head.get('recall'))} / "
        f"F1 {_f(head.get('f1'))} / FPR {_f(head.get('false_positive_rate'))}**"
        f"、基率校正后 **FP‰ {_f(head.get('fp_per_1000_flows'), 1)} / "
        f"P@1% {_f(head.get('precision_at_1pct'))} / "
        f"P@0.1% {_f(head.get('precision_at_0.1pct'))}**"
        f"（{head.get('main_config', '')}，阈值口径 {head.get('threshold_rule', '')}"
        f" = {_f(head.get('threshold'))}）。模型只有 **{head.get('n_params', '-')} 个参数**，"
        f"训练 {_f(head.get('train_seconds'), 1)} 秒，推理单条 "
        f"{_f(_lat.get('ms_per_sample'), 4)} ms。"
    )
    L.append("")
    L.append("> 与论文 Table 1 的 AE 行（0.949/0.82/0.79/0.80）**不可直接比数字**："
             "数据集、特征空间、阈值口径、评估协议四项全不同，详见第 5 节。")
    L.append("")

    L.append("## 1. 复现环境与协议")
    L.append("")
    L.append(_md_table(["项", "值"], [
        ["Python", env.get("python", "-")],
        ["torch", env.get("torch", "-")],
        ["CUDA", env.get("cuda", "-")],
        ["numpy", env.get("numpy", "-")],
        ["平台", env.get("platform", "-")],
        ["数据包", env.get("data_path", "-")],
        ["特征维数", env.get("n_features", "-")],
        ["类别空间", ", ".join(env.get("class_names", []) or [])],
        ["检测器良性训练集", env.get("n_normal_train", "-")],
        ["测试集", env.get("n_test", "-")],
        ["测试集按天", env.get("test_by_day", "-")],
        ["测试集已排除检测器训练数据", env.get("eval_exclude_normal_source", "-")],
    ]))
    L.append("")
    L.append(_md_table(["协议项", "取值"], [
        ["训练集", proto.get("train", "-")],
        ["测试集", proto.get("test", "-")],
        ["检测器训练约束", proto.get("detector_training_rule", "-")],
    ]))
    L.append("")
    L.append(f"> ⚠ {proto.get('note', '')}")
    L.append("")

    # ---------------- 新旧评估口径对照 ----------------
    pc = report.get("protocol_comparison")
    if pc:
        L.append("## 1.1 新旧评估口径对照（同一模型、同一阈值口径）")
        L.append("")
        L.append("修复前：`file_specs()` 的 Monday 项与检测器的 `normal_source()` 抽到**同一批行**"
                 "（同名文件 + 同 nrows + 同 seed），而切分是随机的，于是这批行有 1/5 进了测试集"
                 "——测试集里约 1/3 的样本是检测器**逐位见过**的训练数据，评估变成自评。"
                 "修复方式：切分时把 Monday 行排除在测试集之外"
                 "（`CicidsConfig.eval_exclude_normal_source=True`）。")
        L.append("")
        rows = []
        for tag, key in [("旧口径（自评）", "old_protocol"), ("新口径（已修复）", "new_protocol")]:
            v = pc.get(key) or {}
            m = v.get("metrics") or {}
            rows.append([
                tag, v.get("n_test", "-"),
                _f(m.get("roc_auc")), _f(m.get("pr_auc")), _f(m.get("precision")),
                _f(m.get("recall")), _f(m.get("f1")), _f(m.get("false_positive_rate")),
                _f(m.get("fp_per_1000_flows"), 1), _f(m.get("precision_at_1pct")),
                _f(m.get("precision_at_0.1pct")),
            ])
        L.append(_md_table(["口径", "测试集", "ROC-AUC", "PR-AUC", "P", "R", "F1", "FPR",
                            "FP‰", "P@1%", "P@0.1%"], rows))
        L.append("")
        ov_old = (pc.get("old_protocol") or {}).get("test_rows_identical_to_detector_training")
        ov_new = (pc.get("new_protocol") or {}).get("test_rows_identical_to_detector_training")
        if ov_old is not None:
            L.append(f"- 旧口径测试集里与检测器训练集**逐位完全相同**的样本：**{ov_old:.2%}**"
                     + (f"；新口径 {ov_new:.4%}" if ov_new is not None else ""))
            if ov_new is not None and ov_new > 0:
                L.append(f"- 新口径残留的 {ov_new:.2%} 是 CICIDS 自身的**跨天重复流**"
                         f"（不同天的采集里出现相同特征值的流），不是协议泄漏；"
                         f"想彻底干净需要按行去重后再切分。")
        L.append(f"- {((pc.get('old_protocol') or {}).get('note') or '')}")
        L.append(f"- 旧口径数据来源：`{(pc.get('old_protocol') or {}).get('source')}`"
                 f"（生成于 {(pc.get('old_protocol') or {}).get('generated_at')}）")
        L.append("- 结论：自评会让**误报率被低估**、攻击占比被稀释、precision 被抬高；"
                 "论文里请引用**新口径**，旧口径只作为「为什么要排除」的证据保留。")
        L.append("")

    L.append("## 2. 结果总览（主口径：良性分位数阈值）")
    L.append("")
    rows = []
    for r in results:
        m = r["main_metrics"]
        rows.append([
            r["label"], _f(m.get("roc_auc")), _f(m.get("pr_auc")), _f(m["precision"]),
            _f(m["recall"]), _f(m["f1"]), _f(m["false_positive_rate"]),
            _f(m.get("fp_per_1000_flows"), 1), _f(m.get("precision_at_1pct")),
            r["fit"]["n_params"], r["fit"]["epochs_run"], _f(r["fit"]["seconds"], 1),
        ])
    L.append(_md_table(["实验", "ROC-AUC", "PR-AUC", "P", "R", "F1", "FPR", "FP‰", "P@1%",
                        "参数量", "实际训练轮数", "训练秒"], rows))
    L.append("")
    L.append("> **FP‰ 与 P@1% 是提案第 4 节要求的口径**：FPR = FP/(FP+TN) 只在良性样本上算，"
             "与攻击占比无关，所以 FP‰ 可以直接外推；而 precision 会被基率压垮"
             "（本表里的 P 是测试集内高攻击占比下的值），P@1% = ρ·TPR / (ρ·TPR + (1-ρ)·FPR)，"
             "ρ=1%，才是「能不能部署」的数字。")
    L.append("")

    L.append("## 3. 阈值口径对比（主配置）")
    L.append("")
    rows = []
    for k in ("percentile", "evt", "best_f1", "per_day_calib"):
        m = main["metrics"].get(k)
        if not m:
            continue
        rows.append([k, _f(m.get("threshold")), _f(m["precision"]), _f(m["recall"]),
                     _f(m["f1"]), _f(m["false_positive_rate"]),
                     _f(m.get("fp_per_1000_flows"), 1), _f(m.get("precision_at_1pct"))])
    L.append(_md_table(["口径", "阈值", "P", "R", "F1", "FPR", "FP‰", "P@1%"], rows))
    L.append("")
    evt = main["thresholds"]["evt"]
    L.append(f"- EVT/GPD 拟合：ξ={_f(evt['xi'], 3)}，β={_f(evt['beta'], 3)}，"
             f"u={_f(evt['u'], 3)}，超阈值样本 {evt['n_exceed']} 个 —— "
             f"**{'一致性检验未通过，已退回经验分位数' if evt.get('fallback') else '通过一致性检验'}**。")
    L.append(f"- `best_f1` 是测试集上扫出来的最优阈值，属于**乐观上界**，不能当作上线口径。")
    L.append("- `per_day_calib` 是漂移诊断：每天只用当天良性样本重标定阈值（不使用任何标签），"
             "用来区分「模型排序能力不行」和「全局阈值被跨天漂移带偏」。")
    pdc = main.get("per_day_calibration")
    if pdc:
        L.append("")
        rows = []
        for d, v in pdc["per_day"].items():
            rows.append([d, _f(v["threshold"]), f"{v['vs_global_threshold']:+.4f}",
                         _f(v["fpr_on_this_day_benign"]),
                         _f(v["detection_rate_on_this_day_attacks"])])
        L.append(_md_table(["天", "当天重标定阈值", "与全局阈值之差",
                            "当天良性误报率", "当天攻击检出率"], rows))
    L.append("")

    L.append("## 4. 主配置分解")
    L.append("")
    L.append("### 4.1 按攻击类别（主口径阈值）")
    L.append("")
    rows = []
    for cls, v in main["by_class"].items():
        if "detection_rate" in v:
            rows.append([cls, v["n"], _f(v["detection_rate"]), v["flagged"]])
        else:
            rows.append([f"{cls}（良性）", v["n"], f"误报率 {_f(v['false_positive_rate'])}", v["flagged"]])
    L.append(_md_table(["类别", "样本数", "检出率 / 误报率", "告警数"], rows))
    L.append("")
    L.append("### 4.2 按天的分布漂移诊断")
    L.append("")
    rows = []
    for d, v in main["by_day"].items():
        rows.append([d, v["n"], _f(v["attack_ratio"], 3),
                     _f(v["detection_rate_on_attacks"]) if v["detection_rate_on_attacks"] is not None else "-",
                     _f(v["false_positive_rate_on_benign"]) if v["false_positive_rate_on_benign"] is not None else "-",
                     _f(v.get("benign_score_median")), _f(v.get("attack_score_median"))])
    L.append(_md_table(["天", "测试样本", "攻击占比", "攻击检出率", "良性误报率",
                        "良性分数中位数", "攻击分数中位数"], rows))
    L.append("")
    _worst = None
    for _d, _v in (main.get("by_day") or {}).items():
        if _v.get("benign_score_mean") is None:
            continue
        if _worst is None or _v["benign_score_mean"] > _worst[1]["benign_score_mean"]:
            _worst = (_d, _v)
    if _worst is not None:
        L.append(f"> 分数分布用**中位数**而非均值：CICIDS 里个别良性流的 Flow Bytes/s、"
                 f"Packet Length Variance 之类会算出极端值，均值会被单条样本带跑"
                 f"（例如 {_worst[0]} 的良性分数均值 {_f(_worst[1].get('benign_score_mean'), 2)}、"
                 f"中位数 {_f(_worst[1].get('benign_score_median'))}）。"
                 f"这也是 `trust` 必须用**稳健尺度**标定的原因。")
    else:
        L.append("> 分数分布用**中位数**而非均值：CICIDS 里个别良性流会算出极端值，"
                 "均值会被单条样本带跑。")
    L.append("")
    L.append("### 4.3 逐特征重构误差归因 Top15（论文第 7 步的可解释性）")
    L.append("")
    L.append(_md_table(["排名", "特征", "平均平方重构误差"],
                       [[i + 1, a["feature"], f"{a['mean_sq_residual']:.6g}"]
                        for i, a in enumerate(main["attribution_top15"])]))
    L.append("")
    emb = main.get("embedding", {})
    L.append(f"**embedding 维度**：{emb.get('dim')}，取的特征索引 {emb.get('feature_idx')} "
             f"＝ {emb.get('feature_names')}（带符号标准化残差，供 critic 判方向）")
    L.append("")

    L.append("## 5. 与论文的差异（写论文必须写清）")
    L.append("")
    for i, d in enumerate(report.get("deviation_from_paper", []), 1):
        L.append(f"{i}. {d}")
    L.append("")
    L.append("### 论文参考指标")
    L.append("")
    for name, ref in (report.get("paper_reference") or {}).items():
        L.append(f"**{name}**")
        L.append("")
        for k, v in ref.items():
            L.append(f"- `{k}`: {v}")
        L.append("")

    L.append("## 6. 消融结论")
    L.append("")
    main_f1 = main["main_metrics"]["f1"]
    main_auc = main["main_metrics"].get("roc_auc")
    main_pr = main["main_metrics"].get("pr_auc")
    base = next((r for r in results if r["name"] == "ablation_paper_literal"), None)
    if base:
        b = base["main_metrics"]
        L.append(f"- **主配置 vs 论文字面实现（本次运行实测）**："
                 f"ROC-AUC {_f(main_auc)} vs **{_f(b.get('roc_auc'))}**，"
                 f"PR-AUC {_f(main_pr)} vs **{_f(b.get('pr_auc'))}**，"
                 f"F1 {_f(main_f1)} vs {_f(b['f1'])}。"
                 f"**论文字面实现（原始 MSE 打分）的排序能力更强**（AUC 高 "
                 f"{b.get('roc_auc') - (main_auc or 0):+.4f}，PR-AUC 高 "
                 f"{b.get('pr_auc') - (main_pr or 0):+.4f}）。"
                 f"它的代价有二：① 逐特征残差没有可比量纲，论文第 7 步的归因和 "
                 f"`embedding` 都会失真；② 训练预算敏感——"
                 f"`design_probe/probe1_transform_x_score.json` 里同一配置 "
                 f"76→181 轮时 PR-AUC 从 0.621 掉到 0.556（**注意这三个 probe 文件是"
                 f"修复前旧口径的设计阶段记录、生成脚本未随包提供，只表趋势不可引用**，"
                 f"见 `design_probe/PROVENANCE.md`），而逐特征标准化残差在 "
                 f"15~250 轮之间几乎不动。所以这是「排序能力」与「可解释性 + 训练预算鲁棒性」"
                 f"的取舍；**只比 AUC/PR-AUC 的话应该用字面实现**，要让 critic 拿到"
                 f"有方向的残差则用主配置。")
    for r in results:
        if r["name"] == "ae_paper_main":
            continue
        delta = r["main_metrics"]["f1"] - main_f1
        dauc = (r["main_metrics"].get("roc_auc") or 0) - (main_auc or 0)
        L.append(f"- {r['label']}：F1 {_f(r['main_metrics']['f1'])}（{delta:+.4f}），"
                 f"ROC-AUC {_f(r['main_metrics'].get('roc_auc'))}（{dauc:+.4f}）—— {r.get('note', '')}")
    L.append("")
    L.append("### 结论")
    L.append("")
    _drops = [r for r in results if r["name"] in ("ablation_zscore_input", "ablation_log1p_input")]
    if _drops and main_auc is not None:
        _dmin = min(float(r["main_metrics"].get("roc_auc") or 0.0) for r in _drops)
        L.append(f"1. **输入不要再做一次标准化**：平台已按论文 Eq.1 做过 min–max，"
                 f"核心内部再加 z-score / log1p+z-score 会让 ROC-AUC 掉最多 "
                 f"{_f(main_auc - _dmin)}（{_f(main_auc)} → {_f(_dmin)}），"
                 f"攻击流量的幅度信息被压掉了。")
    else:
        L.append("1. **输入不要再做一次标准化**：平台已按论文 Eq.1 做过 min–max，"
                 "核心内部再加 z-score / log1p+z-score 会让排序能力明显下降，"
                 "攻击流量的幅度信息被压掉了。")
    L.append("2. **打分口径是「排序能力 ↔ 可解释性/鲁棒性」的取舍**：原始 MSE 的 "
             "AUC/PR-AUC 更高（见上），但它的逐特征残差没有可比量纲（归因与 embedding 失真），"
             "且对训练轮数敏感；逐特征标准化残差在 15~250 轮之间几乎不漂移，"
             "并且是逐特征归因的前提。**不要只看 F1 就下结论**：F1 受阈值口径影响更大。")
    L.append("3. **对称/非对称解码器差别不大**，按论文原话用对称解码器完全可行；"
             "具体数字见上表（本文默认非对称）。")
    L.append("4. **真正的瓶颈是阈值，不是模型**：")
    L.append(f"   - `best_f1`（乐观上界）F1 {_f(main['metrics']['best_f1']['f1'])}，"
             f"`evt` F1 {_f(main['metrics']['evt']['f1'])}，"
             f"`percentile` F1 {_f(main['metrics']['percentile']['f1'])}；")
    _pdc = main.get("per_day_calibration") or {}
    _per = _pdc.get("per_day") or {}
    if _per:
        _ths = [float(v["threshold"]) for v in _per.values()]
        _spread = (max(_ths) / min(_ths)) if min(_ths) > 0 else float("nan")
        L.append(f"   - 跨天漂移的直接证据是**阈值本身差多少**：当天重标定阈值 "
                 f"{_f(min(_ths))}~{_f(max(_ths))}（相差 **{_f(_spread, 1)} 倍**）。"
                 f"注意「当天误报率≈5%」是构造出来的（阈值就是当天良性分数的 95% 分位），"
                 f"它**不构成**「阈值可校准」的证据；真正的结论是「一天标定、多天使用」"
                 f"会让某些天的实际误报率远超设计值（见上表）。")
    L.append("5. **弱项是特定攻击族**：PortScan / WebAttack 检出率≈0，"
             "因为这些流在特征空间里离 Monday 良性太近；DDoS/DoS 表现良好。"
             "注意 `Other` 这一类**实测全部是周二 FTP/SSH-Patator 暴力破解**"
             "（Bot 样本数 < min_attack_rows 被并入 Other，Infiltration 没进测试集），"
             "所以「Other 检出率≈0」的真实含义是**暴力破解 100% 漏检**。"
             "这一点建议在论文里如实写，并作为「AE 需要与 IF/Kalman 互补」的论据。")
    L.append("")

    L.append("## 7. 组员接入方法")
    L.append("")
    L.append("```powershell")
    L.append("# 1) 拷贝两个文件到仓库")
    L.append("copy ae_repro\\ae_core.py     <repo>\\agentenvs\\detectors\\ae_core.py")
    L.append("copy ae_repro\\ae_detector.py <repo>\\agentenvs\\detectors\\ae_paper_detector.py")
    L.append("# 2) 在 <repo>\\agentenvs\\detectors\\__init__.py 里加一行：")
    L.append("#    from .ae_paper_detector import AePaperDetector")
    L.append("# 3) 直接跑（注册名 ae_paper / ae_cicids）")
    L.append("& $PY train.py --data bundle --detector ae_paper --embed-dim 4 --out outputs/ae_paper")
    L.append("```")
    L.append("")
    L.append("```python")
    L.append("# 或者不改仓库，直接传对象")
    L.append("from ae_repro.ae_detector import AePaperDetector")
    L.append("det = AePaperDetector(embed_dim=4).load('ae_repro/artifacts/ae_paper')")
    L.append("env = make_env(ds, detector=det, detector_embed_dim=4)")
    L.append("```")
    L.append("")

    return "\n".join(L)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="repro_report.json -> REPORT.md")
    ap.add_argument("--report", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    with open(args.report, "r", encoding="utf-8") as fh:
        report = json.load(fh)
    md = build_markdown(report)
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.report)), "REPORT.md")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(md)
    print(f"[out] Markdown 报告已写入 {out}（{len(md)} 字符）")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
