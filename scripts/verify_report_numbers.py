# -*- coding: utf-8 -*-
"""核对 method_analysis.md 里引用的关键数字与 JSON 源是否一致。

存在的理由：这份报告里的数字全是手抄进 Markdown 的，抄错一位就会
以讹传讹。本脚本把「报告里写死的值」和「JSON 里的真值」逐条比对，
不一致就报错。改了报告或重跑了产物都应该重跑本脚本。

用法：
    python scripts/verify_report_numbers.py
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def _load(rel: str):
    with open(os.path.join(ROOT, rel), "r", encoding="utf-8") as fh:
        return json.load(fh)


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    dc = _load(os.path.join("results", "detector_comparison.json"))
    ps = _load(os.path.join("results", "policy_sweep.json"))
    ae = _load(os.path.join("src", "ae_repro", "results", "repro_report.json"))

    dc_by = {r["name"]: r for r in dc["results"]}
    ps_by = {r["name"]: r for r in ps["results"]}
    fails: list[str] = []
    checks = 0

    def eq(label: str, got, want, tol: float = 5e-5) -> None:
        nonlocal checks
        checks += 1
        try:
            ok = abs(float(got) - float(want)) <= tol
        except (TypeError, ValueError):
            ok = got == want
        if not ok:
            fails.append(f"{label}: JSON={got!r} 报告写的={want!r}")

    # ---------------- 表 5-1 主口径 ----------------
    for name, f1, auc, r, fpr, p1, thr in [
        ("if", 0.4560, 0.7327, 0.3445, 0.0579, 0.0567, 0.53326),
        ("kalman_frozen", 0.5797, 0.7841, 0.5205, 0.0958, 0.0521, 2.33629),
        ("autoencoder", 0.6412, 0.8021, 0.5800, 0.0797, 0.0684, 0.00051),
        ("ae_paper", 0.6521, 0.8474, 0.5689, 0.0612, 0.0858, 2.42414),
    ]:
        m = dc_by[name]["main"]
        eq(f"表5-1 {name}.f1", m["f1"], f1)
        eq(f"表5-1 {name}.roc_auc", m["roc_auc"], auc)
        eq(f"表5-1 {name}.recall", m["recall"], r)
        eq(f"表5-1 {name}.fpr", m["false_positive_rate"], fpr)
        eq(f"表5-1 {name}.p@1%", m["precision_at_1pct"], p1)
        eq(f"表5-1 {name}.threshold", m["threshold"], thr, tol=1e-4)

    # ---------------- 表 5-2 F1 扫描 ----------------
    for name, f1, fpr, thr in [
        ("if", 0.6077, 0.1003, 0.48548),
        ("kalman_frozen", 0.6652, 0.2057, 0.72309),
        ("kalman_decay", 0.6495, 0.2133, 0.68176),
        ("autoencoder", 0.6564, 0.1028, 0.00032),
        ("ae_paper", 0.6775, 0.0355, 4.21303),
    ]:
        o = dc_by[name]["optimistic"]
        eq(f"表5-2 {name}.f1", o["f1"], f1)
        eq(f"表5-2 {name}.fpr", o["false_positive_rate"], fpr)
        eq(f"表5-2 {name}.threshold", o["threshold"], thr, tol=1e-4)

    # F1 涨幅
    for name, gain in [("if", 0.1517), ("kalman_frozen", 0.0855),
                       ("kalman_decay", 0.0731), ("autoencoder", 0.0152),
                       ("ae_paper", 0.0254)]:
        g = dc_by[name]["optimistic"]["f1"] - dc_by[name]["main"]["f1"]
        eq(f"表5-2 {name}.f1涨幅", g, gain, tol=1e-4)

    # ---------------- 逐攻击类别检出率（表 5-3 / 5-4）----------------
    want_det = {
        "if": dict(DDoS=0.1861, PortScan=0.0000, DoS=0.6009, WebAttack=0.0101, Other=0.0000),
        "kalman_frozen": dict(DDoS=0.6746, PortScan=0.0067, DoS=0.5732, WebAttack=0.0000, Other=0.0000),
        "autoencoder": dict(DDoS=0.6685, PortScan=0.1036, DoS=0.6845, WebAttack=0.0101, Other=0.0000),
        "ae_paper": dict(DDoS=0.6677, PortScan=0.0054, DoS=0.6859, WebAttack=0.0000, Other=0.0000),
    }
    for name, d in want_det.items():
        pc = ps_by[name]["policies"]["detector"]["per_class"]
        for cls, v in d.items():
            eq(f"表5-3 {name}.{cls}.alert_rate", pc[cls]["alert_rate"], v)

    # 逐类分数中位数 / 阈值（表 5-3 第三行）
    want_ratio = {
        "if": dict(DDoS=0.93, PortScan=0.66, DoS=1.14, WebAttack=0.68, Other=0.69),
        "kalman_frozen": dict(DDoS=1.50, PortScan=0.007, DoS=1.23, WebAttack=0.097, Other=0.149),
        "autoencoder": dict(DDoS=5.50, PortScan=0.15, DoS=87.5, WebAttack=0.076, Other=0.094),
        "ae_paper": dict(DDoS=4.71, PortScan=0.22, DoS=4.65, WebAttack=0.27, Other=0.17),
    }
    for name, d in want_ratio.items():
        r = ps_by[name]
        by = r["agent0_characterization"]["by_true_class"]
        thr = r["threshold"]
        for cls, v in d.items():
            eq(f"表5-3 {name}.{cls}.中位数/阈值",
               by[cls]["score_median"] / thr, v, tol=0.006)

    # ---------------- 表 5-6 归因分解 ----------------
    want_decomp = {
        "if": (1760, 463, 4230),
        "kalman_frozen": (1679, 1680, 3094),
        "autoencoder": (2005, 1738, 2710),
        "ae_paper": (2009, 1662, 2782),
    }
    for name, (s, ce, df) in want_decomp.items():
        dcmp = ps_by[name]["policies"]["detector"]["decomposition"]
        eq(f"表5-6 {name}.成功", dcmp["success_alerted_and_correct"], s)
        eq(f"表5-6 {name}.告警但判错", dcmp["class_error_alerted"], ce)
        eq(f"表5-6 {name}.双重失败", dcmp["double_failure"], df)
        eq(f"表5-6 {name}.双重失败占比",
           dcmp["double_failure"] / dcmp["n_attack"], 
           {"if": 0.656, "kalman_frozen": 0.479,
            "autoencoder": 0.420, "ae_paper": 0.431}[name], tol=6e-4)

    # 攻击总数
    for name in want_decomp:
        eq(f"攻击总数 {name}", ps_by[name]["policies"]["detector"]["decomposition"]["n_attack"], 6453)

    # ---------------- 全局成功率（判定规则）----------------
    want_glob = {
        "if": dict(detector=0.2727, specialist=0.0716),
        "kalman_frozen": dict(detector=0.2602, specialist=0.2596),
        "autoencoder": dict(detector=0.3107, specialist=0.2572),
        "ae_paper": dict(detector=0.3113, specialist=0.2569),
    }
    for name, d in want_glob.items():
        pol = ps_by[name]["policies"]
        eq(f"§5.7 {name}.detector成功率", pol["detector"]["success_rate_on_attacks"], d["detector"])
        eq(f"§5.7 {name}.specialist成功率",
           pol["specialist"]["vote"]["success_rate_on_attacks"], d["specialist"])

    # ---------------- §5.7 恒判 BENIGN 下界 ----------------
    for name in want_glob:
        eq(f"§5.7 {name}.benign_only", 
           ps_by[name]["policies"]["benign_only"]["success_rate_on_attacks"], 0.0)

    # ---------------- ae_repro（EVT / 阈值 / 逐天）----------------
    th = ae["results"][0]["thresholds"]
    eq("§5.5 EVT阈值", th["evt"]["value"], 4.9539, tol=1e-4)
    eq("§5.5 EVT xi", th["evt"]["xi"], -0.250, tol=1e-3)
    eq("§5.5 EVT beta", th["evt"]["beta"], 5.514, tol=1e-3)
    eq("§5.5 percentile阈值", th["percentile"]["value"], 2.3092, tol=1e-4)
    eq("§5.5 best_f1阈值", th["best_f1"]["value"], 3.6312, tol=1e-4)
    eq("§5.5 headline.roc_auc", ae["headline"]["roc_auc"], 0.8451)
    eq("§5.5 headline.f1", ae["headline"]["f1"], 0.6500)

    # 阈值口径对比三行（metrics 按口径分键）
    modes = ae["results"][0]["metrics"]
    for label, f1, fpr, p1 in [("percentile", 0.6500, 0.0632, 0.0834),
                               ("evt", 0.6698, 0.0319, 0.1482),
                               ("best_f1", 0.6721, 0.0404, 0.1238)]:
        eq(f"§5.5 {label}.f1", modes[label]["f1"], f1)
        eq(f"§5.5 {label}.fpr", modes[label]["false_positive_rate"], fpr)
        eq(f"§5.5 {label}.p@1%", modes[label]["precision_at_1pct"], p1)

    # 逐天（by_day 是 dict：天 -> 指标）
    for name, det, fpr in [("Friday", 0.5138, 0.0745), ("Thursday", 0.0000, 0.0404),
                           ("Tuesday", 0.0000, 0.0529), ("Wednesday", 0.6859, 0.1070)]:
        o = ae["results"][0]["by_day"][name]
        eq(f"§5.5 {name}.攻击检出率", o["detection_rate_on_attacks"], det)
        eq(f"§5.5 {name}.良性误报率", o["false_positive_rate_on_benign"], fpr)

    # 按天重标定阈值（跨天差 3.1 倍）
    per_day = ae["results"][0]["per_day_calibration"]["per_day"]
    thr_min = min(v["threshold"] for v in per_day.values())
    thr_max = max(v["threshold"] for v in per_day.values())
    eq("§5.5 当天阈值下界", thr_min, 2.1139, tol=1e-4)
    eq("§5.5 当天阈值上界", thr_max, 6.4728, tol=1e-4)
    eq("§5.5 跨天阈值倍数", thr_max / thr_min, 3.1, tol=0.05)

    # 逐特征归因 Top3
    attr = ae["results"][0]["attribution_top15"]
    for i, (fname, val) in enumerate([("Init_Win_bytes_forward", 0.04632),
                                      ("URG Flag Count", 0.04408),
                                      ("Destination Port", 0.03707)]):
        eq(f"§5.5 归因Top{i+1} 特征名", attr[i]["feature"], fname)
        eq(f"§5.5 归因Top{i+1} 数值", attr[i]["mean_sq_residual"], val, tol=1e-5)

    # 消融：论文字面实现 AUC 更高 / log1p 掉点最多
    abl = {r["name"]: r["metrics"]["percentile"] for r in ae["results"]}
    eq("§5.5 字面实现 AUC", abl["ablation_paper_literal"]["roc_auc"], 0.8618)
    eq("§5.5 log1p AUC", abl["ablation_log1p_input"]["roc_auc"], 0.6461)
    eq("§5.5 去BN AUC", abl["ablation_no_bn"]["roc_auc"], 0.7968)
    eq("§5.5 字面实现 PR-AUC", abl["ablation_paper_literal"]["pr_auc"], 0.7495)
    eq("§5.5 主配置 PR-AUC", abl["ae_paper_main"]["pr_auc"], 0.6617)
    eq("§5.5 参数量", ae["headline"]["n_params"], 6274)

    # ---------------- MADDPG 旧产物 ----------------
    base = os.path.join("results", "_legacy_platform", "if")
    with open(os.path.join(ROOT, base, "eval_if.json"), "r", encoding="utf-8") as fh:
        ei = json.load(fh)
    eq("§5.9 eval_if.投票准确率", ei["vote"]["accuracy"], 0.0754)
    eq("§5.9 eval_if.macro_f1", ei["vote"]["macro_f1"], 0.0009)
    eq("§5.9 eval_if.kappa", ei["vote"]["kappa"], -0.0219)
    eq("§5.9 eval_if.检测率", ei["vote"]["detection_rate"], 0.8320)
    eq("§5.9 eval_if.误报率", ei["vote"]["false_alarm_rate"], 0.9044)
    with open(os.path.join(ROOT, base, "eval_best.json"), "r", encoding="utf-8") as fh:
        eb = json.load(fh)
    eq("§5.9 eval_best.投票准确率", eb["vote"]["accuracy"], 0.6450)
    eq("§5.9 eval_best.macro_f1", eb["vote"]["macro_f1"], 0.0090)
    with open(os.path.join(ROOT, base, "history.json"), "r", encoding="utf-8") as fh:
        hist = json.load(fh)
    eq("§5.9 history.轮数", len(hist), 200)
    eq("§5.9 history.ep1 reward", hist[0]["mean_reward"], -3.92, tol=1e-2)
    eq("§5.9 history.ep200 reward", hist[-1]["mean_reward"], -10.66, tol=1e-2)
    eq("§5.9 history.actor_loss 终值", hist[-1]["actor_loss"], -1.154e6, tol=1e3)
    evals = [e for e in hist if e.get("eval")]
    eq("§5.9 history 评估点数", len(evals), 20)
    neg = [e for e in evals if e["eval"]["vote_kappa"] < 0]
    eq("§5.9 history Kappa 全为负", len(neg), len(evals))
    eq("§5.9 ep10 准确率", evals[0]["eval"]["vote_accuracy"], 0.6450)

    # ---------------- 自评污染 ----------------
    eq("§5.1 自评污染率", dc["leakage"]["duplicate_rate"], 0.00704, tol=1e-5)

    print(f"检查 {checks} 项，失败 {len(fails)} 项")
    for f in fails:
        print("  [MISMATCH] " + f)
    if fails:
        return 1
    print("[OK] method_analysis.md 引用的数字与 JSON 源全部一致")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
