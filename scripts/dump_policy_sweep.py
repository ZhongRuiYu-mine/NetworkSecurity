# -*- coding: utf-8 -*-
"""把 policy_sweep.json 里报告要引用的数字一次性导成纯文本，避免反复手抄出错。

用法：
    python scripts/dump_policy_sweep.py            # 打到 stdout
    python scripts/dump_policy_sweep.py > x.txt
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    path = os.path.join(ROOT, "results", "policy_sweep.json")
    with open(path, "r", encoding="utf-8") as fh:
        d = json.load(fh)

    print("policies:", d["policies"])
    print("generated_at:", d.get("generated_at"))
    print()
    for r in d["results"]:
        if "policies" not in r:
            print("=== %s  (无策略数据：%s)" % (r["name"], r.get("reason")))
            continue
        env = r["env"]
        print("=== %s | fit=%.2fs | thr=%.5f (%s) | src=%s"
              % (r["name"], r["fit_seconds"], r["threshold"],
                 r["threshold_mode"], r["detector_source"].get("source")))
        print("    env: obs=%s act=%s agents=%s focus=%s"
              % (env.get("obs_dim"), env.get("action_dim"),
                 env.get("num_agents"), env.get("focus")))
        t = r["train_detector_operating_point"]
        print("    训练集工作点: P=%.4f R=%.4f F1=%.4f FPR=%.4f flag=%.4f"
              % (t["precision"], t["recall"], t["f1"],
                 t["false_positive_rate"], t["flag_rate"]))
        b = r["detector_binary"]
        print("    评估集二值  : n=%s P=%.4f R=%.4f F1=%.4f AUC=%.4f PR-AUC=%.4f"
              % (b.get("n"), b["precision"], b["recall"], b["f1"],
                 b["roc_auc"], b.get("pr_auc", float("nan"))))
        print("                  FPR=%.4f FPp=%.1f P@1%%=%.4f P@0.1%%=%.4f"
              % (b["false_positive_rate"], b["fp_per_1000_flows"],
                 b["precision_at_1pct"], b["precision_at_0.1pct"]))
        a = r.get("maddpg_policy_artifact") or {}
        print("    MADDPG 产物 : %s" % (a.get("protocol") if a.get("available")
                                        else a.get("reason")))

        # 逐类机制层反应
        p = r["policies"]["detector"]
        print("    -- detector 规则: 成功率=%.4f 告警时判对=%.4f 告警率=%.4f"
              % (p["success_rate_on_attacks"], p["accuracy_when_alerted"],
                 p["flag_rate"]))
        print("       decomp=%s" % p["decomposition"])
        ch = r["agent0_characterization"]["by_true_class"]
        for c, e in p["per_class"].items():
            if not e.get("n"):
                continue
            cc = ch.get(c, {})
            print("       %-10s n=%-6d alert=%.4f succ=%.4f med=%s p95=%s"
                  % (c, e["n"], e["alert_rate"], e["success_rate"],
                     cc.get("score_median"), cc.get("score_p95")))
        # specialist
        sp = r["policies"].get("specialist")
        if sp:
            focus = env.get("agent_focus") or []
            print("    -- specialist 逐智能体:")
            for ag in sp["per_agent"]:
                aid = ag.get("agent_id", 0)
                fc = ag.get("focus_class") or (focus[aid] if aid < len(focus) else "?")
                print("       %-18s focus=%-10s succ=%.4f acc_alerted=%.4f maj=%.3f"
                      % (ag["agent_name"], fc,
                         ag["success_rate_on_attacks"],
                         ag["accuracy_when_alerted"], ag["majority_share"]))
            v = sp["vote"]
            print("       %-18s %-10s succ=%.4f acc_alerted=%.4f maj=%.3f "
                  "macroF1=%.4f kappa=%.4f"
                  % ("投票", "", v["success_rate_on_attacks"],
                     v["accuracy_when_alerted"], v["majority_share"],
                     v["macro_f1"], v["kappa"]))
            print("       投票逐类: %s"
                  % {c: round(e["success_rate"], 4)
                     for c, e in v["per_class"].items() if e.get("n")})
        bo = r["policies"].get("benign_only")
        if bo:
            print("    -- benign_only: 成功率=%.4f macroF1=%.4f maj=%.3f"
                  % (bo["success_rate_on_attacks"], bo["macro_f1"],
                     bo["majority_share"]))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
