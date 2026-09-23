# -*- coding: utf-8 -*-
"""
verify_contract.py —— 平台检测器契约自检（不需要平台仓库即可跑）。

检查 INTERFACE.md 第 8 章「适配性检查清单」的每一条：
  1. 继承 DetectorBase，fit/score_batch 都有，fit 返回 self
  2. 只用良性流量 fit
  3. score 方向：越高越异常（用合成数据验证 AUC > 0.95）
  4. score/flag/trust 形状 (n,)，embedding (n,k)，全部 finite
  5. flag == (score >= threshold)，threshold 是属性
  6. trust ∈ [0,1]，且语义 = 对当前 flag 的置信度
  7. embed_dim 可配置（0/4/8 三种都试）
  8. reset() 可调用
  9. save()/load() 往返一致（score 完全一致）
 10. signal_dim == 3 + embed_dim，as_signal 形状正确
 11. 单条 / 批量 / 1 维输入 / 鸭子类型 detect() 旧接口都通
 12. 维度不匹配时给出明确报错

用法： python -m ae_repro.verify_contract
"""

from __future__ import annotations

import os
from typing import List

import numpy as np

from .ae_detector import AePaperDetector

# ---------------------------------------------------------------------------
# 契约来源：合并仓库里优先用**真契约**（agentenvs.detectors.base），
# 只有脱离仓库单独运行时才退回自带的逐字副本 _contract.py。
# 两种来源都可用时做一次 AST 漂移检查，防止副本悄悄过期。
# ---------------------------------------------------------------------------
_CONTRACT_SOURCE = "ae_repro._contract"
try:  # pragma: no cover - 取决于运行位置
    from agentenvs.detectors import base as C  # type: ignore

    _CONTRACT_SOURCE = "agentenvs.detectors.base"
except Exception:  # noqa: BLE001
    from . import _contract as C  # type: ignore

PASS, FAIL = "[OK]", "[FAIL]"
_results: List[bool] = []


def _strip_docstrings(src: str) -> str:
    """去掉所有 docstring 后的 AST 转储 —— 用于判断两份契约是否「代码等价」。"""
    import ast

    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                body.pop(0)
                if not body:
                    body.append(ast.Pass())
    return ast.dump(tree, annotate_fields=False)


def check_contract_drift() -> None:
    """自检副本是否与真契约代码等价（只在能同时拿到两份时生效）。"""
    if _CONTRACT_SOURCE != "agentenvs.detectors.base":
        print("  [--] 未找到 agentenvs.detectors.base，跳过漂移检查（用的是自带副本）")
        return
    try:
        from . import _contract as local

        here = os.path.dirname(os.path.abspath(__file__))
        real = os.path.abspath(C.__file__)
        local_p = os.path.abspath(local.__file__)
        same = _strip_docstrings(open(real, encoding="utf-8").read()) == \
            _strip_docstrings(open(local_p, encoding="utf-8").read())
        check(same, f"_contract.py 与真契约代码等价（去掉 docstring 后 AST 一致）；"
                    f"真契约={os.path.relpath(real, here)}")
    except Exception as exc:  # noqa: BLE001
        print(f"  [--] 漂移检查失败：{exc!r}")


def check(cond: bool, msg: str) -> None:
    _results.append(bool(cond))
    print(f"  {PASS if cond else FAIL} {msg}")


def make_data(seed: int = 0, n_normal: int = 2500, n_attack: int = 600, d: int = 24):
    rng = np.random.default_rng(seed)
    Xn = np.abs(rng.normal(0.30, 0.05, size=(n_normal, d))).astype(np.float32)
    Xa = np.abs(rng.normal(0.30, 0.05, size=(n_attack, d))).astype(np.float32)
    Xa[:, :6] += 0.40                     # 攻击类在前 6 维上系统性偏移
    X = np.concatenate([Xn, Xa], axis=0)
    y = np.r_[np.zeros(len(Xn), int), np.ones(len(Xa), int)]
    return Xn, X, y


def main() -> int:
    print("=" * 78)
    print("AePaperDetector 契约自检（对照 INTERFACE.md 第 8 章清单）")
    print(f"契约来源：{_CONTRACT_SOURCE}")
    print("=" * 78)
    print("\n[0] 契约来源与副本漂移检查（合并仓库新增）")
    check_contract_drift()
    Xn, X, y = make_data()
    d = X.shape[1]

    print("\n[1] 继承与必须实现的方法")
    det = AePaperDetector(epochs=40, embed_dim=4, verbose=False)
    check(isinstance(det, C.DetectorBase), "isinstance(det, DetectorBase)")
    check(callable(getattr(det, "fit", None)), "有 fit()")
    check(callable(getattr(det, "score_batch", None)), "有 score_batch()")
    ret = det.fit(Xn, X_test=X, y_test=y)
    check(ret is det, "fit() 返回 self（平台链式调用要求）")

    print("\n[2] 只用良性流量 fit（换进攻击样本会改变基线）")
    det_clean = AePaperDetector(epochs=40, embed_dim=4, seed=42).fit(Xn.copy())
    det_dirty = AePaperDetector(epochs=40, embed_dim=4, seed=42).fit(X.copy())
    diff = abs(float(det_clean.threshold) - float(det_dirty.threshold))
    check(diff > 0, f"喂入攻击样本确实会改变阈值（Δthreshold={diff:.4f}）→ 现实中只喂良性")

    print("\n[3-6] 输出对象与语义")
    out = det.score_batch(X)
    check(out.score.shape == (len(X),), f"score 形状 {out.score.shape} == (n,)")
    check(out.flag.shape == (len(X),), f"flag 形状 {out.flag.shape} == (n,)")
    check(out.trust.shape == (len(X),), f"trust 形状 {out.trust.shape} == (n,)")
    check(out.embedding is not None and out.embedding.shape == (len(X), 4),
          f"embedding 形状 {None if out.embedding is None else out.embedding.shape} == (n, 4)")
    check(bool(np.isfinite(out.score).all()), "score 全 finite")
    check(bool(np.isfinite(out.trust).all()), "trust 全 finite")
    check(bool(np.isfinite(out.embedding).all()), "embedding 全 finite")
    check(float(out.trust.min()) >= 0.0 and float(out.trust.max()) <= 1.0,
          f"trust ∈ [{out.trust.min():.4f}, {out.trust.max():.4f}] ⊆ [0,1]")
    check(float(out.flag.min()) >= 0.0 and float(out.flag.max()) <= 1.0, "flag ∈ {0,1}")
    thr = float(det.threshold)
    check(bool(np.array_equal(out.flag, (out.score >= thr).astype(np.float32))),
          "flag == (score >= self.threshold)")
    from sklearn.metrics import roc_auc_score

    auc = float(roc_auc_score(y, out.score))
    check(auc > 0.95, f"score 方向是「越高越异常」（合成数据 AUC={auc:.4f} > 0.95）")
    hi = out.score >= thr
    conf_hi = out.trust[hi].mean() if hi.any() else float("nan")
    conf_lo = out.trust[~hi].mean() if (~hi).any() else float("nan")
    check(conf_hi > 0.5 and conf_lo > 0.5,
          f"trust 语义：告警时均值={conf_hi:.3f}>0.5 且不告警时均值={conf_lo:.3f}>0.5"
          f"（两边都表示「对当前 flag 的确信」）")

    print("\n[7] embed_dim 可配置")
    for k in (0, 4, 8):
        dk = AePaperDetector(epochs=25, embed_dim=k).fit(Xn)
        ok = dk.signal_dim == 3 + k
        o = dk.score_batch(X[:50])
        ok = ok and (o.embedding is None if k == 0 else o.embedding.shape == (50, k))
        check(ok, f"embed_dim={k} -> signal_dim={dk.signal_dim}, "
                  f"embedding={None if o.embedding is None else o.embedding.shape}")

    print("\n[8] reset()")
    try:
        det.reset()
        check(True, "reset() 可调用（无状态检测器为 no-op）")
    except Exception as exc:  # noqa: BLE001
        check(False, f"reset() 抛异常：{exc}")

    print("\n[9] save()/load() 往返")
    tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_tmp_contract")
    os.makedirs(tmp, exist_ok=True)
    p = os.path.join(tmp, "det")
    det.save(p)
    det2 = AePaperDetector(embed_dim=det.embed_dim).load(p)
    o1, o2 = det.score_batch(X), det2.score_batch(X)
    check(np.allclose(o1.score, o2.score, atol=1e-6),
          f"往返后 score 一致（max|Δ|={float(np.abs(o1.score - o2.score).max()):.2e}）")
    check(abs(float(det2.threshold) - thr) < 1e-9,
          f"往返后 threshold 一致（{det2.threshold}）")
    check(det2.embed_dim == det.embed_dim, f"往返后 embed_dim 一致（{det2.embed_dim}）")
    check(np.allclose(o1.trust, o2.trust, atol=1e-6), "往返后 trust 一致")
    check(os.path.exists(p + ".npz") and os.path.exists(p + ".json"),
          "落盘产物：<path>.npz（权重+统计量）与 <path>.json（配置+元信息）")

    print("\n[10-11] 信号向量 / 各种输入形态 / 旧接口")
    sig = out.as_signal(det.embed_dim)
    check(sig.shape == (len(X), 3 + det.embed_dim),
          f"as_signal({det.embed_dim}) -> {sig.shape} == (n, 3+embed_dim)")
    check(np.allclose(sig[:, C.SIGNAL_SCORE], out.score)
          and np.allclose(sig[:, C.SIGNAL_FLAG], out.flag)
          and np.allclose(sig[:, C.SIGNAL_TRUST], out.trust),
          "信号布局 [score, flag, trust, embedding...] 正确")
    one = det.score(X[0])
    check(one.score.shape == (1,), f"单条（1 维输入）可用，返回 {one.score.shape}")
    small = det.score_batch(X[:7])
    check(small.score.shape == (7,), "小批量（7 条）可用")
    big = det.score_batch(X[:1024])
    check(big.score.shape == (1024,), "大批量（1024 条）可用")
    legacy = det.detect(X[0])
    check(set(legacy) == {"anomaly_score", "anomaly_flag", "trust"},
          f"detect() 旧接口字段正确：{sorted(legacy)}")
    check(C.coerce_detector(det) is det, "coerce_detector 放行（DetectorBase 实例）")

    print("\n[12] 维度不匹配要有明确报错")
    try:
        det.score_batch(np.zeros((5, d + 3), dtype=np.float32))
        check(False, "维度不匹配竟然没报错")
    except ValueError as exc:
        check("特征维度不匹配" in str(exc), f"维度不匹配抛 ValueError 且信息明确：{exc}")

    print("\n" + "=" * 78)
    n_ok, n = sum(_results), len(_results)
    print(f"契约自检：{n_ok}/{n} 通过" + ("  ✅ 全部通过" if n_ok == n else "  ❌ 有失败项"))
    print("=" * 78)
    return 0 if n_ok == n else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
