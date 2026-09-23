# -*- coding: utf-8 -*-
r"""冒烟测试：合并后仓库最容易坏的三件事。

跑法（在仓库根，不需要先 pip install）：
    $env:PYTHONPATH = "$PWD\src"
    python -m pytest tests -q
    # 或直接：python tests/test_smoke.py
"""
from __future__ import annotations

import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(REPO_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(REPO_ROOT, "src"))


# --------------------------------------------------------------------- 1. 契约
def test_registry_has_all_mechanisms():
    """四种机制 + null 都必须在注册表里，且 signal_dim 一致（换检测器不动环境）。"""
    from agentenvs.detectors import list_detectors, make_detector

    keys = list_detectors()
    for k in ("if", "kalman", "autoencoder", "ae_paper", "null"):
        assert k in keys, f"{k} 未注册：{keys}"

    dims = {}
    for k in ("if", "kalman", "autoencoder", "ae_paper"):
        det = make_detector(k, embed_dim=4)
        assert det.signal_dim == 7, f"{k} signal_dim={det.signal_dim} != 7"
        dims[k] = det.signal_dim
    assert len(set(dims.values())) == 1, "四种机制 signal_dim 不一致，无法做「只换检测器」对比"


def test_contract_copy_matches_real_base():
    """ae_repro/_contract.py 与真契约必须代码等价（去掉 docstring 后 AST 一致）。"""
    import agentenvs.detectors.base as real
    from ae_repro import _contract as local
    from ae_repro.verify_contract import _strip_docstrings

    a = _strip_docstrings(open(real.__file__, encoding="utf-8").read())
    b = _strip_docstrings(open(local.__file__, encoding="utf-8").read())
    assert a == b, "副本已与真契约漂移，请重新同步 src/ae_repro/_contract.py"


def test_detector_end_to_end_on_synthetic():
    """把 ae_paper 接进环境跑一步：观测维度/信号维度/动作都正常。"""
    import numpy as np
    from agentenvs import make_env
    from agentenvs.data_loader import make_synthetic
    from agentenvs.detectors import make_detector

    ds = make_synthetic(n_normal=600, n_per_class=80, feat_dim=20, seed=0)
    # 用小轮数构造，避免测试变成一次完整训练
    det = make_detector("ae_paper", embed_dim=4, epochs=15, patience=5)
    env = make_env(ds, detector=det, detector_embed_dim=4,
                   num_agents=3, max_steps=5)
    obs = env.reset()
    assert obs.shape == (3, env.obs_dim)

    out = env.detector.score_batch(ds.X_test[:16])
    assert out.score.shape == (16,)
    assert out.embedding.shape == (16, 4)
    assert np.isfinite(out.score).all()


# --------------------------------------------------------------------- 2. 数据协议
BUNDLE = os.path.join(REPO_ROOT, "data", "cicids_sample.npz")


@pytest.mark.skipif(not os.path.exists(BUNDLE),
                    reason="规范数据包不存在，先跑 python -m ae_repro.data_prep")
def test_protocol_v1_no_self_evaluation_leakage():
    """协议 v1 的核心保证：评估集与检测器拟合集的逐位重复率 ≤1%。"""
    import numpy as np

    d = np.load(BUNDLE, allow_pickle=True)
    Xn, Xt = d["X_normal_train"], d["X_test"]
    d.close()

    def h(X):
        return {r.tobytes() for r in np.ascontiguousarray(X.astype(np.float32))}

    hn = h(Xn)
    dup = sum(1 for r in np.ascontiguousarray(Xt.astype(np.float32)) if r.tobytes() in hn)
    rate = dup / len(Xt)
    assert rate <= 0.01, f"自评污染 {dup}/{len(Xt)} = {rate:.2%} 超过协议上限 1%"


@pytest.mark.skipif(not os.path.exists(BUNDLE),
                    reason="规范数据包不存在，先跑 python -m ae_repro.data_prep")
def test_protocol_v1_shapes():
    import numpy as np

    d = np.load(BUNDLE, allow_pickle=True)
    assert d["X_normal_train"].shape == (60000, 78)
    assert d["X_test"].shape[1] == 78
    assert len(d["class_names"]) == 6
    assert list(d["class_names"])[0] == "BENIGN"
    d.close()


# --------------------------------------------------------------------- 3. 脚本可导入
@pytest.mark.parametrize("name", ["eval_all_detectors", "eval_if_only", "prepare_data",
                                  "verify_pipeline", "train_cicids"])
def test_scripts_are_syntactically_importable(name):
    """scripts/*.py 的引导块 + 路径常量必须能过编译（防止改路径时打错）。"""
    import ast

    p = os.path.join(REPO_ROOT, "scripts", f"{name}.py")
    assert os.path.exists(p), f"缺少脚本 {p}"
    ast.parse(open(p, encoding="utf-8").read())


if __name__ == "__main__":
    raise SystemExit(pytest.main([os.path.abspath(__file__), "-q"]))
