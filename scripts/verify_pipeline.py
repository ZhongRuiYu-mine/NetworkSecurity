"""
环境自检脚本 —— 跑通就说明 MADDPG 环境 + 可插拔检测接口都可用。

用法（工作目录 = 仓库根）：
    python scripts/verify_pipeline.py
    ... python scripts/verify_pipeline.py --data real        # 用真实 CICIDS2017 CSV（较慢）
    ... python scripts/verify_pipeline.py --detectors if,kalman,autoencoder,ae_paper,null

依次检查：
  1. 三种检测器是否都能 fit / 打分，输出契约是否一致（可插拔接口）
  2. 环境构造、观测维度、动作解码、奖励与指标
  3. 三种检测器分别接入环境后行为是否一致（说明解耦成功）
  4. MADDPG 能否选择动作、写入回放池、完成一次集中式更新
  5. 运行期热替换检测器（set_detector）
"""
from __future__ import annotations


# --- 合并仓库引导（scripts/_bootstrap） ---
import os as _os, sys as _sys
REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_os.path.join(REPO_ROOT, "src"), REPO_ROOT):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
del _os, _sys, _p
# --- 合并仓库引导结束 ---

import argparse
import sys
import time
import traceback
from typing import List

import numpy as np


def hr(title: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


def check_detectors(X_normal, X_test, y_test, det_names: List[str]):
    hr("1. 检测器插件自检（fit / score / 契约一致性）")
    from agentenvs.detectors import DetectorOutput, make_detector

    y_bin = (y_test != 0).astype(int)
    results = {}
    for name in det_names:
        t0 = time.time()
        det = make_detector(name)
        det.fit(X_normal, X_test=X_test, y_test=y_bin)
        out = det.score_batch(X_test[:512])
        assert isinstance(out, DetectorOutput), f"{name} 未返回 DetectorOutput"
        assert out.score.shape == (512,), f"{name} score 形状错误: {out.score.shape}"
        assert out.flag.shape == (512,)
        assert out.trust.shape == (512,)
        assert np.isfinite(out.score).all(), f"{name} score 含 NaN/inf"
        sig = out.as_signal(getattr(det, "embed_dim", 0))
        assert sig.shape == (512, det.signal_dim), f"{name} 信号维度不一致"

        # 用 AUC 粗看检测器本身有没有区分度
        auc = float("nan")
        try:
            from sklearn.metrics import roc_auc_score

            s = det.score_batch(X_test).score
            auc = roc_auc_score(y_bin, s)
        except Exception:
            pass
        thr = getattr(det, "threshold", None)
        thr_s = "None" if thr is None else f"{float(thr):.5f}"
        print(
            f"  [OK] {name:<14} signal_dim={det.signal_dim:<2} "
            f"embed_dim={getattr(det, 'embed_dim', 0):<2} "
            f"threshold={thr_s:<10} "
            f"AUC={auc:.4f}  用时 {time.time() - t0:.1f}s"
        )
        results[name] = dict(auc=auc, signal_dim=det.signal_dim)
    return results


def check_env(dataset, det_name: str, num_agents: int = 5, embed_dim: int = 4, max_steps: int = 40):
    """
    统一的 embed_dim 让三种检测器的信号维度完全一致
    （3 + embed_dim = 7），这样观测维度也一致 -> MADDPG 无需任何改动即可换检测器。
    不同检测器的原始 embedding 维度本来不同（IF=0 / Kalman=4 / AE=特征数），
    用 detector_embed_dim 统一是官方推荐的对比实验做法。
    """
    from agentenvs.detectors import make_detector
    from agentenvs import CyberDefenseEnv, EnvConfig

    det = make_detector(det_name, embed_dim=embed_dim)
    env = CyberDefenseEnv(
        X_data=dataset.X_train,
        y_labels=dataset.y_train,
        detector=det,
        spec=dataset.spec,
        config=EnvConfig(
            num_agents=num_agents, max_steps=max_steps, seed=0, detector_embed_dim=embed_dim
        ),
        X_normal_for_detector=dataset.X_normal_train,
        X_test=dataset.X_test,
        y_test=dataset.y_test,
    )
    return env


def check_env_behaviour(dataset, det_names: List[str], num_agents: int = 5):
    hr("2-3. 环境检查（各检测器接入后行为一致性）")
    from algorithms import MADDPG, MADDPGConfig

    baseline_obs_shape = None
    envs = {}
    for name in det_names:
        env = check_env(dataset, name, num_agents)
        print(env.describe())

        obs = env.reset()
        assert obs.shape == (num_agents, env.obs_dim), f"观测形状错误 {obs.shape}"
        assert np.isfinite(obs).all(), "观测含 NaN/inf"
        assert obs.min() >= 0.0 and obs.max() <= 1.0, "观测未落在 [0,1]"

        # 随机策略跑一个 episode
        rng = np.random.default_rng(0)
        ep_reward, steps = 0.0, 0
        for _ in range(5):
            actions = rng.uniform(-1, 1, size=(num_agents, env.action_dim)).astype(np.float32)
            cls = rng.integers(0, env.num_classes, size=num_agents)
            obs, rew, done, info = env.step_with_classes(actions, cls)
            assert rew.shape == (num_agents,), f"奖励形状错误 {rew.shape}"
            ep_reward += float(rew.sum())
            steps += 1
            if done:
                break
        m = env.episode_metrics()
        print(
            f"  随机策略 {steps} 步：累计奖励={ep_reward:.2f}  "
            f"平均准确率={m['mean_accuracy']:.3f}  "
            f"检测器 F1={m['detector']['f1']:.3f} "
            f"检测率={m['detector']['detection_rate']:.3f}"
        )

        # 动作解码检查
        a = np.linspace(-1, 1, 11, dtype=np.float32).reshape(-1, 1)
        _, resp = env.decode_actions(a)
        assert set(resp.tolist()) <= {0, 1, 2}, f"响应档位解码错误: {resp}"

        if baseline_obs_shape is None:
            baseline_obs_shape = (env.obs_dim, env.num_classes, env.action_dim)
        else:
            assert (env.obs_dim, env.num_classes, env.action_dim) == baseline_obs_shape, (
                f"{name} 的维度与首个检测器不一致："
                f"{(env.obs_dim, env.num_classes, env.action_dim)} vs {baseline_obs_shape}"
            )
        envs[name] = env

    print("\n  [OK] 所有检测器产生相同形状的观测/动作空间 -> 换检测器无需改 MADDPG 代码")
    return envs


def check_maddpg(env, episodes: int = 3):
    hr("4. MADDPG 自检（选择动作 / 回放池 / 集中式更新）")
    from algorithms import MADDPG, MADDPGConfig

    cfg = MADDPGConfig(
        num_agents=env.num_agents,
        obs_dim=env.obs_dim,
        num_classes=env.num_classes,
        action_dim=env.action_dim,
        buffer_size=5000,
        batch_size=32,
        warmup_steps=32,
        hidden_dims=(64, 64),
        critic_hidden_dims=(128, 128),
        noise_scale=0.2,
        seed=0,
    )
    agent = MADDPG(cfg)
    print(" ", agent.describe())

    t0 = time.time()
    for ep in range(episodes):
        obs = env.reset()
        agent.reset_noise()
        ep_ret = 0.0
        while True:
            actions, cls, raws = agent.select_actions_raw(obs)
            next_obs, rew, done, info = env.step_with_classes(actions, cls)
            agent.store(obs, raws, rew, next_obs, np.full(env.num_agents, float(done)), cls)
            if agent.ready():
                agent.update()
            ep_ret += float(rew.mean())
            obs = next_obs
            if done:
                break
        m = env.episode_metrics()
        print(
            f"  episode {ep + 1}: 平均奖励={ep_ret:.2f}  缓冲={len(agent.buffer)}  "
            f"更新次数={agent.total_updates}  准确率={m['mean_accuracy']:.3f}"
        )

    assert agent.total_updates > 0, "集中式更新没有发生"
    losses = agent.last_losses
    assert np.isfinite(losses["critic"]).all() and np.isfinite(losses["actor"]).all(), (
        f"损失出现 NaN：{losses}"
    )
    print(f"  [OK] critic_loss={np.round(losses['critic'], 4).tolist()}  "
          f"actor_loss={np.round(losses['actor'], 4).tolist()}  用时 {time.time() - t0:.1f}s")

    # 确定性评估 + 存档往返（写到仓库内的 results/_smoke，避免沙箱临时目录权限问题）
    obs = env.reset()
    det_actions, det_cls = agent.select_actions(obs, deterministic=True)
    assert det_actions.shape == (env.num_agents, env.action_dim)
    import os

    out_dir = os.path.join(REPO_ROOT, "results", "_smoke")
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, "maddpg_roundtrip.pt")
    agent.save(p)
    agent.load(p)
    print(f"  [OK] 确定性执行 / 存档往返 正常（{p}，可随时删除）")
    return agent


def check_hot_swap(dataset, det_names: List[str], num_agents: int = 3):
    hr("5. 运行期热替换检测器（A/B 对比实验的关键）")
    from agentenvs import CyberDefenseEnv, EnvConfig
    from agentenvs.detectors import make_detector

    # 统一 embed_dim（=4），保证三种检测器信号维度一致才能热替换
    env = CyberDefenseEnv(
        X_data=dataset.X_train,
        y_labels=dataset.y_train,
        detector=make_detector(det_names[0], embed_dim=4),
        spec=dataset.spec,
        config=EnvConfig(num_agents=num_agents, max_steps=10, seed=1, detector_embed_dim=4),
        X_normal_for_detector=dataset.X_normal_train,
        X_test=dataset.X_test,
        y_test=dataset.y_test,
    )
    env.reset()
    first = env.obs_dim
    for name in det_names[1:]:
        env.set_detector(make_detector(name, embed_dim=4))
        obs = env.reset()
        assert obs.shape[1] == first, f"热替换后观测维度变了：{obs.shape[1]} != {first}"
        print(f"  [OK] 热替换为 {name:<14} 观测维度仍为 {obs.shape[1]}，检测器={type(env.detector).__name__}")
    return env


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["synthetic", "real"], default="synthetic")
    ap.add_argument("--detectors", default="if,kalman,autoencoder,null")
    ap.add_argument("--num-agents", type=int, default=5)
    ap.add_argument("--rows-per-file", type=int, default=3000)
    ap.add_argument("--ae-epochs", type=int, default=8)
    args = ap.parse_args()

    det_names = [d.strip() for d in args.detectors.split(",") if d.strip()]

    hr(f"0. 加载数据（{args.data}）")
    from agentenvs import make_synthetic, build_dataset, DataConfig, FileSpec, default_file_specs
    from agentenvs.taxonomy import ClassSpec

    t0 = time.time()
    if args.data == "synthetic":
        ds = make_synthetic(n_normal=2000, n_per_class=400, feat_dim=26, num_attack_classes=5)
    else:
        files = [FileSpec(f.filename, min(f.nrows, args.rows_per_file), f.seed) for f in default_file_specs()]
        ds = build_dataset(DataConfig(files=files, class_preset="default", normal_ratio_in_train=0.35))
    print(f"  用时 {time.time() - t0:.1f}s")
    print(ds.summary())

    # Autoencoder 训练轮数可调，冒烟测试用小值
    import agentenvs.detectors.autoencoder_detector as ae_mod

    _orig_init = ae_mod.AutoencoderDetector.__init__

    def _patched(self, *a, **kw):
        kw.setdefault("epochs", args.ae_epochs)
        _orig_init(self, *a, **kw)

    ae_mod.AutoencoderDetector.__init__ = _patched

    check_detectors(ds.X_normal_train, ds.X_test, ds.y_test, det_names)
    envs = check_env_behaviour(ds, det_names, args.num_agents)
    check_maddpg(envs[det_names[0]], episodes=3)
    check_hot_swap(ds, det_names, num_agents=min(3, args.num_agents))

    hr("全部自检通过 ✅")
    print("  环境与可插拔检测接口均可用；接下来可以跑 train.py 训练 MADDPG。")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
