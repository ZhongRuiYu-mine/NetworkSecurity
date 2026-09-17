"""
MADDPG 训练 / 评估入口（CTDE：集中式训练，分散式执行）。

用法（工作目录 = D:\\NETWORK）：
    set PYTHONPATH=D:\\NETWORK
    set PY=D:\\PycharmProjects\\pythonProject\\.venv\\Scripts\\python.exe

  1) 快速跑通（合成数据 + IF 检测器，几十秒）
       %PY% train.py --data synthetic --detector if --episodes 30

  2) 真实 CICIDS2017（先用 build_dataset 落盘的数据包，避免每次解析 CSV）
       %PY% train.py --data bundle --bundle data/cicids_sample.npz --detector kalman --episodes 200

  3) 三种检测机制对比（每个机制一个 run，结果写到 outputs/<detector>/）
       %PY% train.py --data bundle --detector if          --episodes 200 --out outputs/if
       %PY% train.py --data bundle --detector kalman      --episodes 200 --out outputs/kalman
       %PY% train.py --data bundle --detector autoencoder --episodes 200 --out outputs/autoencoder

  4) 直接读原始 CSV 构建数据（慢，但不需要预先落盘）
       %PY% train.py --data real --rows-per-file 20000 --detector if --episodes 50

产物（--out 目录）：
    best.pt / last.pt        MADDPG 权重
    history.json             每轮训练指标（奖励、准确率、损失）
    eval_<detector>.json     测试集评估（各智能体 + 投票 + 检测器）
    curves.png               训练曲线 / 混淆矩阵（需要 matplotlib）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np

from agentenvs import (
    CyberDefenseEnv,
    DataConfig,
    EnvConfig,
    FileSpec,
    build_dataset,
    default_file_specs,
    load_bundle,
    make_synthetic,
    save_bundle,
)
from agentenvs.detectors import list_detectors, make_detector
from agentenvs.taxonomy import ClassSpec
from algorithms import MADDPG, MADDPGConfig
from utils.metrics import evaluate_agents


# --------------------------------------------------------------------------- 工具
def resolve_detector_kwargs(name: str, embed_dim: int) -> Dict[str, Any]:
    """给不同检测器设置统一信号维度（3 + embed_dim）与合理的默认超参。"""
    kwargs: Dict[str, Any] = {"embed_dim": embed_dim}
    if name in ("if", "isolation_forest"):
        kwargs.update(n_estimators=200, max_samples=256, random_state=42, n_jobs=1)
    elif name in ("kalman", "kf"):
        kwargs.update(mode="decay", window=200.0, track_rate=3e-3)
    elif name in ("autoencoder", "ae"):
        kwargs.update(hidden_dims=(64, 32), latent_dim=8, epochs=30, batch_size=256, lr=1e-3)
    return kwargs


def build_or_load_dataset(args) -> Any:
    if args.data == "synthetic":
        return make_synthetic(
            n_normal=args.synthetic_rows,
            n_per_class=max(args.synthetic_rows // 5, 100),
            feat_dim=26,
            num_attack_classes=max(args.num_agents, 3),
        )
    if args.data == "bundle":
        if not args.bundle or not os.path.exists(args.bundle):
            raise FileNotFoundError(
                f"找不到数据包 {args.bundle}。请先执行：python prepare_data.py --out {args.bundle}"
            )
        return load_bundle(args.bundle)
    # real：直接解析原始 CSV（较慢）
    files = [
        FileSpec(f.filename, min(f.nrows, args.rows_per_file), f.seed)
        for f in default_file_specs(nrows=args.rows_per_file)
    ]
    ds = build_dataset(
        DataConfig(files=files, class_preset=args.class_preset,
                   normal_ratio_in_train=args.normal_ratio)
    )
    if args.bundle:
        save_bundle(ds, args.bundle)
        print(f"[数据] 已缓存到 {args.bundle}")
    return ds


def make_env_for(ds, args, detector_kwargs: Dict[str, Any]) -> CyberDefenseEnv:
    det = make_detector(args.detector, **detector_kwargs)
    cfg = EnvConfig(
        num_agents=args.num_agents,
        max_steps=args.max_steps,
        include_focus_onehot=not args.no_focus_onehot,
        detector_embed_dim=args.embed_dim,
        reward_specialist_bonus=args.specialist_bonus,
        reward_detector_align=args.detector_align,
        seed=args.seed,
    )
    env = CyberDefenseEnv(
        X_data=ds.X_train,
        y_labels=ds.y_train,
        detector=det,
        spec=ds.spec or ClassSpec.from_preset(args.class_preset),
        config=cfg,
        X_normal_for_detector=ds.X_normal_train,
        X_test=ds.X_test,
        y_test=ds.y_test,
    )
    return env


# --------------------------------------------------------------------------- 训练
def train(args) -> Dict[str, Any]:
    out_dir = args.out or os.path.join("outputs", args.detector)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 78)
    print(f"  MADDPG 多智能体网络防御训练   |  检测机制 = {args.detector}")
    print("=" * 78)

    t_start = time.time()
    ds = build_or_load_dataset(args)
    print(ds.summary())

    detector_kwargs = resolve_detector_kwargs(args.detector, args.embed_dim)
    env = make_env_for(ds, args, detector_kwargs)
    env_fit_start = time.time()
    print(f"[环境] 检测器 {args.detector} 训练/标定完成，用时 {time.time() - env_fit_start:.1f}s")
    print(env.describe())

    agent = MADDPG(
        MADDPGConfig(
            num_agents=env.num_agents,
            obs_dim=env.obs_dim,
            num_classes=env.num_classes,
            action_dim=env.action_dim,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            gamma=args.gamma,
            tau=args.tau,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            updates_per_step=args.updates_per_step,
            warmup_steps=args.warmup_steps,
            noise_scale=args.noise_scale,
            noise_min=args.noise_min,
            noise_decay=args.noise_decay,
            seed=args.seed,
        )
    )
    print(f"[算法] {agent.describe()}")

    history: List[Dict[str, Any]] = []
    best_metric = -np.inf
    best_path = os.path.join(out_dir, "best.pt")
    last_path = os.path.join(out_dir, "last.pt")

    for ep in range(1, args.episodes + 1):
        obs = env.reset()
        agent.reset_noise()
        ep_start = time.time()
        ep_reward = np.zeros(env.num_agents, dtype=np.float64)
        ep_loss_c = np.zeros(env.num_agents)
        ep_loss_a = np.zeros(env.num_agents)
        n_upd = 0

        while True:
            actions, cls, raws = agent.select_actions_raw(obs)
            next_obs, rew, done, info = env.step_with_classes(actions, cls)
            agent.store(obs, raws, rew, next_obs, np.full(env.num_agents, float(done)), cls)
            ep_reward += rew
            env.record_step_rewards(rew)

            if agent.ready():
                losses = agent.update()
                ep_loss_c += np.asarray(losses["critic"])
                ep_loss_a += np.asarray(losses["actor"])
                n_upd += 1

            obs = next_obs
            if done:
                break

        m = env.episode_metrics()
        rec = {
            "episode": ep,
            "mean_reward": float(ep_reward.mean()),
            "min_reward": float(ep_reward.min()),
            "max_reward": float(ep_reward.max()),
            "per_agent_reward": ep_reward.tolist(),
            "train_accuracy": m["mean_accuracy"],
            "train_f1": m["mean_f1"],
            "detector_f1_running": m["detector"]["f1"],
            "critic_loss": float(ep_loss_c.mean() / max(n_upd, 1)),
            "actor_loss": float(ep_loss_a.mean() / max(n_upd, 1)),
            "updates": agent.total_updates,
            "buffer": len(agent.buffer),
            "noise_scale": agent.noise_scale,
            "steps_per_sec": round(env.cfg.max_steps / max(time.time() - ep_start, 1e-6), 1),
        }
        history.append(rec)

        # ---- 周期性评估 + 保存 ----
        should_eval = (ep % args.eval_every == 0) or ep == args.episodes
        eval_summary = None
        if should_eval:
            res = evaluate_agents(
                env,
                lambda o: agent.select_actions(o, deterministic=True),
                ds.X_test,
                ds.y_test,
                max_samples=args.eval_samples,
                seed=args.seed,
            )
            eval_summary = {
                "vote_accuracy": res.vote["accuracy"],
                "vote_macro_f1": res.vote["macro_f1"],
                "vote_kappa": res.vote["kappa"],
                "vote_detection_rate": res.vote["detection_rate"],
                "vote_false_alarm_rate": res.vote["false_alarm_rate"],
                "mean_agent_accuracy": float(np.mean([a["accuracy"] for a in res.agents])),
                "motif": "vote",
            }
            rec["eval"] = eval_summary
            score = res.vote["macro_f1"]
            if score > best_metric:
                best_metric = score
                agent.save(best_path)
                with open(os.path.join(out_dir, "eval_best.json"), "w", encoding="utf-8") as f:
                    json.dump(_serializable_eval(res), f, ensure_ascii=False, indent=2)

        if ep % args.log_every == 0 or ep == 1 or ep == args.episodes or should_eval:
            msg = (
                f"[ep {ep:>4}/{args.episodes}] reward={rec['mean_reward']:>8.3f}  "
                f"train_acc={rec['train_accuracy']:.3f}  buffer={rec['buffer']:>6}  "
                f"updates={rec['updates']:>6}  noise={rec['noise_scale']:.3f}"
            )
            if eval_summary:
                msg += (
                    f"  | test_vote_acc={eval_summary['vote_accuracy']:.4f} "
                    f"F1={eval_summary['vote_macro_f1']:.4f} "
                    f"Kappa={eval_summary['vote_kappa']:.4f}"
                )
            print(msg, flush=True)

    agent.save(last_path)
    with open(os.path.join(out_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    # ---- 最终评估 ----
    print("\n" + "=" * 78)
    print("  最终测试集评估（分散式执行，确定性策略）")
    print("=" * 78)
    res = evaluate_agents(
        env,
        lambda o: agent.select_actions(o, deterministic=True),
        ds.X_test,
        ds.y_test,
        max_samples=args.eval_samples,
        seed=args.seed,
    )
    print(res.summary())
    with open(os.path.join(out_dir, f"eval_{args.detector}.json"), "w", encoding="utf-8") as f:
        json.dump(_serializable_eval(res), f, ensure_ascii=False, indent=2)

    if args.plots:
        try:
            save_plots(history, res, out_dir)
            print(f"[绘图] 已保存到 {out_dir}/curves.png")
        except Exception as e:  # 绘图失败不影响训练结果
            print(f"[绘图] 跳过（{type(e).__name__}: {e}）")

    print(
        f"\n[完成] 检测机制={args.detector}  总用时 {(time.time() - t_start) / 60:.1f} 分钟\n"
        f"       最优投票 F1={best_metric:.4f}  权重：{best_path}\n"
        f"       评估报告：{os.path.join(out_dir, f'eval_{args.detector}.json')}"
    )
    return {"history": history, "eval": _serializable_eval(res), "out_dir": out_dir}


def _serializable_eval(res) -> Dict[str, Any]:
    """把评估结果转成 JSON 友好的 dict（去掉大数组，保留混淆矩阵）。"""
    out: Dict[str, Any] = {"n_samples": res.n_samples}
    out["agents"] = [
        {
            "name": a["name"],
            "id": a["id"],
            "focus": a["focus"],
            "accuracy": a["accuracy"],
            "macro_f1": a["macro_f1"],
            "macro_precision": a["macro_precision"],
            "macro_recall": a["macro_recall"],
            "kappa": a["kappa"],
            "detection_rate": a["detection_rate"],
            "false_alarm_rate": a["false_alarm_rate"],
            "mean_reward": a["mean_reward"],
            "confusion_matrix": a["confusion_matrix"].tolist(),
            "per_class": a["per_class"],
        }
        for a in res.agents
    ]
    out["vote"] = {
        k: (v.tolist() if isinstance(v, np.ndarray) else v)
        for k, v in res.vote.items()
        if k not in ("labels",)
    }
    out["detector"] = res.detector
    out["labels"] = res.vote["labels"]
    return out


def save_plots(history: List[Dict[str, Any]], res, out_dir: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = [h["episode"] for h in history]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    ax = axes[0, 0]
    ax.plot(ep, [h["mean_reward"] for h in history], label="mean reward")
    ax.plot(ep, [h["min_reward"] for h in history], alpha=0.4, label="min agent reward")
    ax.set_title("平均奖励（训练）")
    ax.set_xlabel("episode")
    ax.legend()

    ax = axes[0, 1]
    ax.plot(ep, [h["train_accuracy"] for h in history], label="train accuracy")
    ev_ep = [h["episode"] for h in history if "eval" in h]
    ax.plot(ev_ep, [h["eval"]["vote_accuracy"] for h in history if "eval" in h],
            "o-", label="test vote accuracy")
    ax.set_title("分类准确率")
    ax.set_xlabel("episode")
    ax.legend()

    ax = axes[1, 0]
    ax.plot(ep, [h["critic_loss"] for h in history], label="critic loss (每个智能体均值)")
    ax2 = ax.twinx()
    ax2.plot(ep, [h["actor_loss"] for h in history], "r-", alpha=0.6, label="actor loss")
    ax.set_title("损失（论文 Fig.12）")
    ax.set_xlabel("episode")
    ax.legend(loc="upper left")

    ax = axes[1, 1]
    cm = np.asarray(res.vote["confusion_matrix"])
    im = ax.imshow(cm, cmap="Blues")
    labels = res.vote["labels"]
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title("投票混淆矩阵（测试集）")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    fig.colorbar(im, ax=ax, fraction=0.046)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "curves.png"), dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MADDPG 多智能体网络防御训练")
    # 数据
    p.add_argument("--data", choices=["synthetic", "bundle", "real"], default="synthetic")
    p.add_argument("--bundle", default="data/cicids_sample.npz")
    p.add_argument("--rows-per-file", type=int, default=20000)
    p.add_argument("--synthetic-rows", type=int, default=3000)
    p.add_argument("--class-preset", default="default", choices=["default", "full", "binary"])
    p.add_argument("--normal-ratio", type=float, default=0.35)

    # 检测器（核心可插拔点）
    p.add_argument("--detector", default="if", choices=list_detectors())
    p.add_argument("--embed-dim", type=int, default=4,
                   help="统一检测器 embedding 维度，保证三种机制的观测维度一致")

    # 环境
    p.add_argument("--num-agents", type=int, default=5, help="论文 Table 2: 5")
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--no-focus-onehot", action="store_true", help="观测里不加入智能体分工 one-hot")
    p.add_argument("--specialist-bonus", type=float, default=0.5)
    p.add_argument("--detector-align", type=float, default=0.2)

    # 算法（默认对齐论文 Table 2）
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--actor-lr", type=float, default=1e-4)
    p.add_argument("--critic-lr", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=1e-3)
    p.add_argument("--buffer-size", type=int, default=500_000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--updates-per-step", type=int, default=1)
    p.add_argument("--warmup-steps", type=int, default=1000)
    p.add_argument("--noise-scale", type=float, default=0.1)
    p.add_argument("--noise-min", type=float, default=0.02)
    p.add_argument("--noise-decay", type=float, default=0.995)

    # 运行
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--eval-samples", type=int, default=5000)
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--out", default=None)
    p.add_argument("--plots", action="store_true")
    p.add_argument("--device", default=None)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    os.makedirs("outputs", exist_ok=True)
    train(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
