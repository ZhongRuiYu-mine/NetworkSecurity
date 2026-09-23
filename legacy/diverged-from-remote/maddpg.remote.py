"""MADDPG（Multi-Agent Deep Deterministic Policy Gradient）—— CTDE 实现。

论文对应关系
------------
* 每个智能体有独立的 actor（分散式执行），但共享集中式 critic 的输入
  （所有智能体的观测 + 动作），即 centralized training / decentralized execution。
* 目标网络 + 软更新 τ（论文 Table 2：lr=1e-4, γ=0.99, buffer=500k, batch=128, τ=0.001）。
* 共享经验回放池。

与"纯 DDPG"的一个必要调整
-------------------------
本任务的决策 = **攻击类型分类（离散）** + **响应策略（连续）**。
DDPG 只能对连续动作求梯度，所以：

    actor 输出 → class_logits（离散决策，不反传） + response（连续动作，走 ∇Q）

离散分类头用 **优势值加权的策略梯度** 学习（等价于 REINFORCE with baseline，
baseline 就是集中式 critic 的 Q 值）：

    L_class = -log π(c_i | o_i) · ( Q(s, a_1..a_N) - Q(s, a'_i, a_-i) )

其中 a'_i 是把第 i 个智能体的分类改成 sampled 类别后的联合动作。
这样既保留了 DDPG 的连续动作优势，又让"分类"这个离散决策能被 critic 评价。
（如果只想复现原始骨架那种"分类+响应拼成一个连续动作向量"的做法，
 把 num_classes 置 0、action_dim 设为 num_classes+1 即可，本实现也支持。）
"""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .networks import Actor, Critic
from .replay_buffer import ReplayBuffer


def gumbel_softmax(
    logits: torch.Tensor, tau: float = 1.0, hard: bool = False, generator: Optional[torch.Generator] = None
) -> torch.Tensor:
    """Jang et al. 的 Gumbel-Softmax（采样 + 可选的直通 one-hot）。"""
    return F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1, generator=generator)


class OUNoise:
    """Ornstein-Uhlenbeck 探索噪声（DDPG 原文用的就是它；默认关闭可用高斯）。"""

    def __init__(self, size: int, mu: float = 0.0, theta: float = 0.15, sigma: float = 0.2, seed: int = 0):
        self.size = int(size)
        self.mu = float(mu)
        self.theta = float(theta)
        self.sigma = float(sigma)
        self.state = np.ones(self.size, dtype=np.float32) * self.mu
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        self.state = np.ones(self.size, dtype=np.float32) * self.mu

    def sample(self, scale: float = 1.0) -> np.ndarray:
        dx = self.theta * (self.mu - self.state) + self.sigma * self.rng.normal(size=self.size)
        self.state = (self.state + dx).astype(np.float32)
        return (self.state * scale).astype(np.float32)


@dataclass
class MADDPGConfig:
    """对齐论文 Table 2 的 MADDPG 超参。"""

    num_agents: int = 5
    obs_dim: int = 32
    num_classes: int = 6          # 分类头输出维度（含 BENIGN）
    action_dim: int = 1           # 连续响应动作维度
    hidden_dims: Sequence[int] = (128, 128)
    critic_hidden_dims: Sequence[int] = (256, 256)

    actor_lr: float = 1e-4
    critic_lr: float = 1e-3
    gamma: float = 0.99
    tau: float = 0.001
    buffer_size: int = 500_000
    batch_size: int = 128
    updates_per_step: int = 1
    warmup_steps: int = 1_000
    grad_clip: float = 1.0

    noise_type: str = "gaussian"      # gaussian | ou
    noise_scale: float = 0.1
    noise_decay: float = 1.0          # 每轮乘性衰减
    noise_min: float = 0.02
    gumbel_tau: float = 1.0
    class_lr_scale: float = 1.0       # 分类头策略梯度项的缩放
    entropy_coef: float = 0.02        # 分类头熵正则（防止早期塌缩到单一类别）

    device: str = "auto"
    seed: int = 0
    shared_critic: bool = False       # True -> 所有智能体共用一个 critic（更像"集中式"）


class MADDPG:
    """多智能体 DDPG 训练器。"""

    def __init__(self, config: Optional[MADDPGConfig] = None, **overrides: Any) -> None:
        cfg = config or MADDPGConfig(**overrides)
        if config is not None and overrides:
            for k, v in overrides.items():
                setattr(cfg, k, v)
        self.cfg = cfg
        self.n = int(cfg.num_agents)
        self.obs_dim = int(cfg.obs_dim)
        self.num_classes = int(cfg.num_classes)
        self.action_dim = int(cfg.action_dim)

        if cfg.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(cfg.device)

        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        if cfg.shared_critic:
            critics = [Critic(self.obs_dim * self.n, self.action_dim * self.n, cfg.critic_hidden_dims).to(self.device)]
            self.critics = [critics[0] for _ in range(self.n)]
            self.target_critics = [copy.deepcopy(critics[0]).to(self.device) for _ in range(self.n)]
            self.critic_optimizers = [
                torch.optim.Adam(critics[0].parameters(), lr=cfg.critic_lr) for _ in range(self.n)
            ]
        else:
            self.critics = [
                Critic(self.obs_dim * self.n, self.action_dim * self.n, cfg.critic_hidden_dims).to(self.device)
                for _ in range(self.n)
            ]
            self.target_critics = [copy.deepcopy(c).to(self.device) for c in self.critics]
            self.critic_optimizers = [
                torch.optim.Adam(c.parameters(), lr=cfg.critic_lr) for c in self.critics
            ]

        self.actors = [
            Actor(self.obs_dim, self.num_classes, cfg.hidden_dims, self.action_dim).to(self.device)
            for _ in range(self.n)
        ]
        self.target_actors = [copy.deepcopy(a).to(self.device) for a in self.actors]
        self.actor_optimizers = [
            torch.optim.Adam(a.parameters(), lr=cfg.actor_lr) for a in self.actors
        ]

        self.buffer = ReplayBuffer(
            cfg.buffer_size, self.n, self.obs_dim, self.action_dim, seed=cfg.seed
        )
        self.noise = [
            OUNoise(self.action_dim, seed=cfg.seed + i) if cfg.noise_type == "ou" else None
            for i in range(self.n)
        ]
        self.noise_scale = float(cfg.noise_scale)

        self.total_steps = 0
        self.total_updates = 0
        self.last_losses: Dict[str, List[float]] = {"critic": [0.0] * self.n, "actor": [0.0] * self.n}

    # ------------------------------------------------------------------ 选择动作
    @torch.no_grad()
    def select_actions(
        self, obs: np.ndarray, noise_scale: Optional[float] = None, deterministic: bool = False
    ):
        """
        分散式执行：每个智能体只看自己的本地观测。

        返回 (actions (N, action_dim) 连续响应, class_idx (N,) 分类判定)
        deterministic=True 用于评估（无噪声、argmax 分类）。
        """
        obs = np.asarray(obs, dtype=np.float32)
        scale = self.noise_scale if noise_scale is None else float(noise_scale)
        actions = np.zeros((self.n, self.action_dim), dtype=np.float32)
        class_idx = np.zeros(self.n, dtype=np.int64)
        for i in range(self.n):
            o = torch.as_tensor(obs[i], dtype=torch.float32, device=self.device).unsqueeze(0)
            logits, raw = self.actors[i](o)
            class_idx[i] = int(torch.argmax(logits, dim=-1).item())
            a = self.actors[i].response(raw)
            if not deterministic:
                if self.noise[i] is not None:
                    a = a + torch.as_tensor(self.noise[i].sample(scale), device=self.device).unsqueeze(0)
                elif scale > 0:
                    a = a + torch.randn_like(a) * scale
                a = torch.clamp(a, -1.0, 1.0)
            actions[i] = a.squeeze(0).detach().cpu().numpy()
        return actions, class_idx

    @torch.no_grad()
    def select_actions_raw(self, obs: np.ndarray, noise_scale: Optional[float] = None):
        """
        训练用：返回 (动作, class_idx, raw_action)。
        raw_action 是**加噪前的线性层输出**，写进回放池让 critic 的输入与
        actor 当前策略的输出处于同一尺度（避免 target Q 被噪声污染）。
        """
        obs = np.asarray(obs, dtype=np.float32)
        scale = self.noise_scale if noise_scale is None else float(noise_scale)
        actions = np.zeros((self.n, self.action_dim), dtype=np.float32)
        raws = np.zeros((self.n, self.action_dim), dtype=np.float32)
        class_idx = np.zeros(self.n, dtype=np.int64)
        for i in range(self.n):
            o = torch.as_tensor(obs[i], dtype=torch.float32, device=self.device).unsqueeze(0)
            logits, raw = self.actors[i](o)
            class_idx[i] = int(torch.argmax(logits, dim=-1).item())
            a = self.actors[i].response(raw)
            if scale > 0:
                if self.noise[i] is not None:
                    noise = torch.as_tensor(self.noise[i].sample(scale), device=self.device).unsqueeze(0)
                else:
                    noise = torch.randn_like(a) * scale
                a = torch.clamp(a + noise, -1.0, 1.0)
            actions[i] = a.squeeze(0).cpu().numpy()
            raws[i] = raw.squeeze(0).cpu().numpy()
        return actions, class_idx, raws

    # ------------------------------------------------------------------ 存储 / 学习
    def store(self, obs, action, reward, next_obs, done, class_idx=None) -> None:
        self.buffer.push(obs, action, reward, next_obs, done, class_idx)
        self.total_steps += 1

    def ready(self) -> bool:
        return len(self.buffer) >= max(self.cfg.batch_size, self.cfg.warmup_steps)

    def update(self, batch_size: Optional[int] = None) -> Dict[str, List[float]]:
        """集中式训练：每个智能体更新自己的 critic + actor。"""
        cfg = self.cfg
        if not self.ready():
            return self.last_losses
        bs = int(batch_size or cfg.batch_size)

        for _ in range(max(1, int(cfg.updates_per_step))):
            obs, act, rew, nobs, done, class_idx = self.buffer.sample(bs)
            obs_t = torch.as_tensor(obs, device=self.device)               # (B, N, O)
            act_t = torch.as_tensor(act, device=self.device)               # (B, N, A)
            rew_t = torch.as_tensor(rew, device=self.device)               # (B, N)
            nobs_t = torch.as_tensor(nobs, device=self.device)
            done_t = torch.as_tensor(done, device=self.device)
            cls_t = torch.as_tensor(class_idx.astype(np.int64), device=self.device)  # (B, N)

            all_obs = obs_t.reshape(bs, -1)
            all_act = act_t.reshape(bs, -1)
            all_nobs = nobs_t.reshape(bs, -1)

            # ---------------- 目标联合动作（连续部分用目标 actor）----------------
            with torch.no_grad():
                target_resp = torch.stack(
                    [self.target_actors[j].response(self.target_actors[j](nobs_t[:, j, :])[1])
                     for j in range(self.n)],
                    dim=1,
                )                                                        # (B, N, A)
                target_actions = target_resp.reshape(bs, -1)

            cur_logits, cur_raw = [], []
            for j in range(self.n):
                lg, rw = self.actors[j](obs_t[:, j, :])
                cur_logits.append(lg)
                cur_raw.append(self.actors[j].response(rw))

            for i in range(self.n):
                # ---------------- Critic ----------------
                with torch.no_grad():
                    target_q = self.target_critics[i](all_nobs, target_actions)
                    y = rew_t[:, i : i + 1] + cfg.gamma * (1.0 - done_t[:, i : i + 1]) * target_q

                q = self.critics[i](all_obs, all_act)
                critic_loss = F.mse_loss(q, y)
                self.critic_optimizers[i].zero_grad(set_to_none=True)
                critic_loss.backward()
                if cfg.grad_clip:
                    nn.utils.clip_grad_norm_(self.critics[i].parameters(), cfg.grad_clip)
                self.critic_optimizers[i].step()

                # ---------------- Actor ----------------
                # (1) 连续响应部分：走标准 DDPG 梯度
                joint = torch.cat(cur_raw, dim=-1).detach()
                joint = torch.cat(
                    [joint[:, : i * self.action_dim], cur_raw[i], joint[:, (i + 1) * self.action_dim :]],
                    dim=-1,
                )
                q_resp = self.critics[i](all_obs, joint)
                loss_dpg = -q_resp.mean()

                # (2) 离散分类部分：优势加权的策略梯度（REINFORCE with baseline）
                #     L = -log π(c_i|o_i) · Â
                #     Â 用"批次内 Q 的均值"作 baseline 再标准化：Q 的量纲大约等于
                #     累积折扣回报（可到几十），若不做标准化会把分类头拉爆或压死，
                #     这是离散决策头在 MADDPG 里最容易崩的地方。
                #     另外加熵正则，避免分类头早期塌缩到单一类别（例如全判 BENIGN）。
                loss_class = torch.zeros((), device=self.device)
                loss_entropy = torch.zeros((), device=self.device)
                if self.num_classes > 0:
                    logprob = F.log_softmax(cur_logits[i], dim=-1)
                    lp_taken = logprob.gather(1, cls_t[:, i : i + 1]).squeeze(1)
                    q_det = q_resp.squeeze(1).detach()
                    adv = q_det - q_det.mean()
                    adv = adv / (adv.std() + 1e-6)
                    loss_class = -(lp_taken * adv).mean() * cfg.class_lr_scale
                    if cfg.entropy_coef > 0:
                        probs = logprob.exp()
                        entropy = -(probs * logprob).sum(dim=-1).mean()
                        loss_entropy = -cfg.entropy_coef * entropy

                actor_loss = loss_dpg + loss_class + loss_entropy
                self.actor_optimizers[i].zero_grad(set_to_none=True)
                actor_loss.backward()
                if cfg.grad_clip:
                    nn.utils.clip_grad_norm_(self.actors[i].parameters(), cfg.grad_clip)
                self.actor_optimizers[i].step()

                self.last_losses["critic"][i] = float(critic_loss.item())
                self.last_losses["actor"][i] = float(actor_loss.item())

            # ---------------- 软更新目标网络 ----------------
            for i in range(self.n):
                self._soft_update(self.actors[i], self.target_actors[i])
                self._soft_update(self.critics[i], self.target_critics[i])

            self.total_updates += 1

        # 探索噪声衰减
        self.noise_scale = max(cfg.noise_min, self.noise_scale * cfg.noise_decay)
        return self.last_losses

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        tau = self.cfg.tau
        with torch.no_grad():
            for tp, sp in zip(target.parameters(), source.parameters()):
                tp.data.mul_(1.0 - tau).add_(sp.data, alpha=tau)
            for tb, sb in zip(target.buffers(), source.buffers()):
                tb.data.copy_(sb.data)

    def reset_noise(self) -> None:
        for n in self.noise:
            if n is not None:
                n.reset()

    # ------------------------------------------------------------------ 落盘
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(
            {
                "config": vars(self.cfg),
                "actors": [a.state_dict() for a in self.actors],
                "critics": [c.state_dict() for c in self.critics],
                "target_actors": [a.state_dict() for a in self.target_actors],
                "target_critics": [c.state_dict() for c in self.target_critics],
                "total_steps": self.total_steps,
                "total_updates": self.total_updates,
                "noise_scale": self.noise_scale,
            },
            path,
        )

    def load(self, path: str) -> "MADDPG":
        blob = torch.load(path, map_location=self.device, weights_only=False)
        for i, sd in enumerate(blob["actors"]):
            self.actors[i].load_state_dict(sd)
        for i, sd in enumerate(blob["critics"]):
            self.critics[i].load_state_dict(sd)
        for i, sd in enumerate(blob["target_actors"]):
            self.target_actors[i].load_state_dict(sd)
        for i, sd in enumerate(blob["target_critics"]):
            self.target_critics[i].load_state_dict(sd)
        self.total_steps = int(blob.get("total_steps", 0))
        self.total_updates = int(blob.get("total_updates", 0))
        self.noise_scale = float(blob.get("noise_scale", self.cfg.noise_min))
        return self

    # ------------------------------------------------------------------ 调试信息
    def describe(self) -> str:
        params = sum(p.numel() for p in self.actors[0].parameters())
        cparams = sum(p.numel() for p in self.critics[0].parameters())
        return (
            f"MADDPG(agents={self.n}, obs_dim={self.obs_dim}, classes={self.num_classes}, "
            f"action_dim={self.action_dim}, device={self.device}, "
            f"actor_params={params}, critic_params={cparams}, buffer={len(self.buffer)})"
        )
