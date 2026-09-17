"""经验回放池：MADDPG 的共享缓冲区（论文：agents use a shared experience replay buffer）。

与原始骨架的差别：额外存储每个智能体的离散分类判定 class_idx，
因为 MADDPG 的连续动作只包含"响应"，分类判定是 actor 内部的离散决策。
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


class ReplayBuffer:
    """
    存储 (obs, act, rew, next_obs, done, class_idx)。

    形状（capacity 为环形缓冲容量）：
        obs       (C, N, obs_dim)
        act       (C, N, act_dim)
        rew       (C, N)
        class_idx (C, N)      int8
    """

    def __init__(
        self,
        capacity: int,
        num_agents: int,
        obs_dim: int,
        act_dim: int,
        seed: int = 0,
    ) -> None:
        self.capacity = int(capacity)
        self.num_agents = int(num_agents)
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.obs = np.zeros((self.capacity, self.num_agents, self.obs_dim), dtype=np.float32)
        self.act = np.zeros((self.capacity, self.num_agents, self.act_dim), dtype=np.float32)
        self.rew = np.zeros((self.capacity, self.num_agents), dtype=np.float32)
        self.next_obs = np.zeros_like(self.obs)
        self.done = np.zeros((self.capacity, self.num_agents), dtype=np.float32)
        self.class_idx = np.zeros((self.capacity, self.num_agents), dtype=np.int8)
        self.ptr = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def push(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        rew: np.ndarray,
        next_obs: np.ndarray,
        done,
        class_idx=None,
    ) -> None:
        self.obs[self.ptr] = obs
        self.act[self.ptr] = act
        self.rew[self.ptr] = rew
        self.next_obs[self.ptr] = next_obs
        if np.isscalar(done):
            self.done[self.ptr] = float(done)
        else:
            self.done[self.ptr] = np.asarray(done, dtype=np.float32).reshape(self.num_agents)
        if class_idx is not None:
            self.class_idx[self.ptr] = np.asarray(class_idx, dtype=np.int8).reshape(self.num_agents)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        if self.size < batch_size:
            raise ValueError(f"缓冲区样本不足：{self.size} < {batch_size}")
        idx = self.rng.integers(0, self.size, size=int(batch_size))
        return (
            self.obs[idx],
            self.act[idx],
            self.rew[idx],
            self.next_obs[idx],
            self.done[idx],
            self.class_idx[idx],
        )

    def __len__(self) -> int:
        return self.size

    def state_dict(self) -> dict:
        return {
            "capacity": self.capacity,
            "num_agents": self.num_agents,
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "obs": self.obs[: self.size],
            "act": self.act[: self.size],
            "rew": self.rew[: self.size],
            "next_obs": self.next_obs[: self.size],
            "done": self.done[: self.size],
            "class_idx": self.class_idx[: self.size],
            "size": self.size,
            "ptr": self.ptr,
        }

    def load_state_dict(self, blob: dict) -> None:
        n = int(blob["size"])
        if n > self.capacity:
            raise ValueError(f"存档里有 {n} 条经验，超过当前容量 {self.capacity}")
        self.obs[:n] = blob["obs"]
        self.act[:n] = blob["act"]
        self.rew[:n] = blob["rew"]
        self.next_obs[:n] = blob["next_obs"]
        self.done[:n] = blob["done"]
        self.class_idx[:n] = blob["class_idx"]
        self.size = n
        self.ptr = int(blob["ptr"])
