"""MADDPG 的网络结构：分散式 Actor（含分类头）+ 集中式 Critic。"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn


def _mlp(dims: Sequence[int], activation=nn.ReLU, out_activation=None, gain: float = 1.0) -> nn.Sequential:
    layers: List[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        is_last = i == len(dims) - 2
        if is_last:
            if out_activation is not None:
                layers.append(out_activation())
        else:
            layers.append(activation())
    net = nn.Sequential(*layers)
    # 最后一层做正交初始化 + gain 缩放，DDPG 常用技巧，能让初期输出接近 0
    last = [m for m in net if isinstance(m, nn.Linear)][-1]
    nn.init.orthogonal_(last.weight, gain=gain)
    nn.init.constant_(last.bias, 0.0)
    return net


class Actor(nn.Module):
    """
    分散式 actor：只用**本地观测** o_i 决策（decentralized execution）。

    输出两个头：
        class_logits : (num_classes,)  离散的攻击类型判定（不直接反传，靠 critic 优势加权）
        response     : (1,)            tanh 到 [-1,1] 的连续响应动作（走 DDPG 可微路径）
    """

    def __init__(
        self,
        obs_dim: int,
        num_classes: int,
        hidden_dims: Sequence[int] = (128, 128),
        action_dim: int = 1,
        response_head: str = "tanh",
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.num_classes = int(num_classes)
        self.action_dim = int(action_dim)
        self.response_head = response_head

        trunk_dims = [self.obs_dim] + [int(h) for h in hidden_dims]
        self.trunk = _mlp(trunk_dims, gain=1.0)
        feat_dim = trunk_dims[-1]
        # 分类头的输出增益稍大，保证初期 logits 有一定分散度，
        # 策略梯度才有非零梯度（增益过小会让分类头长时间不学习）。
        self.class_head = _mlp([feat_dim, feat_dim // 2, self.num_classes], gain=0.1)
        self.response_net = _mlp([feat_dim, feat_dim // 2, self.action_dim], gain=0.01)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回 (class_logits, response_raw)。"""
        h = self.trunk(obs)
        logits = self.class_head(h)
        raw = self.response_net(h)
        return logits, raw

    def response(self, raw: torch.Tensor) -> torch.Tensor:
        if self.response_head == "tanh":
            return torch.tanh(raw)
        return raw


class Critic(nn.Module):
    """集中式 critic：输入**所有**智能体的观测与动作（centralized training）。"""

    def __init__(
        self,
        total_obs_dim: int,
        total_action_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
    ) -> None:
        super().__init__()
        dims = [total_obs_dim + total_action_dim] + [int(h) for h in hidden_dims] + [1]
        self.net = _mlp(dims, gain=1.0)

    def forward(self, all_obs: torch.Tensor, all_actions: torch.Tensor) -> torch.Tensor:
        x = torch.cat([all_obs, all_actions], dim=-1)
        return self.net(x)
