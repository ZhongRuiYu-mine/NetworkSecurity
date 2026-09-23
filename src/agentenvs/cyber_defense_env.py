"""多智能体网络防御环境（MADDPG / CTDE）。

对应论文 System Model 一节：
  Environment     动态网络流量环境（正常 + 恶意流），用 Gym 风格接口实现
  Agents          N 个智能体 {A1..AN}，每个"专注"一类威胁（DDoS / PortScan / Bot ...）
  Observation O_i 局部观测：流量特征 + 检测器信号（可选 + 智能体身份 one-hot）
  Action A_i      分类决策（对攻击类型的判断）+ 响应策略（监控 / 限流 / 阻断）
  Reward R_i      分类正确 +1，误报漏报 -1/0，缓解策略有效性 +0.5~+1

关键解耦点
----------
本环境**完全不关心**检测算法内部实现，只依赖 `DetectorBase` 契约：
    detector.fit(X_normal)      -> 自训练/标定
    detector.score_batch(X)     -> DetectorOutput(score, flag, trust, embedding)
    detector.signal_dim         -> 信号维度（环境启动时校验）
因此 IF / Kalman / Autoencoder / 自定义检测器都能直接替换（见 agentenvs/detectors/）。

动作空间（每个智能体）
---------------------
    class_logits : (num_classes,)  实数，argmax = 攻击类型判定（离散决策，走优势加权）
    response     : (1,)            tanh 输出，按 3 个等宽桶解码成响应等级
                                   in-classification【这里只保留 response 作为可微连续动作】
所以 DDPG actor 的连续动作维度 action_dim = 1，分类判定作为 actor 的"内部离散决策"，
通过 critic 的优势值反传到分类头上（见 algorithms/maddpg.py 的 update 说明）。

观测空间（每个智能体，float32，已裁剪到 [0,1]）
----------------------------------------------
    [0 : d]                       归一化流量特征
    [d : d+signal_dim]            检测器信号（score / flag / trust / embedding）
    [d+signal_dim : +num_classes] one-hot(focus)  —— 仅 include_focus_onehot=True
                                   给每个智能体一个"我是谁"的隐式分工信号
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .detectors.base import DetectorBase, DetectorOutput, coerce_detector
from .taxonomy import ClassSpec, resolve_specialization

try:  # gym/gymnasium 只是可选的"外壳"，不装也能用（gymnasium 是维护中的分支）
    try:
        import gymnasium as gym
        from gymnasium import spaces
    except Exception:
        import gym  # type: ignore[no-redef]
        from gym import spaces  # type: ignore[no-redef]

    _GYM_BASE = gym.Env
except Exception:  # pragma: no cover
    gym = None
    spaces = None

    class _GYM_BASE:  # type: ignore[no-redef]
        """没有 gym 时的最小替身。"""

        metadata: Dict[str, Any] = {}


# --------------------------------------------------------------------------- 配置
@dataclass
class EnvConfig:
    """环境超参数。对齐论文 Table 2 的设定（agents=5 等）。"""

    num_agents: int = 5
    max_steps: int = 100                 # 每个 episode 处理多少条流量

    # ---- 奖励权重（论文：正确 +1 / 误报漏报 -1 或 0 / 缓解有效性 +0.5 ~ +1）----
    reward_correct_class: float = 1.0
    reward_wrong_class: float = -1.0
    reward_specialist_bonus: float = 0.5   # 命中本智能体专精类别时的额外奖励
    reward_response_attack: Tuple[float, ...] = (0.5, 0.75, 1.0)   # 攻击流量：监控/限流/阻断
    reward_response_benign: Tuple[float, ...] = (0.5, -0.25, -1.0)  # 良性流量：越激进罚越多
    reward_detector_align: float = 0.2     # 与检测器告警一致/不一致的塑形奖励
    reward_step_cost: float = 0.0          # 每步固定开销（可为负，惩罚过度动作）

    # ---- 观测构造 ----
    include_focus_onehot: bool = True
    normalize_obs: bool = True
    detector_embed_dim: Optional[int] = None   # None -> 使用 detector.embed_dim
    obs_noise_std: float = 0.0                 # 观测噪声（模拟部分可观测）

    # ---- 其它 ----
    shuffle: bool = True
    seed: Optional[int] = None
    agents: Optional[Sequence[Dict]] = None    # 自定义分工；None -> 自动按类别分配


# --------------------------------------------------------------------------- 环境
class CyberDefenseEnv(_GYM_BASE):
    """多智能体网络防御环境（CTDE：集中式训练 / 分散式执行）。"""

    metadata = {"render.modes": []}

    def __init__(
        self,
        X_data: np.ndarray,
        y_labels: np.ndarray,
        detector: Union[str, DetectorBase, None] = "if",
        spec: Optional[ClassSpec] = None,
        config: Optional[EnvConfig] = None,
        X_normal_for_detector: Optional[np.ndarray] = None,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        auto_fit_detector: bool = True,
        **cfg_overrides: Any,
    ) -> None:
        """
        参数
        ----
        X_data, y_labels      训练用（已归一化）流量特征与类别标签，y=0 表示 BENIGN
        detector              检测器：字符串 key（"if"/"kalman"/"autoencoder"/"null"）、
                              DetectorBase 实例，或实现了 detect() 的任意对象。**核心扩展点**
        spec                  类别空间；None 时按 y_labels 推断
        X_normal_for_detector 训练检测器用的良性流量；None 时从 X_data 里取 y==0 的部分
        auto_fit_detector     是否自动 fit 检测器（已 fit 过的可以关掉）
        """
        X_data = np.asarray(X_data, dtype=np.float32)
        y_labels = np.asarray(y_labels).astype(np.int64).ravel()
        if X_data.ndim != 2:
            raise ValueError(f"X_data 必须是二维数组，收到 {X_data.shape}")
        if len(X_data) != len(y_labels):
            raise ValueError("X_data 与 y_labels 长度不一致")

        self.cfg = config or EnvConfig(**cfg_overrides)
        if config is not None and cfg_overrides:
            for k, v in cfg_overrides.items():
                setattr(self.cfg, k, v)
        self.rng = np.random.default_rng(self.cfg.seed if self.cfg.seed is not None else 0)

        # ---------- 数据与类别 ----------
        self.X_data = X_data
        self.y_labels = y_labels
        if spec is None:
            spec = ClassSpec(attack_classes=[f"Attack{i}" for i in range(1, int(y_labels.max()) + 1)])
        self.spec = spec
        self.num_classes = spec.num_classes

        # ---------- 智能体分工 ----------
        self.agents = resolve_specialization(self.cfg.agents, self.cfg.num_agents, spec)
        self.num_agents = self.cfg.num_agents
        self.focus = np.array([int(a["focus"]) for a in self.agents], dtype=np.int64)

        # ---------- 检测器（可插拔） ----------
        self.detector: DetectorBase = coerce_detector(detector)
        det_embed = int(getattr(self.detector, "embed_dim", 0))
        if self.cfg.detector_embed_dim is None:
            self.detector_embed_dim = det_embed
        else:
            want = int(self.cfg.detector_embed_dim)
            if det_embed < want:
                # 环境已指定统一维度 -> 直接把这个维度应用到检测器上，
                # 而不是静默零填充（零填充会让"统一维度做对比实验"变成假象）。
                # 未训练的检测器实例可以安全应用；已训练/共享的实例则报错，避免改坏它。
                if self._looks_fitted(self.detector) and not auto_fit_detector:
                    raise ValueError(
                        f"检测器 {type(self.detector).__name__} 已训练（embed_dim={det_embed}），"
                        f"但 EnvConfig(detector_embed_dim={want}) 要求 {want} 维 embedding；"
                        f"请用 make_detector(..., embed_dim={want}) 重新构造检测器后再传入。"
                    )
                setattr(self.detector, "embed_dim", want)
            self.detector_embed_dim = want
        self.detector_signal_dim = 3 + self.detector_embed_dim
        self._detector_fitted = self._looks_fitted(self.detector)
        if auto_fit_detector and not self._detector_fitted:
            Xn = X_normal_for_detector
            if Xn is None:
                Xn = X_data[y_labels == 0]
            if len(Xn) == 0:
                raise ValueError("没有良性样本可用于 fit 检测器")
            self.fit_detector(Xn, X_test=X_test, y_test=y_test)

        # ---------- 观测 / 动作维度 ----------
        self.feat_dim = int(X_data.shape[1])
        self.obs_dim = (
            self.feat_dim
            + self.detector_signal_dim
            + (self.num_classes if self.cfg.include_focus_onehot else 0)
        )
        self.action_dim = 1                 # 连续响应动作（分类头是 actor 内部离散决策）
        self.num_responses = 3              # 监控 / 限流 / 阻断

        if spaces is not None:
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(self.num_agents, self.action_dim), dtype=np.float32
            )
            self.observation_space = spaces.Box(
                low=0.0, high=1.0, shape=(self.num_agents, self.obs_dim), dtype=np.float32
            )

        # ---------- 运行时状态 ----------
        self.current_step = 0
        self.current_idx = 0
        self._episode_idx: Optional[np.ndarray] = None
        self._signal: Optional[np.ndarray] = None      # 当前步每个智能体的信号向量（共享）
        self._detector_out: Optional[DetectorOutput] = None
        self._base_obs_cache: Optional[np.ndarray] = None
        self.stats = self._fresh_stats()

    # ================================================================== 检测器接口
    @staticmethod
    def _looks_fitted(det: Any) -> bool:
        for attr in ("model", "x", "_fitted"):
            if getattr(det, attr, None) is not None:
                return True
        return False

    def fit_detector(
        self,
        X_normal: np.ndarray,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
    ) -> DetectorBase:
        """
        训练/标定检测器。**换检测机制时唯一需要调用的地方**：
            env.fit_detector(X_normal)
        """
        y_bin = None
        if y_test is not None:
            y_bin = (np.asarray(y_test) != 0).astype(int)
        self.detector.fit(np.asarray(X_normal, dtype=np.float32), X_test=X_test, y_test=y_bin)
        self._detector_fitted = True
        return self.detector

    def set_detector(self, detector: Union[str, DetectorBase], refit: bool = True) -> None:
        """
        运行期热替换检测器（三种机制做 A/B 对比时很有用）。

            env.set_detector("kalman")     # 维度一致即可无缝切换

        要求新检测器的 `embed_dim` 与环境的 `detector_embed_dim` 一致：
          * 环境是在 `EnvConfig(detector_embed_dim=k)` 下建的 -> k 是硬约束，
            新检测器必须能提供 k 维 embedding（不够会报错，不会静默零填充）；
          * 环境建的时候没指定（None）-> 用新检测器自己的 embed_dim，
            但那样 obs_dim 会变，已有策略权重不再兼容。
        """
        old_dim = self.detector_signal_dim
        new_det = coerce_detector(detector)
        new_embed = int(getattr(new_det, "embed_dim", 0))

        if self.cfg.detector_embed_dim is not None:
            want = int(self.cfg.detector_embed_dim)
            if new_embed < want:
                raise ValueError(
                    f"环境固定 detector_embed_dim={want}，但新检测器 {type(new_det).__name__} "
                    f"只能提供 {new_embed} 维 embedding；请改用 "
                    f"make_detector(..., embed_dim={want}) 构造的检测器。"
                )
            new_idx_dim = 3 + want
        else:
            self.detector_embed_dim = new_embed
            new_idx_dim = 3 + new_embed

        if new_idx_dim != old_dim:
            raise ValueError(
                f"新检测器信号维度 {new_idx_dim} != 原维度 {old_dim}；"
                f"请用 detector_embed_dim / embed_dim 统一到相同维度后再热替换。"
            )
        self.detector = new_det
        if refit:
            Xn = self.X_data[self.y_labels == 0]
            self.fit_detector(Xn)
        else:
            self._detector_fitted = self._looks_fitted(self.detector)
        return None

    # ================================================================== gym 接口
    def seed(self, seed: Optional[int] = None) -> List[int]:
        self.rng = np.random.default_rng(seed if seed is not None else 0)
        return [int(seed or 0)]

    def reset(self, *args: Any, **kwargs: Any):
        """开始一个新 episode（重新打乱流量顺序，重置有状态检测器）。"""
        self.current_step = 0
        self.stats = self._fresh_stats()
        self.detector.reset()

        n = len(self.X_data)
        if self.cfg.shuffle:
            self._episode_idx = self.rng.permutation(n)
        else:
            self._episode_idx = np.arange(n)
        self.current_idx = int(self._episode_idx[0])

        self._refresh_signal()
        obs = self._get_observations()
        if gym is not None and hasattr(self, "observation_space"):
            return obs
        return obs

    def step(
        self,
        actions: np.ndarray,
        class_idx: Optional[np.ndarray] = None,
        response_idx: Optional[np.ndarray] = None,
    ):
        """
        actions      : (num_agents, action_dim) 连续响应动作
        class_idx    : (num_agents,) 可选的分类判定；不传时从 actions 里 argmax 解码
                       （MADDPG 会把 actor 分类头的 argmax 传进来，见 step_with_classes）

        返回符合论文/Gym 约定的 (obs, rewards, done, info)；
        info 里额外给出每个智能体的分类判定、响应等级和评估指标。
        """
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim == 1:
            actions = actions.reshape(self.num_agents, -1)
        if actions.shape[0] != self.num_agents:
            raise ValueError(f"动作数量 {actions.shape[0]} != 智能体数量 {self.num_agents}")

        if class_idx is None or response_idx is None:
            dec_cls, dec_resp = self.decode_actions(actions)
        else:
            dec_cls, dec_resp = None, None
        if class_idx is None:
            class_idx = dec_cls
        if response_idx is None:
            response_idx = dec_resp
        class_idx = np.asarray(class_idx, dtype=np.int64).ravel()
        response_idx = np.asarray(response_idx, dtype=np.int64).ravel()
        true_label = int(self.y_labels[self.current_idx])
        det_flag = float(self._signal[0, 1]) if self._signal is not None else 0.0
        det_score = float(self._signal[0, 0]) if self._signal is not None else 0.0

        rewards = np.zeros(self.num_agents, dtype=np.float32)
        cls_rewards = np.zeros(self.num_agents, dtype=np.float32)
        resp_rewards = np.zeros(self.num_agents, dtype=np.float32)
        for i in range(self.num_agents):
            rc, rr = self._reward_for_agent(
                i, int(class_idx[i]), int(response_idx[i]), true_label, det_flag
            )
            rewards[i] = rc + rr + self.cfg.reward_step_cost
            cls_rewards[i] = rc
            resp_rewards[i] = rr

        self._update_stats(class_idx, response_idx, true_label, det_flag)

        # ---- 推进 ----
        self.current_step += 1
        done = self.current_step >= self.cfg.max_steps
        if not done:
            self.current_idx = int(self._episode_idx[self.current_step % len(self._episode_idx)])
            self._refresh_signal()
            next_obs = self._get_observations()
        else:
            next_obs = self._get_observations()   # done 时仍返回最后一次观测，便于 bootstrapping

        info = {
            "true_label": true_label,
            "true_class": self.spec.name_of(true_label),
            "class_pred": class_idx.copy(),
            "class_pred_names": [self.spec.name_of(int(c)) for c in class_idx],
            "response": response_idx.copy(),
            "detector_score": det_score,
            "detector_flag": int(det_flag),
            "rewards_class": cls_rewards,
            "rewards_response": resp_rewards,
            "mean_reward": float(rewards.mean()),
            "step": self.current_step,
            "agent_names": [a["name"] for a in self.agents],
            "focus": self.focus.copy(),
        }
        return next_obs, rewards, done, info

    # ================================================================== 动作编解码
    def decode_actions(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        连续动作 -> (类别索引, 响应等级)。

        支持两种动作布局：
          * action_dim == 1        : 只有响应通道；类别由 actor 另行给出（推荐，见 MADDPG）
          * action_dim > 1         : [class_logits..., response]，按 argmax 取类别
                                     （兼容原始骨架里"分类+响应拼在一个连续动作"的设计）
        """
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)
        if actions.shape[1] == 1:
            # 只有响应通道：类别判定由 MADDPG 的分类头给出（step 的 class_idx 参数）；
            # 若调用方没给，则退化为 0（BENIGN），便于用纯随机动作冒烟测试。
            class_idx = np.zeros(actions.shape[0], dtype=np.int64)
            response_idx = self.bin_response(actions[:, 0])
            return class_idx, response_idx
        n_cls = actions.shape[1] - 1
        class_idx = np.argmax(actions[:, :n_cls], axis=1).astype(np.int64)
        response_idx = self.bin_response(actions[:, -1])
        return class_idx, response_idx

    @property
    def num_responses_(self) -> int:
        return self.num_responses

    def bin_response(self, r: np.ndarray) -> np.ndarray:
        """把 [-1,1] 的连续响应动作等宽分到 3 档：0=监控, 1=限流, 2=阻断。"""
        r = np.clip(np.asarray(r, dtype=np.float32).ravel(), -1.0, 1.0)
        edges = np.linspace(-1.0, 1.0, self.num_responses + 1)
        idx = np.digitize(r, edges[1:-1], right=False)
        return np.clip(idx, 0, self.num_responses - 1).astype(np.int64)

    def response_center(self, idx: int) -> float:
        """响应等级 -> 该档中心对应的连续动作值（用于"确定性执行"）。"""
        n = self.num_responses
        step = 2.0 / n
        return float(-1.0 + step * (idx + 0.5))

    # ================================================================== 第 2 步：带分类判定的 step
    def step_with_classes(
        self,
        actions: np.ndarray,
        class_idx: np.ndarray,
        response_idx: Optional[np.ndarray] = None,
        *,
        auto_reset: bool = False,
    ):
        """
        MADDPG 训练/评估专用入口：同时接收
            actions      (N, 1)  连续响应动作
            class_idx    (N,)    每个智能体的攻击类型判定（来自 actor 的分类头）
            response_idx (N,)    可选的响应档位；不传则由 actions 解码
        这样分类判定（离散）与响应（连续）可以分开处理，
        连续部分照常走 DDPG 的可微路径，离散部分靠 critic 优势值加权。
        """
        class_idx = np.asarray(class_idx, dtype=np.int64).ravel()
        if class_idx.shape[0] != self.num_agents:
            raise ValueError(f"class_idx 长度 {class_idx.shape[0]} != 智能体数量 {self.num_agents}")
        if response_idx is None:
            response_idx = self.bin_response(
                np.asarray(actions, dtype=np.float32).reshape(self.num_agents, -1)[:, -1]
            )
        response_idx = np.asarray(response_idx, dtype=np.int64).ravel()
        out = self.step(actions, class_idx=class_idx, response_idx=response_idx)
        if auto_reset and out[2]:
            out = (self.reset(),) + out[1:]
        return out

    # ================================================================== 奖励
    def _reward_for_agent(
        self, i: int, cls: int, resp: int, true_label: int, det_flag: float
    ) -> Tuple[float, float]:
        """
        返回 (分类奖励, 响应奖励)，对应论文 Reward function R_i 的分解：
          * 正确识别攻击类型            +1（专注类别命中再 +reward_specialist_bonus）
          * 误报 / 漏报                 -1
          * 缓解策略有效性              监控/限流/阻断 = +0.5 / +0.75 / +1.0
                                        （良性流量上越激进惩罚越重）
          * 与检测器告警的一致性塑形     ±reward_detector_align
        """
        cfg = self.cfg
        r_cls = cfg.reward_correct_class if cls == true_label else cfg.reward_wrong_class
        if cls == true_label and cls == int(self.focus[i]):
            r_cls += cfg.reward_specialist_bonus

        resp = int(np.clip(resp, 0, self.num_responses - 1))
        r_resp = float(
            cfg.reward_response_benign[resp] if true_label == 0 else cfg.reward_response_attack[resp]
        )
        agent_says_attack = 1.0 if cls != 0 else 0.0
        r_resp += (
            cfg.reward_detector_align
            if agent_says_attack == float(det_flag)
            else -cfg.reward_detector_align
        )
        return float(r_cls), float(r_resp)

    # ================================================================== 观测
    def _refresh_signal(self) -> None:
        """对当前这条流量调用检测器，缓存信号向量。"""
        x = self.X_data[self.current_idx : self.current_idx + 1]
        out = self.detector.score_batch(x)
        self._detector_out = out
        self._signal = out.as_signal(self.detector_embed_dim)   # (1, signal_dim)
        self._base_obs_cache = None

    def _base_observation(self) -> np.ndarray:
        """所有智能体共享的观测主干：(feat_dim + signal_dim,)。"""
        if self._base_obs_cache is not None:
            return self._base_obs_cache
        x = self.X_data[self.current_idx]
        sig = self._signal[0] if self._signal is not None else np.zeros(
            self.detector_signal_dim, dtype=np.float32
        )
        base = np.concatenate([x, sig]).astype(np.float32)
        if not self.cfg.normalize_obs:
            # 不裁剪时做标准化，避免不同量纲差异过大
            base = (base - base.mean()) / (base.std() + 1e-8)
        self._base_obs_cache = base
        return base

    def _get_observations(self) -> np.ndarray:
        base = self._base_observation()
        obs = np.tile(base[None, :], (self.num_agents, 1)).astype(np.float32)
        if self.cfg.include_focus_onehot:
            focus_oh = np.zeros((self.num_agents, self.num_classes), dtype=np.float32)
            valid = (self.focus >= 0) & (self.focus < self.num_classes)
            focus_oh[np.arange(self.num_agents)[valid], self.focus[valid]] = 1.0
            obs = np.concatenate([obs, focus_oh], axis=1)
        if self.cfg.obs_noise_std > 0:
            obs = obs + self.rng.normal(0.0, self.cfg.obs_noise_std, size=obs.shape).astype(np.float32)
        if self.cfg.normalize_obs:
            obs = np.clip(obs, 0.0, 1.0)
        return obs.astype(np.float32)

    # ================================================================== 指标
    def _fresh_stats(self) -> Dict[str, Any]:
        return {
            "steps": 0,
            "agent_cls_correct": np.zeros(self.num_agents, dtype=np.float64),
            "agent_is_attack_pred": np.zeros(self.num_agents, dtype=np.float64),
            "agent_attack_correct": np.zeros(self.num_agents, dtype=np.float64),
            "agent_benign_called_attack": np.zeros(self.num_agents, dtype=np.float64),
            "episode_rewards": np.zeros(self.num_agents, dtype=np.float64),
            "true_attack": 0,
            "true_benign": 0,
            "detector_tp": 0,
            "detector_fp": 0,
            "detector_fn": 0,
            "detector_tn": 0,
        }

    def record_episode_rewards(self, rewards: np.ndarray) -> None:
        """记录一个 episode 各智能体的累计回报（训练循环里调用）。"""
        self.stats["episode_rewards"] = np.asarray(rewards, dtype=np.float64).copy()

    def record_step_rewards(self, rewards: np.ndarray) -> None:
        """把每一步的奖励累加进 episode 回报。"""
        self.stats["episode_rewards"] = self.stats["episode_rewards"] + np.asarray(
            rewards, dtype=np.float64
        )

    def _update_stats(
        self, class_idx: np.ndarray, response_idx: np.ndarray, true_label: int, det_flag: float
    ) -> None:
        s = self.stats
        s["steps"] += 1
        is_attack = true_label != 0
        if is_attack:
            s["true_attack"] += 1
        else:
            s["true_benign"] += 1
        if det_flag > 0.5 and is_attack:
            s["detector_tp"] += 1
        elif det_flag > 0.5 and not is_attack:
            s["detector_fp"] += 1
        elif det_flag <= 0.5 and is_attack:
            s["detector_fn"] += 1
        else:
            s["detector_tn"] += 1

        for i in range(self.num_agents):
            pred = int(class_idx[i])
            s["agent_cls_correct"][i] += float(pred == true_label)
            s["agent_is_attack_pred"][i] += float(pred != 0)
            if pred != 0:
                if is_attack and pred == true_label:
                    s["agent_attack_correct"][i] += 1
                if not is_attack:
                    s["agent_benign_called_attack"][i] += 1

    def episode_metrics(self) -> Dict[str, Any]:
        """把 stats 折算成 Accuracy / Precision / Recall / F1 / 检测率 等指标。"""
        s = self.stats
        n = max(s["steps"], 1)
        per_agent = []
        for i in range(self.num_agents):
            tp = s["agent_attack_correct"][i]
            fp = s["agent_benign_called_attack"][i]
            fn = s["true_attack"] - tp
            prec = tp / (tp + fp + 1e-9)
            rec = tp / (tp + fn + 1e-9)
            per_agent.append(
                {
                    "agent": self.agents[i]["name"],
                    "focus": self.spec.name_of(int(self.focus[i])),
                    "accuracy": s["agent_cls_correct"][i] / n,
                    "precision": prec,
                    "recall": rec,
                    "f1": 2 * prec * rec / (prec + rec + 1e-9),
                    "flag_rate": s["agent_is_attack_pred"][i] / n,
                }
            )
        tp, fp, fn, tn = s["detector_tp"], s["detector_fp"], s["detector_fn"], s["detector_tn"]
        det_prec = tp / (tp + fp + 1e-9)
        det_rec = tp / (tp + fn + 1e-9)
        return {
            "steps": s["steps"],
            "mean_accuracy": float(np.mean([a["accuracy"] for a in per_agent])),
            "mean_f1": float(np.mean([a["f1"] for a in per_agent])),
            "episode_rewards": np.asarray(s["episode_rewards"], dtype=np.float64).copy(),
            "mean_episode_reward": float(np.mean(s["episode_rewards"])),
            "per_agent": per_agent,
            "detector": {
                "precision": det_prec,
                "recall": det_rec,
                "f1": 2 * det_prec * det_rec / (det_prec + det_rec + 1e-9),
                "false_positive_rate": fp / (fp + tn + 1e-9),
                "detection_rate": det_rec,
            },
            "class_names": self.spec.names,
        }

    # ================================================================== 辅助
    def sample_episode_classes(self) -> np.ndarray:
        """当前 episode 用到的真实类别序列（画混淆矩阵用）。"""
        idx = self._episode_idx[: self.cfg.max_steps] if self._episode_idx is not None else np.arange(len(self.y_labels))
        return self.y_labels[idx]

    def describe(self) -> str:
        lines = [
            "=" * 68,
            "CyberDefenseEnv",
            "=" * 68,
            f"  检测器          : {type(self.detector).__name__} "
            f"(key={getattr(self.detector, 'name', '?')}, fitted={self._detector_fitted})",
            f"  检测器信号维度  : {self.detector_signal_dim} (score/flag/trust"
            + (f" + {self.detector_embed_dim}D embedding)" if self.detector_embed_dim else ")"),
            f"  特征维度        : {self.feat_dim}",
            f"  智能体数量      : {self.num_agents}  (action_dim={self.action_dim}, 响应档位={self.num_responses})",
            f"  观测维度/智能体 : {self.obs_dim}",
            f"  类别空间 ({self.num_classes})   : {self.spec.names}",
        ]
        for i, a in enumerate(self.agents):
            lines.append(
                f"    - agent{i} {a['name']:<18} 专注={self.spec.name_of(int(self.focus[i]))}"
            )
        lines.append(f"  episode 长度    : {self.cfg.max_steps} 步")
        lines.append("=" * 68)
        return "\n".join(lines)


# --------------------------------------------------------------------------- 工厂
def make_env(
    dataset,
    detector: Union[str, DetectorBase, None] = "if",
    num_agents: int = 5,
    max_steps: int = 100,
    auto_fit_detector: bool = True,
    **cfg_overrides: Any,
) -> CyberDefenseEnv:
    """
    从 TrafficDataset 一步构造环境：

        ds  = load_bundle("data/cicids_sample.npz")
        env = make_env(ds, detector="kalman", num_agents=5)
    """
    cfg = EnvConfig(num_agents=num_agents, max_steps=max_steps, **cfg_overrides)
    return CyberDefenseEnv(
        X_data=dataset.X_train,
        y_labels=dataset.y_train,
        detector=detector,
        spec=dataset.spec or ClassSpec.from_preset("default"),
        config=cfg,
        X_normal_for_detector=dataset.X_normal_train,
        X_test=dataset.X_test,
        y_test=dataset.y_test,
        auto_fit_detector=auto_fit_detector,
    )
