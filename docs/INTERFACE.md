# 接口文档 —— 可替换点与扩展契约

本文档描述本平台**所有可替换的接口**：换检测机制、换特征、换分工、换策略算法、换数据时
分别改哪里、契约是什么、有什么坑。

> 阅读顺序建议：第 1 章（防守方检测器接口）是最常用的扩展点；
> 第 2 章（环境）与第 3 章（内置检测器参数）是调参参照；
> 第 6 章（Recipe）是"我要做 X，改哪里"的速查表。

目录

1. [可替换点总览](#1-可替换点总览)
2. [★ 防守方检测器接口](#2--防守方检测器接口)
3. [环境接口](#3-环境接口)
4. [内置检测器参数手册](#4-内置检测器参数手册)
5. [算法接口](#5-算法接口)
6. [数据接口](#6-数据接口)
7. [Recipe：常见替换任务速查](#7-recipe常见替换任务速查)
8. [适配性检查清单](#8-适配性检查清单)
9. [错误码与排查表](#9-错误码与排查表)

---

## 1. 可替换点总览

| # | 可替换的东西 | 接口 / 契约 | 定义位置 | 替换方式 |
|---|---|---|---|---|
| 1 | **检测机制**（IF / Kalman / AE / 自研） | `DetectorBase` | `agentenvs/detectors/base.py` | `make_detector(key)` / 直接传对象 / `env.set_detector()` |
| 2 | 攻击类别空间 | `ClassSpec` | `agentenvs/taxonomy.py` | `--class-preset` 或传 `spec=` |
| 3 | 智能体分工（谁专注哪类） | `list[dict{id,name,focus}]` | `taxonomy.resolve_specialization` | `EnvConfig(agents=[...])` |
| 4 | 奖励函数 | `EnvConfig` 的 reward_* 字段 | `agentenvs/cyber_defense_env.py` | 传 `EnvConfig` / `cfg_overrides` |
| 5 | 观测构造 | `env._get_observations()` | 同上 | 继承 `CyberDefenseEnv` 覆写 |
| 6 | 决策算法 | `select_actions` / `select_actions_raw` / `store` / `update` 四件套 | `algorithms/maddpg.py` | 换成任何实现这四件的对象 |
| 7 | 网络结构 | `Actor` / `Critic` | `algorithms/networks.py` | 传入 `MADDPGConfig(hidden_dims=...)` 或改类 |
| 8 | 数据来源 | `TrafficDataset` | `agentenvs/data_loader.py` | `build_dataset` / `load_bundle` / `make_synthetic` |
| 9 | 评估指标 | `evaluate_agents` 的 `act_fn` | `utils/metrics.py` | 传自己的策略函数 |

---

## 2. ★ 防守方检测器接口

### 2.1 契约（唯一必须实现的东西）

```python
from agentenvs.detectors import DetectorBase, DetectorOutput, register_detector

@register_detector("your_key")          # 可选：注册后 CLI 也能用 --detector your_key
class YourDetector(DetectorBase):
    embed_dim = 0                        # 类属性：你的 embedding 维度（0 = 没有）

    def fit(self, X_normal, X_test=None, y_test=None, **kw) -> "YourDetector":
        """在【良性流量】上训练/标定。允许 no-op（纯阈值法）。必须 return self。"""

    def score_batch(self, X) -> DetectorOutput:
        """对 (n, d) 打分，必须返回 DetectorOutput。"""
```

**必须实现**：`fit()`、`score_batch()`
**建议实现**：`reset()`（有状态检测器）、`save()/load()`（避免每次重训）

### 2.2 `DetectorOutput`：环境唯一消费的对象

```python
DetectorOutput(
    score:      np.ndarray,          # (n,) float32，连续异常分，【越高越异常】
    flag:       np.ndarray,          # (n,) {0,1}，二值告警。阈值由你标定
    trust:      np.ndarray,          # (n,) [0,1]，可信度
    embedding:  Optional[np.ndarray] = None,   # (n,k)，可选
    extra:      Dict[str, Any] = {},           # 随便放，环境不用（调试用）
)
```

构造时自动做规整：`score` 压成 float32、`flag = (x > 0.5)`、`trust` 裁剪到 [0,1]、
`embedding` 若为 1 维会自动 reshape 成 `(n,1)`。

**约定俗成的语义**（不是强制，但三种内置检测器都遵守，便于横向对比）：

| 字段 | 约定 |
|---|---|
| `score` | **单调方向必须校正**：越高越异常。如果你手里是"越高越正常"（如 sklearn 的 `score_samples`），取负。 |
| `flag` | 由 `score >= threshold` 得到；`threshold` 建议存成 `self.threshold` 便于落盘与复用。 |
| `trust` | **告警时 = 判为异常的置信度；不告警时 = 确信正常的程度**。即 `trust` 恒为"我对这个 flag 有多确信"，这样环境里它才是一个有信息量的输入。 |
| `embedding` | 放"判异常的理由"：AE 用逐维重构残差、Kalman 用标准化新息、IF 用树路径长度。让 critic 看到偏离**方向**而不只是幅度。 |

### 2.3 信号向量布局（环境实际读到的东西）

环境调用 `output.as_signal(embed_dim)` 打包成 **`(n, 3 + embed_dim)`**：

```
索引        含义                     来源
[0]         score                     DetectorOutput.score
[1]         flag                      DetectorOutput.flag
[2]         trust                     DetectorOutput.trust
[3 .. 3+k)  embedding（截断或零填充） DetectorOutput.embedding
```

* 常量：`SIGNAL_SCORE=0`、`SIGNAL_FLAG=1`、`SIGNAL_TRUST=2`、`SIGNAL_TAIL=3`
  （从 `agentenvs.detectors.base` 导入，别写魔数）
* 这个信号向量会**拼在每个智能体观测的尾部**，所以：
  * `signal_dim = 3 + embed_dim`（基类已实现，一般不用改）；
  * **`embed_dim` 变了，`obs_dim` 就变**，策略网络输入维度随之变化 →
    已训练的策略权重不再兼容，必须用同一 `embed_dim` 才能公平对比或热替换。

### 2.4 维度约束（最容易出错的地方）

```
obs_dim = feat_dim + (3 + embed_dim) + (num_classes if include_focus_onehot else 0)
```

| 规则 | 说明 |
|---|---|
| **要热替换检测器 → `embed_dim` 必须相同** | `env.set_detector()` 会校验，不一致抛 `ValueError` 并说明怎么改 |
| **环境会自动应用统一维度** | 若 `EnvConfig(detector_embed_dim=4)` 而传入的检测器 `embed_dim=0`（如 `make_detector("if")` 的默认值），环境会把 4 应用到该检测器上——**不会静默零填充**。所以 CLI 的 `--embed-dim 4` 对三种机制都生效 |
| **已训练的检测器受保护** | 传进来时检测器已有 `model`/状态（如 `det.load(...)` 后），则不允许被改维度，会报错要求用 `make_detector(..., embed_dim=4)` 重建 |
| **要复用同一份策略权重 → `obs_dim`、`num_classes` 必须相同** | 即 `feat_dim`、`embed_dim`、类别数、是否加 focus one-hot 都不能变 |
| **对比实验的推荐做法** | 统一 `detector_embed_dim=4`（CLI `--embed-dim 4`）。三种机制的原始 embedding 维度天生不同（IF=0 / Kalman=4 / AE=特征数），统一后观测维度一致，才能"只换检测器、其它全不动"地对比 |
| 更大的 embedding 允许 | 新检测器 `embed_dim` 大于环境要求时只截断，合法（多出来的维丢弃） |
| `feat_dim` 必须一致 | 检测器的输入维度 = 环境的特征维度，来自 `TrafficDataset.feature_names` 的长度。换特征子集 → 所有检测器都要重新 `fit` |

### 2.5 注册与构造

```python
from agentenvs.detectors import make_detector, list_detectors, register_detector, DETECTOR_REGISTRY

make_detector("kalman", mode="frozen", embed_dim=4)   # 工厂：唯一推荐入口
list_detectors()          # ['ae','autoencoder','if','isolation_forest','kalman','kf','null', ...]
register_detector("x")    # 装饰器，把自定义类注册进 DETECTOR_REGISTRY
```

注册表当前内容（别名指向同一个类）：

| key | 类 |
|---|---|
| `if` / `isolation_forest` | `IsolationForestDetector` |
| `kalman` / `kf` | `KalmanFilterDetector` |
| `autoencoder` / `ae` | `AutoencoderDetector` |
| `null` | `NullDetector` |
| `zscore_demo` / `stateful_demo` | `agentenvs/detectors/templates.py` 里的示例 |

### 2.6 四条接入路径

```python
# ① 字符串 key + 超参
env = make_env(ds, detector="kalman", num_agents=5)

# ② 已 fit 好的实例（跳过自动训练）
env = CyberDefenseEnv(ds.X_train, ds.y_train, detector=my_det,
                      X_normal_for_detector=ds.X_normal_train,
                      auto_fit_detector=False)

# ③ 注册后走统一入口（CLI / 配置文件都能引用）
@register_detector("mydet")
class MyDetector(DetectorBase): ...
env = make_env(ds, detector="mydet")

# ④ 鸭子类型：只要对象有 score_batch 或 detect 方法就会被 coerce_detector 放行
env = make_env(ds, detector=SomeThirdPartyModelWrapper(...))
```

`coerce_detector()` 的判定顺序：`None → null 检测器`；`str → make_detector`；
`DetectorBase 实例 → 原样使用`；有 `score_batch`/`detect` 属性 → 原样使用；否则 `TypeError`。

### 2.7 落盘约定

```python
det.save(path)                 # 建议把 threshold / 归一化统计量 / 内部状态一起存
det.load(path)                 # 返回 self
```

* `fit()` 之后 `threshold` 一般不该再变，否则同一策略在不同阈值下评估结果不可比。
* 标定好的检测器复用一条命令：
  ```python
  det = make_detector("kalman", embed_dim=4).fit(ds.X_normal_train, ds.X_test, (ds.y_test != 0).astype(int))
  det.save("outputs/kalman_det.npz")
  # 之后：make_detector("kalman", embed_dim=4).load("outputs/kalman_det.npz")
  ```
* 有状态检测器（Kalman）`load()` 后状态回到训练基线，`reset()` 不再清空基线，
  只把协方差/步数复位。

### 2.8 阈值标定工具（可直接复用）

```python
DetectorBase._calibrate_threshold(
    scores_normal,                     # 良性流量上的分数
    alpha=0.05,                        # 无标签时：取 1-alpha 分位 → 误报率≈alpha
    scores_test=None, y_test=None,      # 有标签时：扫 F1 最大阈值
) -> float
```

* 给了 `(scores_test, y_test)` 且标签非单类 → **F1 扫描**（乐观，用于复现论文指标）；
* 否则 → **良性分位数**（保守，用于上线/无标签场景）。
* 两种口径的结果差异很大，写论文时务必注明用了哪一种。

### 2.9 完整示例

可直接运行的模板在 **`agentenvs/detectors/templates.py`**：

| 类 | 演示什么 |
|---|---|
| `ZScoreDemoDetector`（`zscore_demo`） | 最小可用：标准化 + L2 范数打分 + 分位数阈值 + `save/load` |
| `StatefulDemoDetector`（`stateful_demo`） | **有状态/时序**检测器怎么写：跨调用维持内部状态、`reset()` 语义、标定时不污染 episode 起点 |
| `WrapExistingModel` | **包装已有模型**：不用改原代码，用 `score_fn/flag_fn/trust_fn/embed_fn` 适配 |

```powershell
& $PY -m agentenvs.detectors.templates     # 三个模板接进环境各跑一遍
# [OK] zscore_demo  signal_dim=7 AUC=0.995 ...
# [OK] stateful_demo signal_dim=7 AUC=0.974 ...
# [OK] WrapExistingModel signal_dim=7
# 模板自检通过 ✅
```

---

## 3. 环境接口

### 3.1 构造

```python
from agentenvs import CyberDefenseEnv, EnvConfig, make_env

env = CyberDefenseEnv(
    X_data,                      # (n, d) float32，已归一化训练集
    y_labels,                    # (n,) int64，0 = BENIGN
    detector="if",               # str | DetectorBase | None
    spec=None,                   # ClassSpec；None → 按 y_labels.max() 推断
    config=None,                 # EnvConfig；与 **cfg_overrides 二选一或混用
    X_normal_for_detector=None,  # 检测器 fit 用的良性流量；None → 取 X_data[y==0]
    X_test=None, y_test=None,    # 仅用于检测器阈值标定（F1 扫描）
    auto_fit_detector=True,      # False → 不 fit（检测器已训练好时用）
)
```

`make_env(dataset, detector="if", num_agents=5, max_steps=100, **cfg_overrides)` 是便捷包装。

### 3.2 只读属性

| 属性 | 类型 | 含义 |
|---|---|---|
| `env.num_agents` | int | 智能体数量 |
| `env.num_classes` | int | 类别数（含 BENIGN） |
| `env.feat_dim` | int | 流量特征维度 |
| `env.detector_signal_dim` | int | `3 + detector_embed_dim` |
| `env.detector_embed_dim` | int | 实际使用的 embedding 维度 |
| `env.obs_dim` | int | **每个智能体**的观测维度 |
| `env.action_dim` | int | 连续响应动作维度（=1） |
| `env.num_responses` | int | 响应档位数（=3：监控/限流/阻断） |
| `env.spec` | `ClassSpec` | 类别空间，`spec.names` / `spec.name_of(i)` / `spec.index_of(name)` |
| `env.focus` | `np.ndarray (N,)` | 每个智能体的专注类别索引 |
| `env.agents` | `list[dict]` | `[{'id','name','focus'}, ...]` |
| `env.detector` | `DetectorBase` | 当前检测器 |
| `env.stats` | dict | 当前 episode 的累计统计 |

### 3.3 观测与动作

**观测** `(num_agents, obs_dim)`，每行三段拼接（`normalize_obs=True` 时整体裁剪到 `[0,1]`）：

```
[0 : feat_dim)                          归一化流量特征（检测器的输入）
[feat_dim : feat_dim+signal_dim)        检测器信号 [score, flag, trust, embedding...]
[... : obs_dim)                         focus one-hot（可选，include_focus_onehot）
```

**动作**（每个智能体两路，见 5.1 为什么拆开）：

| 通道 | 维度 | 取值 | 解码 |
|---|---|---|---|
| 分类判定 `class_idx` | `(N,)` int | `0..num_classes-1` | `argmax(class_logits)`，由 actor 的分类头给出 |
| 连续响应 `action` | `(N, action_dim)` float | `[-1,1]`（tanh） | `bin_response()` 等宽分 3 档：`0=监控 / 1=限流 / 2=阻断` |

```python
env.decode_actions(actions)      # -> (class_idx, response_idx)
env.bin_response(r)              # 连续值 -> 档位
env.response_center(idx)         # 档位 -> 该档中心连续值（确定性执行用）
```

`decode_actions` 兼容两种动作布局：
* `action_dim == 1`：只有响应通道，分类由外部传入（本平台默认）；
* `action_dim == num_classes + 1`：`[class_logits..., response]`，按 argmax 取类别
  （兼容原始骨架"分类+响应拼成一个连续向量"的写法）。

### 3.4 交互 API

```python
obs = env.reset()                                  # (N, obs_dim)；会调用 detector.reset()
actions, class_idx, raws = agent.select_actions_raw(obs)
next_obs, rewards, done, info = env.step_with_classes(actions, class_idx)
#  rewards: (N,) float32 每智能体奖励
#  done:    bool（到 max_steps）
#  info:    true_label / true_class / class_pred / class_pred_names / response /
#           detector_score / detector_flag / rewards_class / rewards_response /
#           mean_reward / step / agent_names / focus

env.step(actions, class_idx=None, response_idx=None)   # 底层接口，三者都要可省
env.episode_metrics()          # dict：mean_accuracy / mean_f1 / episode_rewards / per_agent / detector
env.record_step_rewards(rew)   # 训练循环里累加 episode 回报（可选）
env.describe()                 # 打印环境摘要（检测器、维度、分工）
```

`step_with_classes(actions, class_idx, response_idx=None, auto_reset=False)`
是训练与评估的**推荐入口**：分类与响应分开传入，`auto_reset=True` 时 episode 结束自动重置。

标准 Gym 三件套 `reset() / step() / render` 保留，`gym`/`gymnasium` 缺失时会退化成纯 Python 类，
`observation_space` / `action_space` 仍然可用（需安装了 gym 系列）。

### 3.5 `EnvConfig` 全字段

```python
EnvConfig(
    # 规模
    num_agents=5, max_steps=100,
    # 奖励（论文 R_i 的权重，都可调）
    reward_correct_class=1.0,          # 分类正确
    reward_wrong_class=-1.0,           # 分类错误
    reward_specialist_bonus=0.5,       # 命中本智能体专注类别的额外奖励
    reward_response_attack=(0.5, 0.75, 1.0),    # 攻击流量：监控/限流/阻断
    reward_response_benign=(0.5, -0.25, -1.0),  # 良性流量：越激进罚越重
    reward_detector_align=0.2,         # 判定与 detector.flag 一致/不一致，±该值
    reward_step_cost=0.0,              # 每步固定开销（可为负）
    # 观测
    include_focus_onehot=True,         # 是否把分工 one-hot 放进观测
    normalize_obs=True,                # 是否裁剪到 [0,1]
    detector_embed_dim=None,           # None → 用 detector.embed_dim；统一维度时显式给值
    obs_noise_std=0.0,                 # 观测噪声（模拟部分可观测）
    # 其它
    shuffle=True, seed=None,
    agents=None,                       # 自定义分工，见 3.7
)
```

奖励语义注意点：
* `reward_detector_align` 让**检测器信号进入策略梯度**（判定与告警一致 +0.2，否则 -0.2）。
  做"检测器无用"消融时把它设为 0，否则策略会去讨好检测器；
* 良性流量上"阻断"罚 -1.0，比分类错误的 -1.0 叠加后更重，用于压制误报；
* `reward_response_attack` / `reward_response_benign` 的长度必须等于 `num_responses`（3）。

### 3.6 检测器热替换

```python
env.set_detector(make_detector("autoencoder", embed_dim=4), refit=True)
```
* 校验 `signal_dim` 不变，否则抛 `ValueError`；
* 新检测器 `embed_dim` 小于环境要求时报错（不会静默零填充），大于则截断；
* `refit=True` 会用 `env.X_data[y==0]` 重新 fit 新检测器；
* 用途：同一套策略权重下扫三种检测器的表现（观测维度不变才能做）。

### 3.7 智能体分工

```python
EnvConfig(agents=[
    {"id": 0, "name": "agent_ddos",  "focus": "DDoS"},      # focus 可给类名
    {"id": 1, "name": "agent_scan",  "focus": 2},           # 也可给索引
    {"id": 2, "name": "agent_doS",   "focus": "DoS"},
    {"id": 3, "name": "agent_web",   "focus": "WebAttack"},
    {"id": 4, "name": "agent_bot",   "focus": "Bot"},
])
```
* 不给 `agents` → `taxonomy.default_agents()` 自动给前 5 类各分一个智能体，多出的轮流复用；
* `focus` 只影响两件事：观测里的 one-hot，以及 `reward_specialist_bonus`（命中专注类 +0.5）；
* 想要"通用智能体"（无分工），把 `include_focus_onehot=False` 且
  `reward_specialist_bonus=0.0`。

### 3.8 换观测 / 换奖励（继承覆写）

```python
class MyEnv(CyberDefenseEnv):
    def _get_observations(self):
        """差异化观测：每个智能体只看自己关心的特征子集。"""
        obs = super()._get_observations()
        return obs * self.my_mask          # (N, obs_dim)
```
可覆写点：`_get_observations()`（观测）、`_reward_for_agent(i, cls, resp, true_label, det_flag)`
（奖励）、`_refresh_signal()`（检测器调用与缓存）、`_base_observation()`（共享观测主干）。

---

## 4. 内置检测器参数手册

### 4.1 通用参数（三种都有）

| 参数 | 默认 | 说明 |
|---|---|---|
| `embed_dim` | IF/Null: 0，Kalman: 4，AE: 特征数 | embedding 维度；**对比实验请统一** |
| `alpha` | 0.05 | 无标签时阈值取良性分数的 `1-alpha` 分位，等价目标误报率 |
| `threshold` | None | 显式指定阈值；给了就不再标定 |

### 4.2 `IsolationForestDetector`（`if` / `isolation_forest`）

| 参数 | 默认 | 说明 |
|---|---|---|
| `n_estimators` | 200 | 树数量 |
| `max_samples` | 256 | 每棵树采样量，固定值避免大树（与 `chethuhn/IF.py` 一致） |
| `contamination` | `"auto"` | 只影响 sklearn 自带的 `predict`，本实现不用它 |
| `max_features` | 1.0 | 每棵树的特征采样比例 |
| `bootstrap` | False | |
| `random_state` | 42 | |
| `n_jobs` | **1** | 受限沙箱里 joblib 多进程会 `PermissionError`；本机可设 -1 提速 |
| `embed_dim` | 0 | >0 时用前 k 棵树的路径长度作 embedding |

打分：`-model.score_samples(X)`；`trust` 由良性分数标准化后的 sigmoid 得到。

### 4.3 `KalmanFilterDetector`（`kalman` / `kf`）

| 参数 | 默认 | 说明 |
|---|---|---|
| `mode` | `"decay"` | `frozen` / `decay` / `forget`，见下 |
| `process_noise` (Q) | None | None → `forget` 用 `1/window`，其余用 `1e-6` |
| `measurement_noise` (R) | None | None → 良性数据逐维方差（下限 1e-8） |
| `forgetting` (λ) | 1.0 | <1 时对旧状态指数遗忘（协方差预测乘 λ²） |
| `initial_covariance` | 1.0 | P 初值 |
| `window` | 200.0 | `forget` 模式的等效记忆长度（流量条数） |
| `track_rate` | 3e-3 | `decay` 模式基线每步跟随比例 |
| `anomaly_track_scale` | 0.05 | 告警时跟随速率的缩放（越小越不信任异常样本） |
| `embed_dim` | 4 | 用 `|新息|` 最大的前 k 维 |
| `warmup` | 30 | 前若干步只更新状态、不告警 |

`mode` 的选择直接决定"检出什么"，做对比实验时必须在论文里写明：

| mode | 行为 | 适用 / 风险 |
|---|---|---|
| `frozen` | 基线冻结在良性训练集上，无时序记忆 | **与 IF/AE 最公平**；i.i.d. 逐条流量上表现最好 |
| `decay`（默认） | 基线缓慢跟随，既检出突变也适应正常漂移 | 最稳；`track_rate` 太大会漏检持续性偏移（实测 0.03 明显掉点） |
| `forget` | 基线快速跟随（等效窗口 `window` 条） | 只检出**突发**偏离；持续性攻击会把基线拉过去，之后不再告警——纯时序跟踪模型的固有局限 |

打分：逐维归一化新息平方（NIS）的均值；embedding：标准化新息 `innov/√S`（带符号，含方向信息）。
`reset()` 把状态复位到训练得到的稳态基线（不清空基线）。

### 4.4 `AutoencoderDetector`（`autoencoder` / `ae`）

| 参数 | 默认 | 说明 |
|---|---|---|
| `hidden_dims` | (64, 32) | 编码器各层宽度，解码器镜像 |
| `latent_dim` | 8 | 瓶颈维度 |
| `epochs` | 30 | 训练轮数 |
| `batch_size` | 256 | |
| `lr` | 1e-3 | Adam |
| `weight_decay` | 0.0 | |
| `dropout` | 0.0 | |
| `embed_dim` | None | None → 逐维重构残差（维度 = 特征数，比较大，建议显式设 4） |
| `device` | None | None → 有 CUDA 用 CUDA |
| `random_state` | 42 | |

打分：逐样本平均重构误差 `mean_j (x_j - x̂_j)²`；只在良性流量上训练。
`embed_dim=特征数` 会让 `obs_dim` 明显变大，做三方对比时务必显式 `embed_dim=4`。

### 4.5 `NullDetector`（`null`）

永远返回 `score=0 / flag=0 / trust=0.5`（可配）。用途：消融 baseline（策略完全靠原始特征）、
以及验证"换检测器不影响环境"这条解耦是否成立。

---

## 5. 算法接口

### 5.1 为什么分类与响应要拆开

DDPG 只能对**连续动作**求梯度，而攻击类型判定是**离散**的。把分类硬塞进连续向量
（原骨架的做法）梯度没有意义。本平台的做法：

```
actor(o_i) ─┬─ class_logits (num_classes,)  → argmax = 分类判定
            │     学习方式：REINFORCE + critic 优势基线（离散，不反传）
            └─ response_raw (1,) ─tanh→ [-1,1] → 3 档响应
                  学习方式：标准 DDPG 可微路径 ∇_a Q(s, a_1..a_N)
```

两者共享同一个**集中式 critic**（输入所有智能体的观测与动作），即 CTDE。

### 5.2 决策算法需要实现的方法（换算法的契约）

任何策略对象，只要实现下面四件套，就能直接替换 `MADDPG`（环境与训练脚本都只依赖它们）：

```python
class AnyPolicy:
    def select_actions(self, obs, noise_scale=None, deterministic=False):
        """评估用：返回 (actions (N, action_dim) float32, class_idx (N,) int64)"""

    def select_actions_raw(self, obs, noise_scale=None):
        """训练用：返回 (actions, class_idx, raw_action)。
        raw_action 是【加噪前的线性层输出】，写进回放池让 critic 的输入与
        当前策略输出同尺度（避免 target Q 被探索噪声污染）。"""

    def store(self, obs, action, reward, next_obs, done, class_idx=None) -> None:
        """写入共享回放池；action 传 select_actions_raw 返回的 raw_action"""

    def update(self, batch_size=None) -> dict:
        """集中式更新，返回 {'critic': [...], 'actor': [...]} 供记录"""
```

可选但训练脚本会用到：`ready()`（是否够样本更新）、`reset_noise()`、`total_updates`、
`total_steps`、`buffer`（需支持 `len()`）、`noise_scale`、`save(path)` / `load(path)`、
`describe()`。

### 5.3 `MADDPGConfig` 全字段（默认对齐论文 Table 2）

```python
MADDPGConfig(
    num_agents=5, obs_dim=32, num_classes=6, action_dim=1,
    hidden_dims=(128, 128), critic_hidden_dims=(256, 256),
    actor_lr=1e-4, critic_lr=1e-3,          # 论文 Table 2
    gamma=0.99, tau=0.001,                  # 论文 Table 2
    buffer_size=500_000, batch_size=128,    # 论文 Table 2
    updates_per_step=1, warmup_steps=1_000, grad_clip=1.0,
    noise_type="gaussian",                  # 或 "ou"（Ornstein-Uhlenbeck）
    noise_scale=0.1, noise_decay=1.0, noise_min=0.02,
    gumbel_tau=1.0,
    class_lr_scale=1.0,                     # 分类头策略梯度项缩放
    entropy_coef=0.02,                      # 分类头熵正则，抗早期塌缩
    device="auto", seed=0,
    shared_critic=False,                    # True → 所有智能体共用一个 critic
)
```

调参重点（分类头早期容易整体偏向 BENIGN，属类别不均衡下的正常现象）：
`entropy_coef` ↑、`class_lr_scale` ↑、`warmup_steps` ↑（先让 critic 稳），
或环境侧 `reward_specialist_bonus` / `reward_detector_align` 调整塑形强度。

### 5.4 换网络结构

```python
MADDPGConfig(hidden_dims=(256,256,256), critic_hidden_dims=(512,512))   # 加宽加深
```
或直接改 `algorithms/networks.py` 的 `Actor` / `Critic`：
* `Actor.forward(obs) -> (class_logits, response_raw)`，必须保持这个返回结构；
* `Actor.response(raw) -> action`，默认 tanh；
* `Critic.forward(all_obs, all_actions) -> (B,1)`。

### 5.5 回放池

```python
ReplayBuffer(capacity, num_agents, obs_dim, act_dim, seed=0)
buf.push(obs, act, rew, next_obs, done, class_idx=None)
buf.sample(batch_size)  # -> (obs, act, rew, next_obs, done, class_idx)
len(buf); buf.state_dict(); buf.load_state_dict(blob)
```
形状：`obs (C,N,O)`、`act (C,N,A)`、`rew (C,N)`、`done (C,N)`、`class_idx (C,N) int8`。

### 5.6 评估接口

```python
from utils.metrics import evaluate_agents, stratified_subset, classification_metrics

res = evaluate_agents(
    env,
    act_fn=lambda obs: agent.select_actions(obs, deterministic=True),
    X=ds.X_test, y=ds.y_test,
    max_samples=5000,     # 按类别分层抽样，避免取到的全是 BENIGN
    stratify=True, seed=42,
)
print(res.summary())
```
`res` 字段：`agents`（每个智能体的 accuracy / macro_f1 / kappa / detection_rate /
false_alarm_rate / pred_distribution / confusion_matrix）、`vote`（多智能体多数投票 + 平票偏向攻击侧）、
`detector`（检测器本身的 P/R/F1/误报率）、`n_samples`、`rewards`、`step_info`
（`y_true` / `vote_pred` / `per_agent_pred` / `detector_flags` / `detector_scores`，可画图）。

---

## 6. 数据接口

### 6.1 `TrafficDataset`

```python
X_normal_train   # 良性流量：检测器 fit + 归一化统计（只用它）
X_train, y_train # 带标签训练集（MADDPG 交互），y=0 为 BENIGN
X_test,  y_test  # 带标签测试集
X_val, y_val     # 可选
feature_names: list[str]
class_names:   list[str]           # [BENIGN, ...]
spec:          ClassSpec
meta:          dict                 # 来源、原始标签映射、稀有类合并记录
```

### 6.2 三种来源

```python
from agentenvs import build_dataset, DataConfig, FileSpec, default_file_specs, \
                      load_bundle, save_bundle, make_synthetic

ds = load_bundle("data/cicids_sample.npz")                     # ① 落盘数据包（推荐）
ds = build_dataset(DataConfig(files=default_file_specs(20000))) # ② 直接解析 CSV
ds = make_synthetic(n_normal=3000, n_per_class=600, feat_dim=26)  # ③ 合成数据（冒烟测试）
save_bundle(ds, "data/x.npz")
```

### 6.3 `DataConfig` / `FileSpec`

```python
DataConfig(
    data_dir=DEFAULT_DATA_DIR,
    files=None,                    # None → default_file_specs()
    normal_source=None,            # None → Monday 的 60000 行良性流量
    class_preset="default",        # default(5类+Other) / full / binary
    normal_ratio_in_train=0.30,    # 训练集良性占比上限（下采样）
    test_size=0.2, val_size=0.0,
    min_attack_rows=200,           # 少于该数的攻击类并入 Other（不会退化成 BENIGN）
    max_features=None,             # 按方差筛特征（换特征子集就靠它）
    seed=42,
)
FileSpec(filename, nrows=20000, seed=42, encoding="latin-1")
```

### 6.4 换特征子集

```python
cfg = DataConfig(max_features=30)                     # 方差 Top-30
ds = build_dataset(cfg)
```
注意：换特征后 `feat_dim` 变化 → 检测器必须重新 `fit`，已训练策略权重不再兼容。

---

## 7. Recipe：常见替换任务速查

| 我想…… | 改哪里 | 命令 / 代码 |
|---|---|---|
| 换检测机制 | 只改 `detector` 参数 | `train.py --detector kalman --embed-dim 4` |
| 新增自研检测算法 | 新增 `DetectorBase` 子类 | 抄 `agentenvs/detectors/templates.py`，`@register_detector("x")` |
| 包装已有的检测模型 | `WrapExistingModel` | `WrapExistingModel(model, score_fn=..., flag_fn=...)` |
| 三方对比且策略权重可复用 | 统一 `embed_dim` | `--embed-dim 4`（三种机制都传） |
| 用预训练检测器、不重训 | `auto_fit_detector=False` | 见 2.6 ②，或 `det.load(path)` 后传入（此时 `embed_dim` 已在训练时固定，不可再改） |
| 无标签场景（只控制误报率） | 不传 `X_test/y_test` | `det.fit(X_normal)`，阈值走分位数 |
| 改变攻击类别数 | `--class-preset` 或传 `ClassSpec` | `--class-preset full` / `ClassSpec.from_preset("full")` |
| 改智能体数量/分工 | `--num-agents` / `EnvConfig(agents=[...])` | 见 3.7 |
| 调奖励塑形 | `EnvConfig` 的 reward_* | 见 3.5；消融检测器影响时把 `reward_detector_align=0` |
| 换决策算法 | 实现 5.2 四件套 | 训练脚本里把 `MADDPG(...)` 换成你的策略类 |
| 换网络结构 | `MADDPGConfig(hidden_dims=...)` | 见 5.4 |
| 换特征子集 | `DataConfig(max_features=...)` | 换后检测器要重新 fit |
| 用真实数据 | `prepare_data.py` → `--data bundle` | `python prepare_data.py --out data/x.npz` |
| 只要环境不要训练 | 直接用 `CyberDefenseEnv` | 见 3.1 / 3.4 |

---

## 8. 适配性检查清单

**新增/替换检测器时逐条对照：**

- [ ] 继承了 `DetectorBase`，实现了 `fit()` 与 `score_batch()`，`fit()` 返回了 `self`
- [ ] 只用**良性流量** fit（不要把攻击样本喂给 `fit`，否则阈值与基线全错）
- [ ] `score` 方向是**越高越异常**（`score_samples` 这类要取负）
- [ ] `score` / `flag` / `trust` 形状都是 `(n,)`，`embedding` 是 `(n,k)`，且 `np.isfinite` 全通过
- [ ] `flag` 由你标定的 `threshold` 产生，`threshold` 存成属性（便于落盘）
- [ ] `trust` 语义 = "对当前 flag 的置信度"（告警时为置信度，不告警时为确信正常的程度）
- [ ] 声明了 `embed_dim`（并对齐对比实验用的统一值）
- [ ] 有状态检测器实现了 `reset()`，且**标定过程不会污染 episode 起点状态**
- [ ] 实现了 `save()/load()`（强烈建议，避免每次重训）
- [ ] 跑一遍 `verify_pipeline.py` 或 `python -m agentenvs.detectors.templates` 确认能接进环境

**做"只换检测器"的对比实验时：**

- [ ] `--embed-dim` 三种机制一致
- [ ] `feat_dim`、`--class-preset`、`--num-agents`、`include_focus_onehot` 一致
- [ ] 阈值口径一致（都用 F1 扫描，或都用良性分位数）
- [ ] `--seed` 一致
- [ ] Kalman 的 `mode` 在论文里明确写出并说明理由

---

## 9. 错误码与排查表

| 现象 / 报错 | 原因 | 解决 |
|---|---|---|
| `KeyError: 未知检测器 'xxx'，已注册：[...]` | 名字不在注册表 | 用 `list_detectors()` 里的名字，或 `@register_detector("xxx")` 注册 |
| `ValueError: 新检测器信号维度 3 != 原维度 7` | 热替换时 `embed_dim` 不一致 | 统一 `embed_dim`（如都传 `embed_dim=4`） |
| `ValueError: 环境固定 detector_embed_dim=4，但新检测器 ... 只能提供 0 维` | `set_detector` 传了 `embed_dim` 太小的检测器 | `env.set_detector(make_detector("if", embed_dim=4))` |
| `ValueError: 检测器 ... 已训练（embed_dim=0），但 EnvConfig(detector_embed_dim=4)...` | 想给已训练/已加载的检测器改维度 | 用 `make_detector(..., embed_dim=4)` 重建后重新 `fit`/`load` |
| `RuntimeError: XxxDetector 尚未 fit()` | 直接 `score_batch` 没先 `fit`，或 `auto_fit_detector=False` 但检测器确实没训练过 | 调 `env.fit_detector(X_normal)` 或 `det.fit(...)` |
| `ValueError: 没有良性样本可用于 fit 检测器` | `X_data[y==0]` 为空且没给 `X_normal_for_detector` | 传 `X_normal_for_detector` |
| `ValueError: embedding 行数与 score 不一致` | `DetectorOutput` 里 embedding 长度 ≠ 批大小 | 检查打分函数是否对 batch 逐行返回 |
| `TypeError: 无法识别的检测器对象` | 对象既不是 `DetectorBase` 也没有 `score_batch`/`detect` | 用 `WrapExistingModel` 包一层 |
| `PermissionError: [WinError 5] 拒绝访问`（建 IF 时） | 受限沙箱里 joblib 多进程被拒 | 用默认 `n_jobs=1`；本机可 `n_jobs=-1` |
| `RuntimeError: File ... cannot be opened`（save/load） | 往系统临时目录写 | 输出写到工作区内（`outputs/`、`data/`） |
| `Usecols do not match columns` | CICIDS CSV 表头带空格 | 已修（`_read_csv_sampled` 用原始列名）；自己改 CSV 时注意别 strip 表头 |
| `AssertionError: 训练没有发生 / total_updates == 0` | `warmup_steps`/`batch_size` 大于一个 episode 的步数 | 调小 `--warmup-steps --batch-size` 或调大 `--max-steps` |
| 训练 loss 正常但 F1 恒为 0、判定分布全是 BENIGN | 类别不均衡 + 分类头未收敛 | 见 5.3 调参重点；确认评估集是分层抽样的（`stratify=True`） |
| 评估 acc 很高但 F1=0、检测率=0 | 评估子集全是 BENIGN | 用 `stratified_subset` / `evaluate_agents(stratify=True)` |

---

## 附：与旧接口的兼容

* `detector.detect(X)` → `{'anomaly_score', 'anomaly_flag', 'trust'}`（单条），
  沿用 `chethuhn/IF.py` 风格的旧脚本可直接用；
* 环境 `step()` 同时支持 `(obs, rew, done, info)` 四元组；
* `decode_actions` 支持原始骨架的 `action_dim = num_classes + 1` 连续动作布局。
