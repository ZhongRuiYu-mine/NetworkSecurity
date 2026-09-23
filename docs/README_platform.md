# MADDPG 多智能体网络攻击分类 —— 环境与可插拔检测接口

> 📌 **本文件已并入合并仓库 `netsec`**（原 `safenetwork/NetworkSecurity-main/README.md`）。
> 里面的相对路径（`agentenvs/`、`algorithms/`）现在位于 **`src/`** 下，
> 脚本位于 **`scripts/`** 下；快速开始请以仓库根的 **`README.md`** 与
> **`PROTOCOL.md`** 为准。下面的"快速开始"已按新布局改写。

复现论文 *Adaptive Multiagent Reinforcement Learning Framework for Cyber Attack
Detection and Tracking*（`papers/AdaptiveMultiagent.pdf`）里的 **Phase 2 / MADDPG**
部分：多个智能体各自专注一类攻击（DDoS / PortScan / Bot / DoS / WebAttack），
通过 **CTDE（集中式训练 + 分散式执行）** 协同完成攻击类型分类与响应决策。

本阶段的目标是**搭好 agent–env 平台并预留防守方接口**，让多种检测机制
（Isolation Forest / Kalman Filter / Autoencoder / ae_paper）可以随便换进来跑通对比实验。
检测算法本身怎么写不是这里的事——环境只认一个固定契约。

---

## 1. 快速开始

```powershell
cd <仓库根>                      # 即 netsec/
$env:PYTHONPATH = "$PWD\src"     # 脚本自带引导块，这一步可省
$PY = python

# ① 环境自检（检测器 + 环境 + MADDPG 更新 + 热替换），一分钟内跑完
& $PY scripts\verify_pipeline.py

# ② 生成规范数据包（协议 v1，约 20 秒；见 PROTOCOL.md）
& $PY -m ae_repro.data_prep

# ③ ★ 四种检测机制同口径对比（一条命令产出主表）
& $PY scripts\eval_all_detectors.py

# ④ 各检测机制下的策略训练（对比实验；目前只有 if 跑过）
& $PY scripts\train.py --data bundle --detector if          --episodes 200 --out results\if --plots
& $PY scripts\train.py --data bundle --detector kalman      --episodes 200 --out results\kalman --plots
& $PY scripts\train.py --data bundle --detector autoencoder --episodes 200 --out results\autoencoder --plots
```

`verify_pipeline.py` 通过 = 平台可用。训练超参先不用纠结，默认值已对齐论文
Table 2（agents=5, lr=1e-4, γ=0.99, buffer=500k, batch=128, τ=0.001）。

> ⚠️ 对比实验的口径由 `PROTOCOL.md` 固定：主口径 = 良性分位数（`fit()` 不传 `y_test`），
> F1 扫描只能作乐观上界。合并前那张混口径的"三方对比表"已作废，
> 替换物是 `results/detector_comparison.md`（见 `docs/reports/README.md`）。

---

## 2. 目录结构

```
agentenvs/
├── cyber_defense_env.py     ★ Gym 风格多智能体环境（观测/动作/奖励/指标）
├── detectors/               ★ 防守方接口（可插拔检测机制）
│   ├── base.py                  DetectorBase 抽象契约 + 注册表 + 工厂 make_detector
│   ├── isolation_forest_detector.py   IF：在良性流量上 fit，score_samples 取负
│   ├── kalman_detector.py             Kalman：随机游走状态空间 + 新息(NIS)打分
│   ├── autoencoder_detector.py        AE：良性流量重构误差
│   ├── null_detector.py               空检测器（消融 baseline / 验证解耦）
│   └── templates.py                   ★ 自定义检测器模板（最小可用 / 有状态 / 包装已有模型）
├── data_loader.py           CICIDS2017 流式采样/归一化/切分 + 合成数据 + .npz 落盘
└── taxonomy.py              标签归一化 + 攻击类别表 + 智能体分工生成
algorithms/
├── networks.py              Actor（分类头 + 连续响应头）/ 集中式 Critic
├── replay_buffer.py         共享经验池
└── maddpg.py                MADDPG 训练器（CTDE、目标网络软更新、集中式更新）
utils/metrics.py             Accuracy / Precision / Recall / F1 / Kappa / 检测率 / 混淆矩阵
docs/INTERFACE.md            ★ 完整接口文档（契约、维度约束、参数手册、替换 Recipe、排查表）
prepare_data.py              CSV → .npz 数据包
train.py                     训练 / 评估入口
verify_pipeline.py           环境与接口自检
```

> 接口细节、各检测器全部参数、常见替换任务和报错排查见 **`docs/INTERFACE.md`**。
> 想加自己的检测算法：抄 `agentenvs/detectors/templates.py`，然后
> `python -m agentenvs.detectors.templates` 自检。

---

## 3. ★ 防守方接口（三种检测机制怎么接）

### 3.1 契约

任何检测器只要实现这三件事，就能直接塞进环境，**MADDPG 与环境代码一行都不用改**：

```python
class MyDetector(DetectorBase):
    embed_dim = 0                       # 可选 embedding 维度

    def fit(self, X_normal, X_test=None, y_test=None): ...   # 在良性流量上训练/标定
    def score_batch(self, X) -> DetectorOutput: ...          # 对 (n, d) 打分
    # reset(self)                       可选：有状态检测器在 episode 边界清状态
    # save/load(path)                   可选：落盘避免重复训练
```

`DetectorOutput` 是环境唯一消费的东西，字段固定：

| 字段 | 形状 | 含义 |
|---|---|---|
| `score` | (n,) | 连续异常分，**越高越异常**（方向要自己校正） |
| `flag` | (n,) | 0/1 二值告警 |
| `trust` | (n,) | [0,1] 可信度（Kalman 用 NIS 一致性，IF/AE 用分数显著度） |
| `embedding` | (n,k) / None | 可选：让 critic 看到"为什么判异常" |

环境把 signal 向量 `[score, flag, trust, embedding...]` 拼到每个智能体观测后面，
所以**信号维度必须统一**：`detector.signal_dim == 3 + embed_dim`。
三种机制的原始 embedding 维度不同（IF=0 / Kalman=4 / AE=特征数），
用 `detector_embed_dim=4` 统一即可：环境会把该维度应用到检测器上（不会静默零填充），
这样观测维度一致，策略网络可以复用/对比。

完整契约说明、维度约束、各检测器全部参数、替换 Recipe 与排查表见 **`docs/INTERFACE.md`**；
可直接运行的模板（最小可用 / 有状态 / 包装已有模型）见 `agentenvs/detectors/templates.py`。

### 3.2 三种用法

```python
from agentenvs import make_env, make_detector, register_detector

# ① 按名字构造（注册表里内置 if / kalman / autoencoder / null）
env = make_env(ds, detector="kalman", num_agents=5)

# ② 用自定义超参
det = make_detector("kalman", mode="frozen")     # 或 mode="decay" / "forget"
env = make_env(ds, detector=det)

# ③ 自己写一个（对接你已有的检测代码也可以，只要是这个契约）
@register_detector("mydet")
class MyDetector(DetectorBase):
    embed_dim = 4
    def fit(self, X_normal, **kw): self.model = fit_something(X_normal); return self
    def score_batch(self, X): ...  # -> DetectorOutput

# ④ 直接把已 fit 好的对象交给环境（auto_fit_detector=False 跳过训练）
env = CyberDefenseEnv(ds.X_train, ds.y_train, detector=my_fitted_det,
                      X_normal_for_detector=ds.X_normal_train, auto_fit_detector=False)

# ⑤ 运行期热替换（同一套策略权重下做 A/B，观测量纲不变）
env.set_detector(make_detector("autoencoder", embed_dim=4))
```

命令行只改一个参数即可切换：

```powershell
& $PY train.py --data bundle --detector if          --out outputs/if
& $PY train.py --data bundle --detector kalman      --out outputs/kalman
& $PY train.py --data bundle --detector autoencoder --out outputs/autoencoder
& $PY train.py --data bundle --detector null        --out outputs/baseline_no_detector
```

### 3.3 三种检测机制的实现要点

| 机制 | 打分 | 阈值标定 | embedding | 备注 |
|---|---|---|---|---|
| **IF** | `-score_samples` | 良性分位数 / 有标签时 F1 扫描 | 可选：树路径长度 | `n_jobs=1`，沙箱里 joblib 多进程会被拒 |
| **Kalman** | 逐维新息的均值（NIS） | 同上 | 标准化新息 `innov/√S` | 有状态、流式；`mode` 见下 |
| **AE** | 平均重构误差 | 同上 | 逐维重构残差 | PyTorch，仅在良性流量上训练 |

Kalman 的 `mode` 很关键，做对比实验时值得在论文里说明：

* `decay`（默认）：基线缓慢跟随，既能检出突变也能适应正常分布漂移；
* `frozen`：基线冻结在良性训练集上（无时序记忆），i.i.d. 表格流量上表现最好，**与 IF/AE 最公平**；
* `forget`：基线快速跟随（等效窗口 `window` 条流量），只检出**突发**偏离——
  持续性攻击会把基线拉过去，之后不再告警。这是纯时序跟踪模型的固有局限。

---

## 4. 环境接口

```python
from agentenvs import CyberDefenseEnv, EnvConfig, make_env

env = make_env(ds, detector="if", num_agents=5, max_steps=100)
print(env.describe())

obs = env.reset()                                   # (num_agents, obs_dim)
actions, class_idx, raws = maddpg.select_actions_raw(obs)
next_obs, rewards, done, info = env.step_with_classes(actions, class_idx)
metrics = env.episode_metrics()                     # acc / P / R / F1 / 检测率 / 误报率
```

**观测** `(num_agents, obs_dim)`，每行 = 
`[归一化流量特征 d] + [检测器信号 3+embed_dim] + [focus one-hot num_classes]`

**动作**（每个智能体）
* `class_logits` → argmax 即攻击类型判定（离散；用 critic 优势加权的策略梯度学习）
* `response` → 连续 1 维，tanh 后按 3 档解码：`0=监控 / 1=限流 / 2=阻断`

> 为什么这么拆：DDPG 只能对连续动作求梯度，分类是离散决策。
> 若把分类硬塞进连续动作向量（原骨架的做法）梯度无意义；
> 现在连续部分（响应）走标准 DDPG 可微路径，离散部分（分类）走
> REINFORCE + critic 优势基线，二者共享同一个集中式 critic。
> 需要兼容旧布局时，`decode_actions` 也支持 `action_dim = num_classes + 1` 的连续向量。

**奖励**（对齐论文 Reward function R_i，权重都在 `EnvConfig` 里可调）

| 项 | 值 |
|---|---|
| 分类正确 | `+1`（命中本智能体专精类别再 `+specialist_bonus`） |
| 分类错误 | `-1` |
| 攻击流量响应：监控/限流/阻断 | `+0.5 / +0.75 / +1.0` |
| 良性流量响应：监控/限流/阻断 | `+0.5 / -0.25 / -1.0`（越激进罚越重） |
| 与检测器告警一致 / 不一致 | `±0.2`（把检测器信号变成可学的塑形奖励） |

---

## 5. 数据

`chethuhn/network-intrusion-dataset/versions/1/*.csv`（CICIDS2017，78 特征）。
`data_loader.py` 会：

1. 按文件**流式分块采样**（不用把 200MB+ 的 CSV 整个读进内存）；
2. 归一化 `Label`（原始文件里的 `Web Attack ? Brute Force` 是编码损坏，统一成 `WebAttack`）；
3. 只用 **Monday 良性流量** 做 Min-Max 归一化（论文 Eq.1）和检测器训练；
4. 按类别分层切分 train/test，训练集里良性流量下采样到 `--normal-ratio`；
5. 样本数不足 `min_attack_rows` 的攻击族并入 `Other` 兜底类，不会退化成 BENIGN。

默认类别空间：`['BENIGN','DDoS','PortScan','DoS','WebAttack','Other']`，
前 5 类正好分给 5 个专注型智能体（`taxonomy.default_agents`）。
`--rows-per-file` 调大类别更全（Bot/WebAttack 样本很稀疏，20000 才比较稳）。

---

## 6. 训练产物

`outputs/<detector>/` 下：`best.pt` / `last.pt`（权重）、`history.json`（每轮指标）、
`eval_<detector>.json`（各智能体 + 投票 + 检测器的完整评估）、`curves.png`（`--plots`）。

评估既看**单个智能体**（accuracy / macro-F1 / 检测率 / 误报率 / 判定分布），
也看**多智能体投票**（集成判定，平票偏向攻击侧），还会单独报**检测器本身**的
检测率与误报率——这样三种机制的差异才能拆开看。

评估子集是**按类别分层抽样**的（`utils.metrics.stratified_subset`）：真实测试集里
BENIGN 占 ~80%，若直接取前 N 条会全是良性流量，准确率虚高而 F1 恒为 0。

---

## 7. 依赖

统一依赖清单见仓库根的 `requirements.txt`（版本已锁定；历史清单见
`docs/requirements_platform.txt`）。核心：`numpy pandas scikit-learn scipy torch
matplotlib joblib tqdm`；`gymnasium`/`gym` 只是可选外壳（没有也能跑）。

两个环境相关的注意点（非代码 bug）：

* `IsolationForestDetector` 默认 `n_jobs=1`：在受限沙箱里 joblib 的多进程后端会
  因命名管道被拒而报 `PermissionError`；在你自己的机器上可以传 `n_jobs=-1`。
* 所有产出（权重、图、数据包）都写在**仓库内**（`results/`、`data/`），
  不要用系统临时目录。

---

## 8. 已知边界（留给下一阶段）

* **验收标准**：`verify_pipeline.py` 全部通过 = 平台可用（环境 + 四种检测器 +
  MADDPG 集中式更新 + 存档 + 热替换），这已经跑通；训练超参、收敛效果、分类精度
  属于下一阶段的任务，当前默认值只保证"能跑起来、不报错、指标可测"。
* 环境每个 step 只处理一条流量，智能体之间是**共享观测**（`_get_observations` 里
  留了差异化观测的扩展位）；若要模拟不同网段/不同传感器的部分可观测性，
  在 `_get_observations` 里按 agent 拆分特征子集即可。
* PPO 响应阶段（论文 Phase 3）与 DDQN 检测阶段（Phase 1）尚未实现——
  检测阶段现在由可插拔检测器承担，响应阶段暂由 actor 的连续动作头承担。
* 分类头用 REINFORCE + critic 优势基线学习，早期易偏向 BENIGN（类别不均衡所致），
  调参时可关注 `MADDPGConfig.entropy_coef`、`class_lr_scale` 与环境的
  `reward_specialist_bonus` / `reward_detector_align`。
