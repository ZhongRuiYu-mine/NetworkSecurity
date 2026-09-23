# -*- coding: utf-8 -*-
"""
网络层流量异常检测模块
=====================================================================
1. NBurstFeatureExtractor
   将网络流量建模为 N 个独立 ON/OFF（交替更新）源叠加的 N-Burst
   过程，从包到达时间序列提取突发统计特征（ON/OFF 时长分布、
   空闲率、突发强度、Hurst 自相似参数等）。

2. CSEKM (Clustering-Stability based Evolutionary K-Means)
   基于“聚类稳定性”的进化 K-Means：以 K 个质心为种群个体，
   适应度同时考虑簇内紧度（SSE）与代际稳定性（匈牙利匹配），
   缓解经典 K-Means 对初值敏感、易陷局部最优的问题。

3. CPLMatcher
   缓存模式库 (Cached Pattern Library) + 生存值 (Survival Value)
   的余弦相似度在线模式匹配：命中则刷新 SV，未命中则判为未知
   （可疑）模式并入缓存；SV 持续衰减，过低模式被淘汰。

纯 numpy/scipy 实现，无需 torch。
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

EPS = 1e-12


# =====================================================================
# 1. N-Burst (ON/OFF) 流量建模与特征提取
# =====================================================================
def hurst_rs(series):
    r"""
    重标极差 (Rescaled Range, R/S) 分析估计 Hurst 指数 H。

    对长度为 n 的子序列：
        均值中心化:  y_t = x_t - x_bar
        累积离差:    Y_t = sum_{i=1}^t y_i
        极差:        R = max(Y) - min(Y)
        标准差:      S = std(x)
        R/S ~ (n/2)^H
    在多个尺度 n 上做 log(R/S) 对 log(n) 的线性回归，斜率即 H：
        H = 0.5  无相关（类白噪声/泊松）
        H > 0.5  长程正相关（自相似/突发，流量常见 0.5~1.0）
        H < 0.5  反相关
    """
    x = np.asarray(series, dtype=float)
    N = x.size
    ns, rs_list = [], []
    for k in range(2, int(np.floor(np.log2(N)))):
        n = 2 ** k
        nblocks = N // n
        if nblocks < 1:
            continue
        vals = []
        for b in range(nblocks):
            block = x[b * n:(b + 1) * n]
            y = block - block.mean()
            Y = np.cumsum(y)
            R = Y.max() - Y.min()
            S = block.std(ddof=1)
            if S > EPS:
                vals.append(R / S)
        if vals:
            ns.append(n)
            rs_list.append(np.mean(vals))
    if len(ns) < 2:
        return 0.5
    return float(np.polyfit(np.log(ns), np.log(rs_list), 1)[0])


class NBurstFeatureExtractor:
    r"""
    N-Burst / ON-OFF 交替更新模型特征提取器。

    模型：N 个相互独立的源在 ON / OFF 两状态间交替切换——
        ON  : 以高速率 r_on 突发发包
        OFF : 静默（或以低背景速率发包）
    N 个源叠加形成聚合流量，其突发与自相似特性由 ON/OFF 时长分布
    及源数量 N 决定。

    处理流程：
    1) 将时间轴按 bin_width 分箱，统计每箱包数 c_t（聚合到达计数）；
    2) 以 on_threshold 判定各箱 ON(1)/OFF(0)；
    3) 对 0/1 序列做游程分析，得到 ON/OFF 持续时间样本；
    4) 汇总成特征向量：
       [mean_on, var_on, mean_off, var_off, on_ratio, idle_ratio,
        burst_rate, arrival_rate, peak_to_mean, hurst, n_bursts]
    """

    FEATURE_NAMES = [
        "mean_on", "var_on", "mean_off", "var_off", "on_ratio",
        "idle_ratio", "burst_rate", "arrival_rate", "peak_to_mean",
        "hurst", "n_bursts",
    ]

    def __init__(self, bin_width=0.1, on_threshold=1.0):
        """
        Parameters
        ----------
        bin_width    : 分箱时长（秒）
        on_threshold : 箱内包数 >= 该阈值判为 ON
        """
        self.bin_width = bin_width
        self.on_threshold = on_threshold

    @staticmethod
    def _run_lengths(flags, target):
        """提取 0/1 序列中取值为 target 的连续游程长度（箱数）。"""
        lengths, cnt = [], 0
        for f in flags:
            if f == target:
                cnt += 1
            elif cnt > 0:
                lengths.append(cnt)
                cnt = 0
        if cnt > 0:
            lengths.append(cnt)
        return np.asarray(lengths, dtype=float)

    def extract(self, timestamps, t_end=None):
        """
        Parameters
        ----------
        timestamps : 一维数组，各数据包的到达时刻（秒，升序）
        t_end      : 观测结束时刻；None 时取最后一个包时刻

        Returns
        -------
        features : dict（含特征向量 vector 与命名字段）
        """
        ts = np.asarray(timestamps, dtype=float)
        if ts.size == 0:
            raise ValueError("时间戳为空")
        t0, t1 = ts.min(), (ts.max() if t_end is None else t_end)
        n_bins = max(1, int(np.ceil((t1 - t0) / self.bin_width)))
        counts, _ = np.histogram(ts, bins=n_bins,
                                 range=(t0, t0 + n_bins * self.bin_width))

        flags = (counts >= self.on_threshold).astype(int)
        on_runs = self._run_lengths(flags, 1)
        off_runs = self._run_lengths(flags, 0)

        on_durs = on_runs * self.bin_width
        off_durs = off_runs * self.bin_width

        mean_on = on_durs.mean() if on_durs.size else 0.0
        var_on = on_durs.var() if on_durs.size else 0.0
        mean_off = off_durs.mean() if off_durs.size else 0.0
        var_off = off_durs.var() if off_durs.size else 0.0

        total_on = on_durs.sum()
        total_off = off_durs.sum()
        on_ratio = total_on / (total_on + total_off + EPS)
        idle_ratio = total_off / (total_on + total_off + EPS)

        duration = n_bins * self.bin_width
        # ON 期间的包速率（突发强度）
        burst_rate = ts.size / (total_on + EPS)
        arrival_rate = ts.size / (duration + EPS)
        peak_to_mean = counts.max() / (arrival_rate * self.bin_width + EPS)
        H = hurst_rs(counts)

        feat = {
            "mean_on": mean_on, "var_on": var_on,
            "mean_off": mean_off, "var_off": var_off,
            "on_ratio": on_ratio, "idle_ratio": idle_ratio,
            "burst_rate": burst_rate, "arrival_rate": arrival_rate,
            "peak_to_mean": peak_to_mean,
            "hurst": float(H),
            "n_bursts": float(on_durs.size),
        }
        feat["vector"] = np.asarray([feat[k] for k in self.FEATURE_NAMES])
        return feat


# =====================================================================
# 2. CSEKM —— 基于聚类稳定性的进化 K-Means
# =====================================================================
class EvolutionaryKMeans:
    r"""
    CSEKM: Clustering-Stability based Evolutionary K-Means。

    个体编码：一组 K 个质心 C = (c_1,...,c_K)，即 (K, d) 矩阵。
    适应度由两部分构成：

    (a) 聚类质量（紧度，越小越好 -> 转为 [0,1] 越大越好）
        J(C) = sum_{x} min_k ||x - c_k||^2
        Q_i = 1 - (J_i - J_min) / (J_max - J_min)

    (b) 代际稳定性（父代 P 与子代 O 的质心经匈牙利算法最优配对后
        的位移，位移越小越稳定）
        设代价 D_{ij} = ||c_i^P - c_j^O||，匈牙利算法求最小总代价 pi*:
        stab_i = exp( - sum_j D_{j,pi*(j)} / (K * d_scale) )
        d_scale 取全体样本对距离的均值作为尺度归一化。

        Fitness_i = alpha * Q_i + (1 - alpha) * stab_i

    进化算子：
    - K-Means 算子（主变异）：分配 -> 更新质心，确定性局部搜索；
    - 随机替换：以 mutation_prob 把一个质心替换为随机样本，
      提供跳出局部最优的多样性；
    - 精英保留 + 按 Fitness 锦标赛式选择；
    - 周期性注入随机重启个体，维持种群多样性。
    """

    def __init__(self, n_clusters, pop_size=10, n_generations=50,
                 alpha=0.5, mutation_prob=0.1, n_restart=2,
                 tol=1e-4, patience=10, random_state=None):
        """
        Parameters
        ----------
        n_clusters   : K
        pop_size     : 种群大小 P
        n_generations: 进化代数
        alpha        : 质量权重；(1-alpha) 为稳定性权重
        mutation_prob: 质心随机替换概率
        n_restart    : 每代注入的随机重启个体数
        """
        self.K = n_clusters
        self.P = pop_size
        self.G = n_generations
        self.alpha = alpha
        self.mutation_prob = mutation_prob
        self.n_restart = n_restart
        self.tol = tol
        self.patience = patience
        self.rng = np.random.default_rng(random_state)

        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.history_ = []

    # ---------- K-Means++ 初始化单个个体 ----------
    def _init_individual(self, X):
        n = X.shape[0]
        K = min(self.K, n)
        first = self.rng.integers(n)
        centers = [X[first]]
        for _ in range(1, K):
            d2 = ((X[:, None, :] - np.asarray(centers)[None]) ** 2).sum(-1)
            d2_min = d2.min(axis=1)
            probs = d2_min / (d2_min.sum() + EPS)
            centers.append(X[self.rng.choice(n, p=probs)])
        C = np.asarray(centers, dtype=float)
        if K < self.K:  # 样本数不足 K 时复制补全
            C = np.vstack([C] + [C[-1:]] * (self.K - K))
        return C

    @staticmethod
    def _assign(X, C):
        """最近质心分配（基于平方欧氏距离）。"""
        return cdist(X, C, metric="sqeuclidean").argmin(axis=1)

    @staticmethod
    def _inertia(X, labels, C):
        return float(((X - C[labels]) ** 2).sum())

    def _update_centroids(self, X, labels, C):
        """K-Means 质心更新；空簇用随机样本重播种。"""
        new_C = C.copy()
        for k in range(self.K):
            members = X[labels == k]
            if members.shape[0] > 0:
                new_C[k] = members.mean(axis=0)
            else:
                new_C[k] = X[self.rng.integers(X.shape[0])]
        return new_C

    def _mutate(self, C, X):
        """随机变异：替换一个质心为随机数据点。"""
        C = C.copy()
        if self.rng.random() < self.mutation_prob:
            C[self.rng.integers(self.K)] = X[self.rng.integers(X.shape[0])]
        return C

    @staticmethod
    def _stability(C_parent, C_off, d_scale):
        """
        代际稳定性：匈牙利算法配对父/子代质心。
        返回 [0,1]，1 表示质心完全未移动。
        """
        D = cdist(C_parent, C_off)
        row, col = linear_sum_assignment(D)
        total = D[row, col].sum() / C_parent.shape[0]
        return float(np.exp(-total / (d_scale + EPS)))

    # ---------- 主训练流程 ----------
    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        if n < 2:
            raise ValueError("样本太少，无法聚类")

        # 全局尺度：样本两两距离均值（用于稳定性归一化）
        sample = X if n <= 500 else X[self.rng.choice(n, 500, replace=False)]
        d_scale = cdist(sample, sample).mean() + EPS

        # 初始化种群
        population = [self._init_individual(X) for _ in range(self.P)]
        best_J, stale = np.inf, 0

        for gen in range(self.G):
            # --- 产生子代：K-Means 局部搜索 + 变异 ---
            offspring, J_list = [], []
            for C in population:
                labels = self._assign(X, C)
                C_new = self._update_centroids(X, labels, C)
                C_new = self._mutate(C_new, X)
                labels_new = self._assign(X, C_new)
                offspring.append(C_new)
                J_list.append(self._inertia(X, labels_new, C_new))

            # 注入随机重启个体（多样性，防止早熟收敛）
            for _ in range(self.n_restart):
                offspring.append(self._init_individual(X))
                labels = self._assign(X, offspring[-1])
                J_list.append(self._inertia(X, labels, offspring[-1]))

            J_arr = np.asarray(J_list)
            # --- 质量得分 [0,1] ---
            J_min, J_max = J_arr.min(), J_arr.max()
            quality = (1.0 - (J_arr - J_min) / (J_max - J_min + EPS)
                       if J_max > J_min else np.ones_like(J_arr))

            # --- 稳定性得分（子代对应当代父代；重启个体稳定度低）---
            stab = np.zeros(len(offspring))
            for i, C_off in enumerate(offspring):
                parent = population[i % self.P]
                stab[i] = self._stability(parent, C_off, d_scale)
            # 随机重启个体不给稳定性加成，鼓励但不纵容
            stab[self.P:] = 0.0

            fitness = self.alpha * quality + (1.0 - self.alpha) * stab

            # --- 选择：精英 + 适应度 Top-(P-1) ---
            order = np.argsort(-fitness)
            new_pop = [offspring[order[0]]]
            for idx in order[1:]:
                if len(new_pop) >= self.P:
                    break
                new_pop.append(offspring[idx])
            population = new_pop

            cur_best = J_arr[order[0]]
            self.history_.append({"generation": gen,
                                  "best_inertia": float(cur_best),
                                  "mean_stability": float(stab.mean())})

            # --- 早停：最优 inertia 长期无显著改善 ---
            if best_J - cur_best > self.tol * (abs(best_J) + EPS):
                best_J, stale = cur_best, 0
            else:
                stale += 1
            if stale >= self.patience:
                break

        # 对最优个体再跑标准 Lloyd 直到收敛（局部精修）
        C = population[0]
        for _ in range(100):
            labels = self._assign(X, C)
            C_new = self._update_centroids(X, labels, C)
            if np.abs(self._inertia(X, labels, C_new)
                      - self._inertia(X, labels, C)) < self.tol:
                C = C_new
                break
            C = C_new

        self.cluster_centers_ = C
        self.labels_ = self._assign(X, C)
        self.inertia_ = self._inertia(X, self.labels_, C)
        return self

    def predict(self, X):
        return self._assign(np.asarray(X, dtype=float),
                            self.cluster_centers_)


# =====================================================================
# 3. CPL + SV 余弦相似度在线模式匹配
# =====================================================================
class CPLMatcher:
    r"""
    带缓存模式库 (CPL) 与生存值 (SV) 的余弦相似度在线匹配器。

    对缓存模式 p_i 与新模式 x，余弦相似度：
        cos(x, p_i) = (x . p_i) / (||x|| ||p_i||)

    生存值更新（指数遗忘 + 命中强化）：
        SV_i(t) = decay * SV_i(t-1)              # 每个观测周期衰减
        若命中 p_{i*}: SV_{i*}(t) += reward
        淘汰: SV_i < evict_threshold 的模式从库中删除

    判定逻辑：
        max_i cos >= sim_threshold -> 命中（已知正常模式）
        否则                       -> 未命中（未知/可疑流量模式，
                                      网络层异常），并将其注册为新模式
    """

    def __init__(self, sim_threshold=0.9, initial_sv=1.0, reward=1.0,
                 decay=0.99, evict_threshold=0.2, max_size=1000):
        """
        Parameters
        ----------
        sim_threshold   : 余弦相似度命中门限
        initial_sv      : 新注册模式的初始生存值
        reward          : 命中奖励
        decay           : 每周期 SV 衰减系数 (0,1]
        evict_threshold : SV 低于该值的模式被淘汰
        max_size        : CPL 最大容量，超出时淘汰 SV 最低者
        """
        self.sim_threshold = sim_threshold
        self.initial_sv = initial_sv
        self.reward = reward
        self.decay = decay
        self.evict_threshold = evict_threshold
        self.max_size = max_size

        self.patterns = []          # list of (d,) ndarray
        self.survival = []          # list of float
        self.hit_counts = []

    # ---------- 批量初始化（如用 CSEKM 质心作为初始模式库） ----------
    def initialize(self, patterns):
        self.patterns = [np.asarray(p, dtype=float) for p in patterns]
        self.survival = [self.initial_sv] * len(self.patterns)
        self.hit_counts = [0] * len(self.patterns)
        return self

    @staticmethod
    def _cosine(x, P):
        """x 与库中所有模式的余弦相似度。"""
        x = np.asarray(x, dtype=float)
        P = np.asarray(P, dtype=float)
        x_norm = np.linalg.norm(x) + EPS
        p_norm = np.linalg.norm(P, axis=1) + EPS
        return (P @ x) / (p_norm * x_norm)

    def _evict(self, keep_idx=None):
        """淘汰 SV 过低者；容量超限时淘汰 SV 最低者。"""
        keep = set(range(len(self.patterns)))
        for i in range(len(self.patterns)):
            if i != keep_idx and self.survival[i] < self.evict_threshold:
                keep.discard(i)
        # 容量控制
        if len(keep) > self.max_size:
            sorted_idx = sorted(keep, key=lambda i: self.survival[i])
            for i in sorted_idx[:len(keep) - self.max_size]:
                keep.discard(i)
        keep = sorted(keep)
        self.patterns = [self.patterns[i] for i in keep]
        self.survival = [self.survival[i] for i in keep]
        self.hit_counts = [self.hit_counts[i] for i in keep]

    # ---------- 在线匹配 ----------
    def match(self, x):
        """
        处理一个到达的流量模式。

        Returns
        -------
        result : dict
            matched (True=已知正常 / False=未知可疑), index, similarity,
            anomaly (== not matched), n_patterns
        """
        # 先执行全库时间衰减
        self.survival = [sv * self.decay for sv in self.survival]

        if not self.patterns:
            # 空库：注册为首条基线模式（首批通常为可信基线流量）
            self.patterns.append(np.asarray(x, dtype=float))
            self.survival.append(self.initial_sv)
            self.hit_counts.append(0)
            return {"matched": True, "index": 0, "similarity": 1.0,
                    "anomaly": False, "n_patterns": 1}

        sims = self._cosine(x, self.patterns)
        idx = int(np.argmax(sims))
        best = float(sims[idx])

        if best >= self.sim_threshold:
            # 命中：强化生存值
            self.survival[idx] += self.reward
            self.hit_counts[idx] += 1
            self._evict(keep_idx=idx)
            return {"matched": True, "index": idx, "similarity": best,
                    "anomaly": False, "n_patterns": len(self.patterns)}

        # 未命中：注册为新模式（未知流量 -> 标记异常）
        self.patterns.append(np.asarray(x, dtype=float))
        self.survival.append(self.initial_sv)
        self.hit_counts.append(0)
        new_idx = len(self.patterns) - 1
        self._evict(keep_idx=new_idx)
        return {"matched": False, "index": new_idx, "similarity": best,
                "anomaly": True, "n_patterns": len(self.patterns)}

    def library_size(self):
        return len(self.patterns)
