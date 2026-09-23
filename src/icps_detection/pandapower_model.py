# -*- coding: utf-8 -*-
"""
基于 pandapower 的电力系统物理模型
=====================================================================
为卡尔曼滤波提供“真实”的状态转移函数 f(x) 与非线性观测函数 h(x)：

1. PandapowerGridModel
   - 加载 pandapower 内置 IEEE 标准算例（case14 / case30 / case118）；
   - 用 pandapower(vendor PYPOWER) 构造节点导纳矩阵 Ybus；
   - 状态向量 x = [theta_2...theta_n, V_2...V_n]（平衡/slack 节点
     theta=0、V 固定，不纳入状态）；
   - 观测 h(x)：由 Ybus 精确计算各节点注入功率
         S_i = P_i + jQ_i = V_i [ sum_j Y_ij V_j ]^* * baseMVA
     并选取部分节点电压幅值量测，构成 SCADA 量测向量
         z = [P..., Q..., V...]
   - f(x) = x（准稳态跟踪状态估计的随机游走模型，F=I）；
   - 提供负荷随机游走 + 潮流计算生成“真值”轨迹，以及 FDIA
     （虚假数据注入）攻击向量。

2. DCMotorModel（论文提到的另一类物理模型：直流电机）
   状态 [theta, omega, i_a]，欧拉离散的解析 f/h/F，可在无
   pandapower 时作为替代基线。

依赖：pandapower 延迟导入；DCMotorModel 仅需 numpy。
"""

import copy

import numpy as np

# ---------------------------------------------------------------------
# NumPy 2.x 兼容垫片：pandapower 2.14 vendor 的 PYPOWER 仍使用
# numpy.Inf（NumPy 2 已移除该别名）。必须在 import pandapower 之前
# 完成补丁，故放在模块顶层。
# ---------------------------------------------------------------------
if not hasattr(np, "Inf"):
    np.Inf = np.inf
if not hasattr(np, "NaN"):
    np.NaN = np.nan
# NumPy 2 中被移除/改名的符号（pandapower 2.14 仍在使用）
if not hasattr(np, "in1d"):
    np.in1d = np.isin
if not hasattr(np, "trapz"):
    np.trapz = getattr(np, "trapezoid", None)
if not hasattr(np, "product"):
    np.product = np.prod
if not hasattr(np, "alltrue"):
    np.alltrue = np.all
if not hasattr(np, "sometrue"):
    np.sometrue = np.any
if not hasattr(np, "float_"):
    np.float_ = np.float64
if not hasattr(np, "complex_"):
    np.complex_ = np.complex128
if not hasattr(np, "unicode_"):
    np.unicode_ = np.str_

from .physical_layer import numerical_jacobian

EPS = 1e-12


# =====================================================================
# 1. pandapower IEEE 总线模型
# =====================================================================
class PandapowerGridModel:
    r"""
    IEEE 标准算例的物理模型封装。

    交流电网节点注入方程（观测函数的物理依据）：

        I_i = sum_j Y_{ij} V_j
        S_i = V_i I_i^* * baseMVA
            = V_i ( sum_j Y_{ij} V_j )^* * baseMVA
        P_i = Re(S_i),   Q_i = Im(S_i)

    状态（slack 节点作为相角参考，theta=0；V 固定）：
        x = [ theta_{nonslack} (rad) ; V_{nonslack} (p.u.) ]

    状态转移：
        x_k = x_{k-1} + w_k       （准稳态随机游走，F = I）
    真实的负荷缓变通过“扰动负荷 -> 潮流计算”生成真值轨迹，
    与过程噪声假设一致。
    """

    def __init__(self, case_name="case14"):
        try:
            import pandapower as pp
            import pandapower.networks as pn
            from pandapower.pd2ppc import _pd2ppc
        except ImportError as exc:
            raise ImportError(
                "需要 pandapower：pip install pandapower "
                "-i https://mirrors.aliyun.com/pypi/simple/"
            ) from exc

        self.pp = pp
        self._pd2ppc = _pd2ppc
        if not hasattr(pn, case_name):
            raise ValueError(f"pandapower.networks 中不存在 {case_name}")
        self.net0 = getattr(pn, case_name)()
        self.net = copy.deepcopy(self.net0)
        self.pp.runpp(self.net, numba=False)

        # ---------- 节点与状态索引 ----------
        self.n_bus = self.net.bus.shape[0]
        # net 内部 bus 索引就是 0..n-1
        self.slack_buses = list(self.net.ext_grid.bus)
        self.state_buses = [i for i in range(self.n_bus)
                            if i not in self.slack_buses]
        self.n_state_bus = len(self.state_buses)
        self.n_state = 2 * self.n_state_bus

        # ---------- 构造 Ybus（内部 ppci 行序 + net bus 映射） ----------
        self._build_ybus()

        # ---------- 量测量配置 ----------
        # 默认：全部非 slack 节点 P/Q 注入 + 有发电机节点的电压幅值
        gen_buses = sorted(set(self.net.gen.bus) | set(self.net.sgen.bus))
        self.v_meas_buses = [b for b in gen_buses
                             if b not in self.slack_buses] \
            or self.state_buses[:max(1, self.n_state_bus // 3)]
        # 量测段在向量中的顺序与长度
        self.p_idx = self.state_buses
        self.q_idx = self.state_buses
        self.v_idx = self.v_meas_buses
        self.n_meas = (len(self.p_idx) + len(self.q_idx)
                       + len(self.v_idx))

        # slack 节点参考电压（V 与相角）
        self.slack_vm = float(
            self.net.ext_grid.vm_pu.iloc[0])
        self.slack_va = 0.0

    # -----------------------------------------------------------------
    # Ybus 构建
    # -----------------------------------------------------------------
    def _build_ybus(self):
        """
        由 net 构造内部 ppci 并调用版本匹配的 makeYbus。

        关键映射（全部以 net bus 索引访问，避免依赖内部行序）：
            row_of_bus[b] = net["_pd2ppc_lookups"]["bus"][b]
        该查找表在 _ppc2ppci 中已更新为 net bus -> ppci 内部行。
        """
        from pandapower.pypower.makeYbus import makeYbus

        # 注意：net 已由 runpp 初始化 _options/_is_elements，_pd2ppc 可复用
        ppc, ppci = self._pd2ppc(self.net)
        self.ppc = ppc
        self.Ybus, _Yf, _Yt = makeYbus(
            ppci["baseMVA"], ppci["bus"], ppci["branch"])
        self.baseMVA = ppci["baseMVA"]

        # 稀疏 -> dense；查找表: 以 net bus id 为下标
        if hasattr(self.Ybus, "toarray"):
            self.Ybus = self.Ybus.toarray()
        self.row_of_bus = np.asarray(
            self.net["_pd2ppc_lookups"]["bus"])

    # -----------------------------------------------------------------
    # 状态 <-> 全网复电压
    # -----------------------------------------------------------------
    def _voltages_from_state(self, x):
        """
        将状态 x 展开为 ppci 内部行序的复电压向量 V。
        slack 节点: V = vm * exp(j*theta_ref)；其余: vm*exp(j*theta)。
        """
        x = np.asarray(x, dtype=float).reshape(-1)
        ns = self.n_state_bus
        thetas = x[:ns]
        vms = x[ns:]

        V = np.zeros(self.n_bus, dtype=complex)
        for b in self.slack_buses:
            r = self.row_of_bus[b]
            V[r] = self.slack_vm * np.exp(1j * self.slack_va)
        for k, b in enumerate(self.state_buses):
            r = self.row_of_bus[b]
            V[r] = vms[k] * np.exp(1j * thetas[k])
        return V

    def state_from_results(self):
        """潮流收敛后，从 net.res_bus 提取状态向量。"""
        res = self.net.res_bus
        thetas = np.deg2rad(
            res.va_degree.loc[self.state_buses].to_numpy())
        vms = res.vm_pu.loc[self.state_buses].to_numpy()
        return np.concatenate([thetas, vms])

    # -----------------------------------------------------------------
    # f(x) / h(x)
    # -----------------------------------------------------------------
    def f(self, x):
        """准稳态状态转移：x_k = x_{k-1}（F=I，变化由过程噪声承载）。"""
        return np.asarray(x, dtype=float).reshape(-1)

    def F(self, x):
        return np.eye(self.n_state)

    def injections(self, x):
        """
        由状态计算全网节点注入功率（MW, MVAr）：

            S = V * conj(Ybus V) * baseMVA
        返回按 net bus 顺序排列的 P, Q 数组。
        """
        V = self._voltages_from_state(x)
        S = V * np.conj(self.Ybus @ V) * self.baseMVA
        P = np.zeros(self.n_bus)
        Q = np.zeros(self.n_bus)
        for b in range(self.n_bus):
            r = self.row_of_bus[b]
            P[b] = S[r].real
            Q[b] = S[r].imag
        return P, Q

    def h(self, x):
        """
        量测函数：z = [P 注入(选中节点); Q 注入(选中节点); V 幅值]。
        """
        P, Q = self.injections(x)
        z = np.concatenate([
            P[self.p_idx],
            Q[self.q_idx],
            np.asarray(x, dtype=float).reshape(-1)[
                self.n_state_bus:][
                [self.state_buses.index(b) for b in self.v_idx]],
        ])
        return z

    def H(self, x):
        """量测雅可比（中心差分数值解）。"""
        return numerical_jacobian(self.h, x)

    # -----------------------------------------------------------------
    # 真值轨迹生成：负荷随机游走 -> runpp
    # -----------------------------------------------------------------
    def reset(self):
        self.net = copy.deepcopy(self.net0)
        self.pp.runpp(self.net, numba=False)
        return self

    def step_load_walk(self, sigma=0.01, rng=None):
        """
        对全部负荷做对数随机游走：
            p_k = p_{k-1} * exp(sigma * z),  q 同理
        再运行潮流，返回真值状态。rng 可传入 np.random.Generator
        以保证轨迹可复现。
        """
        rng = rng or np.random.default_rng()
        z = rng.normal(0, sigma, size=len(self.net.load))
        self.net.load.p_mw *= np.exp(z)
        self.net.load.q_mvar *= np.exp(z)
        self.pp.runpp(self.net, numba=False)
        return self.state_from_results()

    # -----------------------------------------------------------------
    # FDIA 攻击向量
    # -----------------------------------------------------------------
    def fdia_vector(self, bus_targets, bias_p=0.0, bias_q=0.0,
                    bias_v=0.0):
        """
        构造量测域攻击向量 a：对指定节点的 P/Q 注入量测与 V 量测
        施加恒定偏置（最常见的盲 FDIA 形式 a = H c）。

        被攻击量测: z_a = h(x) + a
        """
        a = np.zeros(self.n_meas)
        n_p, n_q = len(self.p_idx), len(self.q_idx)
        for b in bus_targets:
            if b in self.p_idx:
                a[self.p_idx.index(b)] += bias_p
            if b in self.q_idx:
                a[n_p + self.q_idx.index(b)] += bias_q
            if b in self.v_idx:
                a[n_p + n_q + self.v_idx.index(b)] += bias_v
        return a


# =====================================================================
# 2. 直流电机模型（备选物理模型）
# =====================================================================
class DCMotorModel:
    r"""
    电枢控制直流电机的连续时间模型（欧拉离散，步长 dt）：

        J dot(omega) = K_t i_a - b omega - T_L        # 机械方程
        L dot(i_a)   = u - R i_a - K_e omega           # 电枢回路
        dot(theta)   = omega                           # 角位置

    状态 x = [theta, omega, i_a]，输入 u = 电枢电压，
    T_L 为负载转矩（建模为恒定/缓慢扰动）。

    离散状态转移 f(x)：
        theta' = theta + dt omega
        omega' = omega + dt (K_t i_a - b omega - T_L) / J
        i_a'   = i_a   + dt (u - R i_a - K_e omega) / L
    观测 h(x) = x（三传感器：编码器/测速机/电流采样），H = I。
    """

    def __init__(self, dt=0.01, R=1.0, L=0.5, K_t=0.05, K_e=0.05,
                 J=0.02, b=0.1, T_load=0.0, u=12.0):
        self.dt, self.R, self.L = dt, R, L
        self.K_t, self.K_e = K_t, K_e
        self.J, self.b = J, b
        self.T_load, self.u = T_load, u
        self.n_state, self.n_meas = 3, 3

    def f(self, x):
        theta, omega, i = x
        domega = (self.K_t * i - self.b * omega - self.T_load) / self.J
        di = (self.u - self.R * i - self.K_e * omega) / self.L
        return np.array([
            theta + self.dt * omega,
            omega + self.dt * domega,
            i + self.dt * di,
        ])

    def F(self, x):
        """f 对 x 的解析雅可比。"""
        dt = self.dt
        return np.array([
            [1.0, dt, 0.0],
            [0.0, 1.0 - dt * self.b / self.J, dt * self.K_t / self.J],
            [0.0, -dt * self.K_e / self.L, 1.0 - dt * self.R / self.L],
        ])

    def h(self, x):
        return np.asarray(x, dtype=float).reshape(-1)

    def H(self, x):
        return np.eye(3)

    @staticmethod
    def sensor_attack_vector(bias_theta=0.0, bias_omega=0.0,
                             bias_current=0.0):
        """量测域攻击：对传感器施加恒定偏置。"""
        return np.array([bias_theta, bias_omega, bias_current])
