# -*- coding: utf-8 -*-
"""
物理层状态异常检测模块
=====================================================================
包含两个层次的实现：

1. WLS + EKF（基础版）
   - WLS  : 加权最小二乘状态估计（电力系统经典静态状态估计）
   - EKF  : 扩展卡尔曼滤波（非线性动态状态估计）
   - ADI  : 基于新息（innovation）卡方检验的异常检测指数
            ADI = nu^T S^{-1} nu ~ chi2(m)

2. IVB-NCA-NLKF（进阶版）
   Improved Variational Bayesian Noise-Covariance Adaptive
   Nonlinear Kalman Filter
   - 过程噪声协方差 Q_k、测量噪声协方差 R_k 不再假设为固定已知量，
     而是赋予逆威沙特 (Inverse-Wishart, IW) 先验：
        Q_k ~ IW(t_k, T_k),  R_k ~ IW(u_k, U_k)
   - 用变分推断求近似后验 q(x_k) q(Q_k) q(R_k)，通过定点迭代
     联合更新状态与噪声协方差。
   - 残差 xi_k = || x_k - x_hat_k ||_2 超过阈值 tau 触发警报。

注意：numpy/scipy 为硬依赖且独立导入，不依赖 torch，
      因此本模块在未安装 PyTorch 的环境下仍可独立运行。
"""

import numpy as np
from scipy import stats
from scipy.linalg import solve, LinAlgError

EPS = 1e-12


# =====================================================================
# 通用工具：数值雅可比（中心差分）
# =====================================================================
def numerical_jacobian(func, x, eps=1e-6):
    """
    数值计算向量值函数 func 在 x 处的雅可比矩阵（中心差分）：

        J[i, j] = ( func_i(x + eps*e_j) - func_i(x - eps*e_j) ) / (2 eps)

    Parameters
    ----------
    func : callable,  x (n,) -> y (m,)
    x    : (n,) 当前状态
    Returns
    -------
    J : (m, n) 雅可比矩阵
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    n = x.shape[0]
    f0 = np.asarray(func(x), dtype=float).reshape(-1)
    m = f0.shape[0]
    J = np.empty((m, n))
    for j in range(n):
        xp, xm = x.copy(), x.copy()
        xp[j] += eps
        xm[j] -= eps
        J[:, j] = (np.asarray(func(xp)).reshape(-1)
                   - np.asarray(func(xm)).reshape(-1)) / (2.0 * eps)
    return J


def _sym_inv(A):
    """对（半）正定矩阵求逆，优先 Cholesky 求解，失败退回通用逆。"""
    A = np.asarray(A, dtype=float)
    I = np.eye(A.shape[0])
    try:
        return solve(A + EPS * I, I, assume_a="pos")
    except LinAlgError:
        return solve(A + 1e-8 * np.trace(A) / A.shape[0] * I + EPS * I, I)


def _as_col(v):
    return np.asarray(v, dtype=float).reshape(-1, 1)


# =====================================================================
# 1.1 WLS 加权最小二乘静态状态估计
# =====================================================================
class WLSStateEstimator:
    r"""
    电力系统加权最小二乘状态估计：

        J(x) = [z - h(x)]^T R^{-1} [z - h(x)]

    其中 h(x) 为非线性量测方程（如极坐标下支路潮流量测）。
    采用高斯-牛顿迭代求解一阶最优性条件 dJ/dx = 0：

        H(x) = dh/dx                         # 量测雅可比
        G(x) = H^T R^{-1} H                 # 增益矩阵 (Gain Matrix)
        G(x) [x^{(l+1)} - x^{(l)}]
             = H^T R^{-1} [z - h(x^{(l)})]

    目标函数值 J(x) 本身服从 chi2(m-n)，可作为坏数据/攻击检测依据。
    """

    def __init__(self, h_fun, R, x0, H_fun=None, max_iter=30, tol=1e-8):
        """
        Parameters
        ----------
        h_fun : callable, x (n,) -> z (m,) 非线性量测函数
        R     : (m,m) 量测噪声协方差
        x0    : (n,) 状态初值（电压相角/幅值平启动值）
        H_fun : callable, 可选的解析雅可比；不给定则用数值雅可比
        """
        self.h = h_fun
        self.H_fun = H_fun
        self.R = np.asarray(R, dtype=float)
        self.R_inv = _sym_inv(self.R)
        self.x = np.asarray(x0, dtype=float).reshape(-1)
        self.max_iter = max_iter
        self.tol = tol

    def _jacobian(self, x):
        if self.H_fun is not None:
            return np.atleast_2d(self.H_fun(x))
        return numerical_jacobian(self.h, x)

    def estimate(self, z):
        """
        执行一次 WLS 估计。

        Returns
        -------
        x_hat : (n,) 状态估计值
        info  : dict, 含 cost(目标函数值)、residual、n_iter、converged
        """
        z = np.asarray(z, dtype=float).reshape(-1)
        x = self.x.copy()
        converged = False
        n_iter = 0
        for l in range(self.max_iter):
            n_iter = l + 1
            H = self._jacobian(x)
            r = z - np.asarray(self.h(x)).reshape(-1)          # 量测残差
            # 正规方程: G dx = H^T R^{-1} r
            G = H.T @ self.R_inv @ H
            rhs = H.T @ self.R_inv @ r
            dx = solve(G + EPS * np.eye(G.shape[0]), rhs)
            x = x + dx
            if np.linalg.norm(dx, ord=np.inf) < self.tol:
                converged = True
                break

        r = z - np.asarray(self.h(x)).reshape(-1)
        cost = float(r @ self.R_inv @ r)       # J(x) ~ chi2(m-n)
        self.x = x                              # 热启动，供下一时刻使用
        return x, {"cost": cost, "residual": r,
                   "n_iter": n_iter, "converged": converged}


# =====================================================================
# 1.2 EKF + 卡方检验 (ADI)
# =====================================================================
class EKFChiSquareDetector:
    r"""
    扩展卡尔曼滤波 + 新息卡方异常检测。

    非线性状态空间模型：
        x_k = f(x_{k-1}) + w_k,   w_k ~ N(0, Q)
        z_k = h(x_k)     + v_k,   v_k ~ N(0, R)

    --- 预测 (Time Update) ---
        F = df/dx |_{x_hat_{k-1}}
        x_hat_k^- = f(x_hat_{k-1})
        P_k^-     = F P_{k-1} F^T + Q

    --- 校正 (Measurement Update) ---
        H = dh/dx |_{x_hat_k^-}
        nu_k = z_k - h(x_hat_k^-)                    # 新息
        S_k  = H P_k^- H^T + R                       # 新息协方差
        K_k  = P_k^- H^T S_k^{-1}                    # 卡尔曼增益
        x_hat_k = x_hat_k^- + K_k nu_k
        P_k     = (I - K_k H) P_k^-

    --- 异常检测指数 ADI (Anomaly Detection Index) ---
        ADI_k = nu_k^T S_k^{-1} nu_k
    系统正常时 ADI_k ~ chi2(m)；若
        ADI_k > chi2_{1-alpha}(m)
    则拒绝“量测正常”假设，判定物理层异常（如 FDIA 虚假数据注入）。
    """

    def __init__(self, f_fun, h_fun, Q, R, x0, P0=None,
                 F_fun=None, H_fun=None, alpha=0.05):
        self.f = f_fun
        self.h = h_fun
        self.F_fun = F_fun
        self.H_fun = H_fun

        self.Q = np.asarray(Q, dtype=float)
        self.R = np.asarray(R, dtype=float)

        self.x = np.asarray(x0, dtype=float).reshape(-1)
        n = self.x.shape[0]
        self.P = np.eye(n) if P0 is None else np.asarray(P0, dtype=float)

        m = self.R.shape[0]
        # 卡方门限：显著性水平 alpha（误报率），自由度为量测维数 m
        self.chi_threshold = float(stats.chi2.ppf(1.0 - alpha, df=m))
        self.m = m

    def _jacobian(self, fun, fun_jac, x):
        if fun_jac is not None:
            return np.atleast_2d(fun_jac(x))
        return numerical_jacobian(fun, x)

    def predict(self):
        """EKF 预测步，返回 (x_pred, P_pred, F)。"""
        F = self._jacobian(self.f, self.F_fun, self.x)
        x_pred = np.asarray(self.f(self.x)).reshape(-1)
        P_pred = F @ self.P @ F.T + self.Q
        return x_pred, P_pred, F

    def update(self, z, x_pred, P_pred):
        """
        EKF 校正步。

        Returns
        -------
        info : dict, 含 ADI、threshold、alarm、innovation、S、K
        """
        z = np.asarray(z, dtype=float).reshape(-1)
        H = self._jacobian(self.h, self.H_fun, x_pred)

        nu = z - np.asarray(self.h(x_pred)).reshape(-1)  # 新息
        S = H @ P_pred @ H.T + self.R                    # 新息协方差
        S_inv = _sym_inv(S)
        K = P_pred @ H.T @ S_inv

        self.x = x_pred + K @ nu
        self.P = (np.eye(self.x.shape[0]) - K @ H) @ P_pred

        ADI = float(nu @ S_inv @ nu)
        return {"ADI": ADI,
                "threshold": self.chi_threshold,
                "alarm": bool(ADI > self.chi_threshold),
                "innovation": nu, "S": S, "K": K}

    def step(self, z):
        """完整执行一个量测周期：预测 -> 校正 -> 卡方检验。"""
        x_pred, P_pred, _ = self.predict()
        info = self.update(z, x_pred, P_pred)
        info["x_hat"] = self.x
        info["P"] = self.P
        return info


# =====================================================================
# 1.3 WLS 参考 + 标准化残差卡方坏数据检测（无记忆、不发散）
# =====================================================================
class WLSResidualDetector:
    r"""
    以 WLS 提供的鲁棒参考状态 x_ref 为基准，对单量测断面做标准化
    残差（坏数据）卡方检测：

        nu = z - h(x_ref)
        ADI = nu^T R^{-1} nu ~ chi2(m)
        alarm = ADI > chi2_{1-alpha}(m)

    架构对应论文"WLS 作参考 / 检测器比对"：参考状态 x_ref 由冗余
    量测的 WLS（带坏数据剔除）给出，攻击量测不参与参考估计，因此
    攻击时 z 偏离 h(x_ref)，残差骤增。

    相比时序 EKF：无跨步状态记忆，在强非线性 + 恒定偏置攻击下不会
    因协方差初值不当而一步发散，检测结果对每个断面独立、可解释。
    """

    def __init__(self, h_fun, R, alpha=0.05):
        self.h = h_fun
        self.R = np.asarray(R, dtype=float)
        self.R_inv = _sym_inv(self.R)
        m = self.R.shape[0]
        self.m = m
        self.chi_threshold = float(stats.chi2.ppf(1.0 - alpha, df=m))

    def step(self, z, x_ref):
        """
        Parameters
        ----------
        z     : (m,) 当前量测（可能被攻击）
        x_ref : (n,) WLS 参考状态（不被攻击污染）
        Returns 与 EKF.step 同构的 dict（含 alarm/ADI/threshold）。
        """
        z = np.asarray(z, dtype=float).reshape(-1)
        nu = z - np.asarray(self.h(x_ref)).reshape(-1)
        ADI = float(nu @ self.R_inv @ nu)
        return {"ADI": ADI, "threshold": self.chi_threshold,
                "alarm": bool(ADI > self.chi_threshold),
                "innovation": nu}


# =====================================================================
# 2. IVB-NCA-NLKF 变分贝叶斯自适应非线性卡尔曼滤波
# =====================================================================
class IVB_NCA_NLKF:
    r"""
    改进型变分贝叶斯噪声协方差自适应非线性卡尔曼滤波器。

    -----------------------------------------------------------------
    0. 概率模型
    -----------------------------------------------------------------
    状态空间（与 EKF 相同，但 Q_k、R_k 时变且未知）：
        x_k = f(x_{k-1}) + w_k,   w_k ~ N(0, Q_k)
        z_k = h(x_k)     + v_k,   v_k ~ N(0, R_k)

    对未知协方差矩阵施加逆威沙特 (Inverse-Wishart) 共轭先验：
        Q_k ~ IW(Q_k; t_k, T_k)       # t: 自由度, T: 尺度矩阵
        R_k ~ IW(R_k; u_k, U_k)

    逆威沙特分布关键性质（d 为矩阵维数，要求自由度 nu > d+1）：
        E[Sigma]   = Psi / (nu - d - 1)                 # 后验均值
        E[Sigma^{-1}] = nu * Psi^{-1}                   # 逆的期望

    目标：求后验 p(x_k, Q_k, R_k | z_{1:k})。
    变分法用可分离的自由分布近似：
        p(x_k, Q_k, R_k | z_{1:k})
            ~= q(x_k) q(Q_k) q(R_k)
    其中
        q(x_k) = N(x_k; x_hat_k, P_k)
        q(Q_k) = IW(Q_k; t_{k|k}, T_{k|k})
        q(R_k) = IW(R_k; u_{k|k}, U_{k|k})
    通过最小化 KL 散度得到定点迭代（fixed-point）更新方程。

    -----------------------------------------------------------------
    1. 时间更新（噪声协方差的“演化”：引入遗忘因子 rho）
    -----------------------------------------------------------------
    令 n = dim(x), m = dim(z)，rho_Q, rho_R ∈ (0,1] 为遗忘因子
    （Sarkka & Nummenmaa 建议 rho = 1 - exp(-4) ~= 0.982，
      rho 越小，协方差估计对新数据响应越快，但越不平稳）：

        t_k^- = rho_Q (t_{k-1} - n - 1) + n + 1
        T_k^- = rho_Q T_{k-1}
        u_k^- = rho_R (u_{k-1} - m - 1) + m + 1
        U_k^- = rho_R U_{k-1}

    状态预测：
        x_hat_k^- = f(x_hat_{k-1}),  F = df/dx|_{x_hat_{k-1}}

    -----------------------------------------------------------------
    2. 测量更新（变分定点迭代，l = 1..N_iter）
    -----------------------------------------------------------------
    利用 IW 共轭性，第 l 轮迭代的超参数更新为：

    [过程噪声]
        dx_k^{(l-1)} = x_hat_k^{(l-1)} - x_hat_k^-
        t_{k|k}^{(l)} = t_k^- + 1
        T_{k|k}^{(l)} = T_k^-
                       + dx dx^T
                       + F P_{k-1} F^T
        E[Q_k^{-1}]^{(l)} = t_{k|k}^{(l)} (T_{k|k}^{(l)})^{-1}
        Q_hat_k^{(l)} = T_{k|k}^{(l)} / (t_{k|k}^{(l)} - n - 1)

    [测量噪声]
        nu_k^{(l-1)} = z_k - h(x_hat_k^{(l-1)})
        u_{k|k}^{(l)} = u_k^- + 1
        U_{k|k}^{(l)} = U_k^-
                       + nu nu^T
                       + H_k^{(l-1)} P_k^{(l-1)} (H_k^{(l-1)})^T
        E[R_k^{-1}]^{(l)} = u_{k|k}^{(l)} (U_{k|k}^{(l)})^{-1}
        R_hat_k^{(l)} = U_{k|k}^{(l)} / (u_{k|k}^{(l)} - m - 1)

    [非线性卡尔曼更新（用逆协方差期望代替固定的 Q^{-1}, R^{-1}）]
        H = dh/dx |_{x_hat_k^{(l-1)}}
        P_k^{-,(l)} = F P_{k-1} F^T + (E[Q_k^{-1}])^{-1}
        S_k^{(l)}   = H P_k^{-,(l)} H^T + (E[R_k^{-1}])^{-1}
        K_k^{(l)}   = P_k^{-,(l)} H^T (S_k^{(l)})^{-1}
        x_hat_k^{(l)} = x_hat_k^- + K_k^{(l)} [z_k - h(x_hat_k^-)]
        P_k^{(l)}     = P_k^{-,(l)} - K_k^{(l)} S_k^{(l)} (K_k^{(l)})^T

    迭代收敛（状态修正 < eps）或达到 N_iter 后停止。

    -----------------------------------------------------------------
    3. 物理层告警
    -----------------------------------------------------------------
        xi_k = || x_k - x_hat_k ||_2
    真实状态 x_k 不可观测；工程上以参考状态（WLS 估计 x_k^WLS，
    或无迹/预测参考值）代入；若未提供参考状态，则以先验预测
    x_hat_k^- 作为可计算替代（即状态后验修正量）。
        alarm = (xi_k > tau)
    阈值 tau 必须仅由“干净/正常运行”数据标定（如 99.5 分位数或
    mu + K sigma），避免用攻击样本选阈导致检测率虚高。
    """

    def __init__(self, f_fun, h_fun, Q0, R0, x0, P0=None,
                 F_fun=None, H_fun=None,
                 tau_q=5.0, tau_r=5.0,
                 rho_q=None, rho_r=None,
                 n_iter=10, xi_threshold=None,
                 k_sigma=3.0, conv_tol=1e-6):
        """
        Parameters
        ----------
        Q0, R0    : (n,n)/(m,m) 初始名义噪声协方差（同时确定先验中心）
        tau_q/r   : IW 先验强度（自由度余量）。越大表示对 Q0/R0 越确信。
                    初始化为:
                        t0 = n + tau_q + 1, T0 = tau_q * Q0
                        u0 = m + tau_r + 1, U0 = tau_r * R0
        rho_q/r   : 遗忘因子，None 时取 1 - exp(-4) ≈ 0.9817
        n_iter    : 变分定点迭代最大次数
        xi_threshold : 残差告警阈值 tau；None 表示不告警（待标定）
        k_sigma   : 标定时 mu + k_sigma * sigma 的系数
        """
        self.f, self.h = f_fun, h_fun
        self.F_fun, self.H_fun = F_fun, H_fun

        Q0 = np.asarray(Q0, dtype=float)
        R0 = np.asarray(R0, dtype=float)
        self.n, self.m = Q0.shape[0], R0.shape[0]

        self.x = np.asarray(x0, dtype=float).reshape(-1)
        self.P = np.eye(self.n) if P0 is None else np.asarray(P0, float)

        # ---- 逆威沙特先验超参数初始化 ----
        if tau_q <= 0 or tau_r <= 0:
            raise ValueError("tau_q/tau_r 必须 > 0 以保证尺度矩阵正定")
        self.t = self.n + tau_q + 1
        self.T = tau_q * Q0
        self.u = self.m + tau_r + 1
        self.U = tau_r * R0

        # 遗忘因子
        self.rho_q = 1.0 - np.exp(-4.0) if rho_q is None else rho_q
        self.rho_r = 1.0 - np.exp(-4.0) if rho_r is None else rho_r

        self.n_iter = n_iter
        self.tau = xi_threshold
        self.k_sigma = k_sigma
        self.conv_tol = conv_tol

        # 最近一次估计的协方差后验均值（便于外部读取）
        self.Q_hat = Q0.copy()
        self.R_hat = R0.copy()

    # ---------- 雅可比 ----------
    def _jacobian(self, fun, fun_jac, x):
        if fun_jac is not None:
            return np.atleast_2d(fun_jac(x))
        return numerical_jacobian(fun, x)

    # ---------- 单个滤波周期 ----------
    def step(self, z, x_ref=None):
        """
        执行一次 IVB-NCA-NLKF 滤波。

        Parameters
        ----------
        z     : (m,) 当前量测
        x_ref : (n,) 可选参考状态 x_k（如 WLS 估计），用于计算残差；
                      None 时用先验预测 x_hat_k^- 作替代。

        Returns
        -------
        info : dict
            x_hat, P, Q_hat, R_hat, xi, threshold, alarm, n_iter, converged
        """
        z = np.asarray(z, dtype=float).reshape(-1)

        # ============ 1) 时间更新 ============
        F = self._jacobian(self.f, self.F_fun, self.x)
        x_pred = np.asarray(self.f(self.x)).reshape(-1)

        # IW 超参数的“衰减-重参数化”传播
        t_pred = self.rho_q * (self.t - self.n - 1.0) + self.n + 1.0
        T_pred = self.rho_q * self.T
        u_pred = self.rho_r * (self.u - self.m - 1.0) + self.m + 1.0
        U_pred = self.rho_r * self.U

        # 迭代初值：名义 EKF 形式的先验
        x_cur = x_pred.copy()
        P_cur = F @ self.P @ F.T + self.Q_hat

        converged = False
        used_iters = 0
        for l in range(self.n_iter):
            used_iters = l + 1

            # ---- 2a) 非线性量测线性化 ----
            H = self._jacobian(self.h, self.H_fun, x_cur)

            # ---- 2b) q(Q_k): 过程噪声 IW 超参数 ----
            dx = _as_col(x_cur - x_pred)
            t_post = t_pred + 1.0
            T_post = T_pred + dx @ dx.T + F @ self.P @ F.T
            E_Q_inv = t_post * _sym_inv(T_post)        # E[Q_k^{-1}]
            Q_hat = T_post / (t_post - self.n - 1.0)   # E[Q_k]

            # ---- 2c) q(R_k): 测量噪声 IW 超参数 ----
            nu_i = _as_col(z - np.asarray(self.h(x_cur)).reshape(-1))
            u_post = u_pred + 1.0
            U_post = U_pred + nu_i @ nu_i.T + H @ P_cur @ H.T
            E_R_inv = u_post * _sym_inv(U_post)        # E[R_k^{-1}]
            R_hat = U_post / (u_post - self.m - 1.0)   # E[R_k]

            # ---- 2d) 变分 EKF 更新 ----
            P_minus = F @ self.P @ F.T + _sym_inv(E_Q_inv)
            S = H @ P_minus @ H.T + _sym_inv(E_R_inv)
            S_inv = _sym_inv(S)
            K = P_minus @ H.T @ S_inv

            # 新息以先验预测点 x_hat_k^- 计算（标准 EKF 口径）
            innov = _as_col(z - np.asarray(self.h(x_pred)).reshape(-1))
            x_new = x_pred + (K @ innov).reshape(-1)
            P_new = P_minus - K @ S @ K.T
            # 对称化，抑制数值误差
            P_new = 0.5 * (P_new + P_new.T)

            # ---- 2e) 定点收敛判定 ----
            if np.linalg.norm(x_new - x_cur) < self.conv_tol:
                x_cur, P_cur = x_new, P_new
                converged = True
                break
            x_cur, P_cur = x_new, P_new

        # ============ 3) 固化后验，供下一时刻传播 ============
        self.x = x_cur
        self.P = P_cur
        self.t, self.T = t_post, T_post
        self.u, self.U = u_post, U_post
        self.Q_hat, self.R_hat = Q_hat, R_hat

        # ============ 4) 残差与告警 ============
        reference = x_pred if x_ref is None else np.asarray(x_ref).reshape(-1)
        xi = float(np.linalg.norm(reference - self.x, ord=2))
        alarm = (self.tau is not None) and (xi > self.tau)

        return {"x_hat": self.x, "P": self.P,
                "Q_hat": self.Q_hat, "R_hat": self.R_hat,
                "xi": xi, "threshold": self.tau,
                "alarm": bool(alarm),
                "n_iter": used_iters, "converged": converged}

    # ---------- 阈值标定 ----------
    def calibrate_threshold(self, z_sequence, refs=None,
                            percentile=99.5):
        """
        仅使用正常（干净）运行数据标定残差阈值：

            tau = max( percentile(xi, q),
                        mean(xi) + k_sigma * std(xi) )

        Parameters
        ----------
        z_sequence : list/ndarray, 形状 (T, m) 的正常量测序列
        refs       : 可选参考状态序列 (T, n)
        Returns
        -------
        tau : 标定得到的阈值（同时写入 self.tau）
        """
        xis = []
        for k, z in enumerate(z_sequence):
            ref = None if refs is None else refs[k]
            info = self.step(z, x_ref=ref)
            xis.append(info["xi"])
        xis = np.asarray(xis)
        tau_pct = np.percentile(xis, percentile)
        tau_sig = xis.mean() + self.k_sigma * xis.std()
        self.tau = float(max(tau_pct, tau_sig))
        return self.tau


# =====================================================================
# 3. Sage-Husa 自适应 EKF（IVB-NCA-NLKF 的工程平替）
# =====================================================================
def _pd_clip(M, eig_floor):
    """
    将对称矩阵投影到“特征值 >= eig_floor”的半正定锥内：
        M = U diag(lambda) U^T -> M' = U diag(max(lambda,floor)) U^T
    防止 Sage-Husa 更新中出现负特征值导致滤波发散。
    """
    M = 0.5 * (M + M.T)
    w, U = np.linalg.eigh(M)
    w = np.maximum(w, eig_floor)
    return (U * w) @ U.T


class SageHusaAEKF:
    r"""
    Sage-Husa 自适应扩展卡尔曼滤波（Sage & Husa, 1969）。

    作为 IVB-NCA-NLKF 的平替：不做逆威沙特-变分推断，而是用指数
    加权的渐消记忆在线递推噪声的“一、二阶矩”，计算简单、实时性好。

    带未知噪声均值的模型：
        x_k = f(x_{k-1}) + q_k + w_k,   w_k ~ N(0, Q_k)
        z_k = h(x_k)     + r_k + v_k,   v_k ~ N(0, R_k)

    --- 预测 ---
        F = df/dx|_{x_{k-1}}
        x_k^- = f(x_{k-1}) + q_hat
        P_k^- = F P_{k-1} F^T + Q_{k-1}

    --- 校正 ---
        H = dh/dx|_{x_k^-}
        nu_k = z_k - h(x_k^-) - r_hat
        S_k  = H P_k^- H^T + R_{k-1}
        K_k  = P_k^- H^T S_k^{-1}
        x_hat_k = x_k^- + K_k nu_k
        P_k     = (I - K_k H) P_k^-

    --- Sage-Husa 自适应（指数加权，b 为遗忘因子 0<b<1）---
        d_k = (1 - b) / (1 - b^{k+1})       # 权重（早期权重大，稳态≈1-b）
        噪声均值：
        r_k = (1-d_k) r_{k-1}
              + d_k (z_k - h(x_hat_k))
        q_k = (1-d_k) q_{k-1}
              + d_k (x_hat_k - f(x_{k-1}))
        噪声协方差（无偏形式，减去线性化贡献项）：
        R_k = (1-d_k) R_{k-1}
              + d_k [nu_k nu_k^T - H P_k^- H^T]
        Q_k = (1-d_k) Q_{k-1}
              + d_k [K_k nu_k nu_k^T K_k^T
                     + P_k - F P_{k-1} F^T]

    --- 鲁棒保护 ---
    1) 每次更新做正定化（特征值下限 R_floor/Q_floor），防止
       R 估计出现负值而发散；
    2) 攻击门控：ADI 卡方超限时冻结自适应更新——否则攻击造成的
       巨大新息会被误学习为"测量噪声变大"（这正是协方差自适应类
       滤波器共同的失效模式）。
    """

    def __init__(self, f_fun, h_fun, Q0, R0, x0, P0=None,
                 F_fun=None, H_fun=None,
                 forgetting=0.97, alpha=0.05,
                 Q_floor=None, R_floor=None,
                 gate_adaptation=True, warmup=5,
                 xi_threshold=None):
        """
        Parameters
        ----------
        forgetting    : 遗忘因子 b（常用 0.95~0.99；越小响应越快）
        alpha         : 卡方门控显著性水平
        Q_floor/R_floor : 协方差特征值下限；None 时取初值最小特征值
                          的 1e-3 倍
        gate_adaptation : 卡方报警时是否冻结噪声统计更新
        warmup        : 前 warmup 步只跑标准 EKF 不做自适应。
                        首步权重 d_0=1，若 P0 未收敛，[nu nu^T
                        - H P^- H^T] 会使 R 一步崩溃，故必须预热。
        xi_threshold  : 残差告警阈值（需用干净数据标定）
        """
        self.f, self.h = f_fun, h_fun
        self.F_fun, self.H_fun = F_fun, H_fun

        self.Q = np.asarray(Q0, dtype=float)
        self.R = np.asarray(R0, dtype=float)
        self.x = np.asarray(x0, dtype=float).reshape(-1)
        n = self.x.shape[0]
        self.P = np.eye(n) if P0 is None else np.asarray(P0, dtype=float)

        self.q_mean = np.zeros(n)
        self.r_mean = np.zeros(self.R.shape[0])

        self.b = forgetting
        self.k = 0
        self.gate_adaptation = gate_adaptation
        self.warmup = warmup
        self.tau = xi_threshold

        m = self.R.shape[0]
        self.m = m
        self.chi_threshold = float(stats.chi2.ppf(1.0 - alpha, df=m))

        qf = np.linalg.eigvalsh(self.Q).min() * 1e-3
        rf = np.linalg.eigvalsh(self.R).min() * 1e-3
        self.Q_floor = qf if Q_floor is None else Q_floor
        self.R_floor = rf if R_floor is None else R_floor

    def _jacobian(self, fun, fun_jac, x):
        if fun_jac is not None:
            return np.atleast_2d(fun_jac(x))
        return numerical_jacobian(fun, x)

    def step(self, z, x_ref=None):
        """
        执行一个 Sage-Husa AEKF 周期。

        Returns
        -------
        info : dict，含 x_hat/P/Q_hat/R_hat、ADI/alarm、xi、adapted
        """
        z = np.asarray(z, dtype=float).reshape(-1)

        # ============ 1) 预测 ============
        F = self._jacobian(self.f, self.F_fun, self.x)
        f_x = np.asarray(self.f(self.x)).reshape(-1)
        x_pred = f_x + self.q_mean
        P_pred = F @ self.P @ F.T + self.Q

        # ============ 2) 校正 ============
        H = self._jacobian(self.h, self.H_fun, x_pred)
        nu = (z - np.asarray(self.h(x_pred)).reshape(-1)
              - self.r_mean)
        S = H @ P_pred @ H.T + self.R
        S_inv = _sym_inv(S)
        K = P_pred @ H.T @ S_inv

        x_upd = x_pred + K @ nu
        P_upd = (np.eye(self.x.shape[0]) - K @ H) @ P_pred
        P_upd = 0.5 * (P_upd + P_upd.T)

        ADI = float(nu @ S_inv @ nu)
        chi_alarm = ADI > self.chi_threshold

        # ============ 3) Sage-Husa 自适应 ============
        # d_k = (1-b)/(1-b^(k+1))
        d = (1.0 - self.b) / (1.0 - self.b ** (self.k + 1))
        adapted = False
        # 预热期不更新；攻击门控：报警时冻结，避免攻击被当作噪声学走
        if (self.k >= self.warmup
                and not (self.gate_adaptation and chi_alarm)):
            nu_col = _as_col(nu)
            h_x_upd = np.asarray(self.h(x_upd)).reshape(-1)

            self.r_mean = ((1.0 - d) * self.r_mean
                           + d * (z - h_x_upd))
            self.q_mean = ((1.0 - d) * self.q_mean
                           + d * (x_upd - f_x))

            R_raw = ((1.0 - d) * self.R
                     + d * (nu_col @ nu_col.T - H @ P_pred @ H.T))
            Q_raw = ((1.0 - d) * self.Q
                     + d * (K @ nu_col @ nu_col.T @ K.T
                             + P_upd - F @ self.P @ F.T))
            self.R = _pd_clip(R_raw, self.R_floor)
            self.Q = _pd_clip(Q_raw, self.Q_floor)
            adapted = True

        # ============ 4) 状态推进 / 残差告警 ============
        self.x, self.P = x_upd, P_upd
        self.k += 1

        reference = x_pred if x_ref is None \
            else np.asarray(x_ref).reshape(-1)
        xi = float(np.linalg.norm(reference - self.x, ord=2))
        alarm = chi_alarm or (
            (self.tau is not None) and (xi > self.tau))

        return {"x_hat": self.x, "P": self.P,
                "Q_hat": self.Q, "R_hat": self.R,
                "q_mean": self.q_mean, "r_mean": self.r_mean,
                "ADI": ADI, "chi_alarm": chi_alarm,
                "xi": xi, "threshold": self.tau,
                "alarm": bool(alarm), "adapted": adapted}

    def calibrate_threshold(self, z_sequence, refs=None,
                            percentile=99.5):
        """
        仅用干净数据标定残差阈值（同 IVB 的口径）：
            tau = max(percentile(xi,q), mean(xi)+3*std(xi))
        """
        xis = []
        for k, z in enumerate(z_sequence):
            ref = None if refs is None else refs[k]
            xis.append(self.step(z, x_ref=ref)["xi"])
        xis = np.asarray(xis)
        self.tau = float(max(
            np.percentile(xis, percentile),
            xis.mean() + 3.0 * xis.std()))
        return self.tau
