import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple


# =========================
#  車両パラメータ
# =========================

@dataclass
class VehicleParams:
    m: float = 1500.0      # 質量 [kg]
    Iz: float = 2250.0     # ヨー慣性モーメント [kg m^2]
    lf: float = 1.2        # 前軸〜重心距離 [m]
    lr: float = 1.6        # 後軸〜重心距離 [m]
    Cf: float = 8.0e4      # 前輪コーナリングスティフネス [N/rad]
    Cr: float = 8.0e4      # 後輪コーナリングスティフネス [N/rad]

    tau_a: float = 0.1     # 加速度アクチュエータ時定数 [s]
    tau_delta: float = 0.1 # ステアアクチュエータ時定数 [s]

    max_steer_deg: float = 30.0        # 最大舵角 [deg]
    max_steer_rate_deg: float = 180.0  # 最大舵角指令レート [deg/s]
    max_accel: float = 3.0             # 最大加速 [m/s^2]
    min_accel: float = -6.0            # 最大減速 [m/s^2]（負）

    # ---- 安定化用（タイヤ力飽和）----
    mu: float = 0.9        # 摩擦係数（簡易飽和）
    g: float = 9.81        # 重力加速度 [m/s^2]

    @property
    def max_steer(self) -> float:
        return self.max_steer_deg * 3.141592653589793 / 180.0

    @property
    def max_steer_rate(self) -> float:
        return self.max_steer_rate_deg * 3.141592653589793 / 180.0


# =========================
#  バッチ対応 DynamicBicycleModel（PyTorch）
# =========================

class BatchedDynamicBicycleModel(nn.Module):
    """
    state: (B, 9)
        [0] x
        [1] y
        [2] psi
        [3] v
        [4] a
        [5] delta
        [6] beta
        [7] r
        [8] delta_ref

    action: (B, 2)
        [0] a_ref
        [1] d_delta_ref

    ここでの方針（あなたの要望）:
      - ZOH: 1回の step(dt) の間、action は固定
      - サブステップ: 内部刻み dt_internal で dt を満たすまで複数回 Euler 更新
      - タイヤ力飽和: Fy を +/- mu*Fz でクリップ（前後輪それぞれ）
    """

    def __init__(
        self,
        params: Optional[VehicleParams] = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        dt_internal: float = 0.01,  # 内部更新周期
        v_eff_min: float = 20./3.6,     # 低速特異性回避
    ):
        super().__init__()
        self.params = params if params is not None else VehicleParams()
        self.device = device
        self.dtype = dtype

        self.dt_internal = float(dt_internal)
        self.v_eff_min = float(v_eff_min)

        # resetで初期化される
        self.state: torch.Tensor
        self.F_yf: torch.Tensor
        self.F_yr: torch.Tensor

        # 定数Fy上限をキャッシュ（テンソル化）
        p = self.params
        L = p.lf + p.lr
        Fzf = (p.m * p.g) * (p.lr / L)
        Fzr = (p.m * p.g) * (p.lf / L)
        self.Fy_f_max = p.mu * Fzf
        self.Fy_r_max = p.mu * Fzr



    @property
    def batch_size(self) -> int:
        return self.state.shape[0]

    def reset(self, init_state: torch.Tensor) -> torch.Tensor:
        assert init_state.dim() == 2, f"init_state must be 2D (B,8|9), got {init_state.shape}"
        assert init_state.size(1) in (8, 9), f"last dim must be 8 or 9, got {init_state.size(1)}"

        init_state = init_state.to(device=self.device, dtype=self.dtype)

        if init_state.size(1) == 8:
            delta = init_state[:, 5:6]
            init_state = torch.cat([init_state, delta], dim=1)

        B = init_state.size(0)
        self.state = init_state.clone()

        self._postprocess_inplace(self.state)

        self.F_yf = torch.zeros(B, dtype=self.dtype, device=self.device)
        self.F_yr = torch.zeros(B, dtype=self.dtype, device=self.device)
        return self.state.clone()

    def _postprocess_inplace(self, state: torch.Tensor) -> None:
        p = self.params
        # v >= 0
        state[:, 3] = torch.clamp(state[:, 3], min=0.0)
        # delta, delta_ref within physical bounds
        state[:, 5] = torch.clamp(state[:, 5], -p.max_steer, p.max_steer)
        state[:, 8] = torch.clamp(state[:, 8], -p.max_steer, p.max_steer)

    def _tire_forces_with_saturation(self, beta, r, v_eff, delta):
        p = self.params
        alpha_f = beta + p.lf * r / v_eff - delta
        alpha_r = beta - p.lr * r / v_eff

        F_yf = -p.Cf * alpha_f
        F_yr = -p.Cr * alpha_r

        F_yf = torch.clamp(F_yf, -self.Fy_f_max, self.Fy_f_max)
        F_yr = torch.clamp(F_yr, -self.Fy_r_max, self.Fy_r_max)
        return F_yf, F_yr


    @torch.no_grad()
    def step(self, action: torch.Tensor, dt: float) -> torch.Tensor:
        """
        action はこの dt 区間で ZOH（固定）。
        内部刻み dt_internal で Euler を複数回回して dt を消化する。
        """
        p = self.params
        B = self.batch_size
        assert action.shape == (B, 2), f"action shape must be ({B},2), got {action.shape}"

        # --- 入力クリップ（ZOHで固定）---
        a_ref = torch.clamp(action[:, 0], p.min_accel, p.max_accel)
        # action[1] is now delta_ref command, not rate
        delta_ref_cmd = torch.clamp(action[:, 1], -p.max_steer, p.max_steer)

        # Update state delta_ref immediately (ZOH)
        self.state[:, 8] = delta_ref_cmd

        dt_total = float(dt)
        dt_int = float(self.dt_internal)
        if dt_int <= 0.0:
            raise ValueError(f"dt_internal must be > 0, got {dt_int}")

        # 固定回数 + 端数
        n_full = int(dt_total // dt_int)
        dt_rem = dt_total - n_full * dt_int

        # 力ログ（最後のサブステップの値にする／必要なら平均化も可能）
        last_F_yf = None
        last_F_yr = None

        def euler_substep(h: float) -> None:
            nonlocal last_F_yf, last_F_yr

            h_t = torch.as_tensor(h, dtype=self.dtype, device=self.device)

            # state unpack (views)
            x = self.state[:, 0]
            y = self.state[:, 1]
            psi = self.state[:, 2]
            v = self.state[:, 3]
            a = self.state[:, 4]
            delta = self.state[:, 5]
            beta = self.state[:, 6]
            r = self.state[:, 7]
            delta_ref = self.state[:, 8]  # Already updated

            # --- 1st order actuators ---
            a_dot = (a_ref - a) / p.tau_a
            delta_dot = (delta_ref - delta) / p.tau_delta

            # --- effective speed ---
            v_eff = torch.clamp(v, min=self.v_eff_min)

            # --- tire forces with saturation ---
            F_yf, F_yr = self._tire_forces_with_saturation(beta=beta, r=r, v_eff=v_eff, delta=delta)
            last_F_yf, last_F_yr = F_yf, F_yr

            # --- beta, r dynamics ---
            beta_dot = (F_yf + F_yr) / (p.m * v_eff) - r
            r_dot = (p.lf * F_yf - p.lr * F_yr) / p.Iz

            # --- longitudinal ---
            v_dot = a

            # --- kinematics ---
            psi_eff = psi + beta
            x_dot = v * torch.cos(psi_eff)
            y_dot = v * torch.sin(psi_eff)
            psi_dot = r

            # --- Euler update ---
            x = x + x_dot * h_t
            y = y + y_dot * h_t
            psi = psi + psi_dot * h_t
            v = v + v_dot * h_t
            a = a + a_dot * h_t
            delta = delta + delta_dot * h_t
            beta = beta + beta_dot * h_t
            r = r + r_dot * h_t

            # postprocess
            v = torch.clamp(v, min=0.0)
            delta = torch.clamp(delta, -p.max_steer, p.max_steer)

            # write back
            self.state[:, 0] = x
            self.state[:, 1] = y
            self.state[:, 2] = psi
            self.state[:, 3] = v
            self.state[:, 4] = a
            self.state[:, 5] = delta
            self.state[:, 6] = beta
            self.state[:, 7] = r
            # delta_ref is constant within step

        # full substeps
        for _ in range(n_full):
            euler_substep(dt_int)

        # remaining fraction
        if dt_rem > 1e-12:
            euler_substep(dt_rem)

        # save last forces
        if last_F_yf is not None:
            self.F_yf = last_F_yf
            self.F_yr = last_F_yr

        return self.state.clone()

    @torch.no_grad()
    def steering_from_kappa(self, kappa0: torch.Tensor) -> torch.Tensor:
        p = self.params
        kappa0 = kappa0.to(device=self.device, dtype=self.dtype)

        L = torch.as_tensor(p.lf + p.lr, dtype=self.dtype, device=self.device)
        delta_geom = torch.atan(L * kappa0)
        return torch.clamp(delta_geom, -p.max_steer, p.max_steer)



def calculate_max_d_delta_ref(
    vehicle_params: VehicleParams,
    max_dk_ds: Optional[float] = None,
    max_dk_dt: Optional[float] = None,
    v_max: Optional[float] = None,
) -> float:
    """
    軌道の曲率変化率から、幾何学的に必要となる最大ステアリング指令レート (d_delta_ref) を算出する。

    近似式:
        δ ≈ atan(L * κ)
        dδ/dt = (L / (1 + (L*κ)^2)) * (dκ/dt)
        conservative bound: |dδ/dt| <= L * |dκ/dt|

    Args:
        vehicle_params: 車両パラメータ (lf, lr を使用)
        max_dk_ds: 空間に対する最大曲率変化率 [1/m^2] (dκ/ds)
        max_dk_dt: 時間に対する最大曲率変化率 [1/(m*s)] (dκ/dt)
                   ※ max_dk_dt が与えられた場合はそちらを優先
                   ※ max_dk_ds を使う場合は v_max が必須
        v_max: 想定する最大速度 [m/s] (max_dk_ds を使う場合に必要)

    Returns:
        max_d_delta_ref: 推奨される最大ステアレート [rad/s]
    """
    # ホイールベース L = lf + lr
    L = vehicle_params.lf + vehicle_params.lr

    # dκ/dt (時間あたりの曲率変化) を決定
    if max_dk_dt is not None:
        target_dk_dt = max_dk_dt
    elif max_dk_ds is not None and v_max is not None:
        # dκ/dt = (dκ/ds) * (ds/dt) = (dκ/ds) * v
        target_dk_dt = max_dk_ds * v_max
    else:
        raise ValueError("max_dk_dt または (max_dk_ds と v_max) のどちらかを指定してください。")

    # dδ/dt ≈ L * dκ/dt
    # ※ 厳密には dδ/dt = L/(1+(Lκ)^2) * dκ/dt ですが、
    #    分母 >= 1 なので L * dκ/dt が安全側の最大見積もり(上限)になります。
    max_d_delta_ref = L * target_dk_dt

    return max_d_delta_ref
