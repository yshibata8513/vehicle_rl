
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple, Dict


# =========================
#  Vehicle parameters
# =========================

@dataclass
class VehicleParams:
    m: float = 1500.0      # mass [kg]
    Iz: float = 2250.0     # yaw inertia [kg m^2]
    lf: float = 1.2        # CG to front axle [m]
    lr: float = 1.6        # CG to rear axle [m]
    Cf: float = 8.0e4      # front cornering stiffness [N/rad]
    Cr: float = 8.0e4      # rear cornering stiffness [N/rad]

    tau_a: float = 0.1     # accel actuator time constant [s]
    tau_delta: float = 0.1 # steering actuator time constant [s]

    max_steer_deg: float = 30.0        # max steering angle [deg]
    max_steer_rate_deg: float = 180.0  # (compat) max steering command rate [deg/s]
    max_accel: float = 3.0             # max accel [m/s^2]
    min_accel: float = -6.0            # max decel [m/s^2] (negative)

    # ---- tire force saturation (simple) ----
    mu: float = 0.9        # friction coefficient
    g: float = 9.81        # gravity [m/s^2]

    @property
    def max_steer(self) -> float:
        return self.max_steer_deg * 3.141592653589793 / 180.0

    @property
    def max_steer_rate(self) -> float:
        return self.max_steer_rate_deg * 3.141592653589793 / 180.0


# =========================
#  Differentiable batched dynamic bicycle model
# =========================
#
# State: (B, 9)
#   [0] x         [m]
#   [1] y         [m]
#   [2] psi       [rad]
#   [3] v         [m/s]
#   [4] a         [m/s^2]
#   [5] delta     [rad]
#   [6] beta      [rad]
#   [7] r         [rad/s]
#   [8] delta_ref [rad]  (ZOH command for steering actuator)
#
# Action: (B, 2)
#   [0] a_ref          [m/s^2]  (ZOH)
#   [1] delta_ref_cmd  [rad]    (ZOH)
#
# Notes:
# - Differentiable w.r.t. action (and network parameters producing the action).
# - No @torch.no_grad(), and no in-place column writes inside integration.
# - clamp-based saturations remain; gradients are subgradients and can vanish when saturated.
#
class BatchedDifferentiableDynamicBicycleModel(nn.Module):
    def __init__(
        self,
        params: Optional[VehicleParams] = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        dt_internal: float = 0.01,
        v_eff_min: float = 20.0 / 3.6,   # avoid low-speed singularity
    ):
        super().__init__()
        self.params = params if params is not None else VehicleParams()
        self.device = torch.device(device)
        self.dtype = dtype

        self.dt_internal = float(dt_internal)
        self.v_eff_min = float(v_eff_min)

        # stateful buffers (optional; for env-style usage)
        self.state: torch.Tensor = torch.zeros(1, 9, device=self.device, dtype=self.dtype)
        self.F_yf: torch.Tensor = torch.zeros(1, device=self.device, dtype=self.dtype)
        self.F_yr: torch.Tensor = torch.zeros(1, device=self.device, dtype=self.dtype)

        # precompute constant tire-force saturation bounds from static normal loads
        p = self.params
        L = p.lf + p.lr
        Fzf = (p.m * p.g) * (p.lr / L)
        Fzr = (p.m * p.g) * (p.lf / L)
        self._Fy_f_max = float(p.mu * Fzf)
        self._Fy_r_max = float(p.mu * Fzr)

    @property
    def batch_size(self) -> int:
        return int(self.state.shape[0])

    # ---------- stateless / functional core ----------
    def tire_forces_with_saturation(
        self,
        beta: torch.Tensor,
        r: torch.Tensor,
        v_eff: torch.Tensor,
        delta: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        p = self.params
        alpha_f = beta + p.lf * r / v_eff - delta
        alpha_r = beta - p.lr * r / v_eff

        Fy_f_max = torch.as_tensor(self._Fy_f_max, device=self.device, dtype=self.dtype)
        Fy_r_max = torch.as_tensor(self._Fy_r_max, device=self.device, dtype=self.dtype)

        # F = F_max * tanh( -Cf * alpha / F_max )
        F_yf = Fy_f_max * torch.tanh(-p.Cf * alpha_f / Fy_f_max)
        F_yr = Fy_r_max * torch.tanh(-p.Cr * alpha_r / Fy_r_max)

        return F_yf, F_yr

    def functional_step(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        dt: float,
        return_info: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Differentiable one-step transition: next_state = f(state, action, dt)

        state: (B,9) or (B,8). If (B,8), delta_ref is initialized to delta.
        action: (B,2) = [a_ref, delta_ref_cmd]
        """
        if state.dim() != 2:
            raise ValueError(f"state must be 2D (B,8|9), got {tuple(state.shape)}")
        if state.size(1) not in (8, 9):
            raise ValueError(f"state last dim must be 8 or 9, got {state.size(1)}")
        if action.dim() != 2 or action.size(1) != 2:
            raise ValueError(f"action must be (B,2), got {tuple(action.shape)}")
        if state.size(0) != action.size(0):
            raise ValueError(f"batch size mismatch: state B={state.size(0)} action B={action.size(0)}")

        p = self.params
        state = state.to(device=self.device, dtype=self.dtype)
        action = action.to(device=self.device, dtype=self.dtype)

        # (B,8) -> (B,9) by copying delta into delta_ref
        if state.size(1) == 8:
            delta0 = state[:, 5:6]
            state = torch.cat([state, delta0], dim=1)

        # clip inputs (ZOH)
        a_ref = torch.clamp(action[:, 0], p.min_accel, p.max_accel)
        delta_ref = torch.clamp(action[:, 1], -p.max_steer, p.max_steer)

        # unpack
        x, y, psi, v, a, delta, beta, r, _delta_ref_old = state.unbind(dim=1)

        dt_total = float(dt)
        dt_int = float(self.dt_internal)
        if dt_int <= 0.0:
            raise ValueError(f"dt_internal must be > 0, got {dt_int}")
        if dt_total < 0.0:
            raise ValueError(f"dt must be >= 0, got {dt_total}")

        n_full = int(dt_total // dt_int)
        dt_rem = dt_total - n_full * dt_int

        last_F_yf = None
        last_F_yr = None

        def substep(h: float):
            nonlocal x, y, psi, v, a, delta, beta, r, last_F_yf, last_F_yr
            if h <= 0.0:
                return
            h_t = torch.as_tensor(h, device=self.device, dtype=self.dtype)

            # 1st-order actuators
            a_dot = (a_ref - a) / p.tau_a
            delta_dot = (delta_ref - delta) / p.tau_delta

            # avoid low-speed singularity
            v_eff = torch.clamp(v, min=self.v_eff_min)

            # tire forces
            F_yf, F_yr = self.tire_forces_with_saturation(beta=beta, r=r, v_eff=v_eff, delta=delta)
            last_F_yf, last_F_yr = F_yf, F_yr

            # lateral dynamics
            beta_dot = (F_yf + F_yr) / (p.m * v_eff) - r
            r_dot = (p.lf * F_yf - p.lr * F_yr) / p.Iz

            # longitudinal
            v_dot = a

            # kinematics
            psi_eff = psi + beta
            x_dot = v * torch.cos(psi_eff)
            y_dot = v * torch.sin(psi_eff)
            psi_dot = r

            # Euler update (rebind vars)
            x = x + x_dot * h_t
            y = y + y_dot * h_t
            psi = psi + psi_dot * h_t
            v = v + v_dot * h_t
            a = a + a_dot * h_t
            delta = delta + delta_dot * h_t
            beta = beta + beta_dot * h_t
            r = r + r_dot * h_t

            # clamps
            v = torch.clamp(v, min=0.0)
            delta = torch.clamp(delta, -p.max_steer, p.max_steer)

        for _ in range(n_full):
            substep(dt_int)
        substep(dt_rem)

        next_state = torch.stack([x, y, psi, v, a, delta, beta, r, delta_ref], dim=1)

        if not return_info:
            return next_state, None

        if last_F_yf is None:
            # dt == 0 case
            last_F_yf = torch.zeros_like(v)
            last_F_yr = torch.zeros_like(v)

        info = {"F_yf": last_F_yf, "F_yr": last_F_yr}
        return next_state, info

    # ---------- stateful API (env-style usage) ----------
    def reset(self, init_state: torch.Tensor) -> torch.Tensor:
        """
        init_state: (B,8|9). If (B,8), delta_ref is set to delta.
        """
        init_state = init_state.to(device=self.device, dtype=self.dtype)
        if init_state.dim() != 2 or init_state.size(1) not in (8, 9):
            raise ValueError(f"init_state must be (B,8|9), got {tuple(init_state.shape)}")

        if init_state.size(1) == 8:
            delta0 = init_state[:, 5:6]
            init_state9 = torch.cat([init_state, delta0], dim=1)
        else:
            init_state9 = init_state

        # postprocess clamps
        p = self.params
        init_state9 = init_state9.clone()
        init_state9[:, 3] = torch.clamp(init_state9[:, 3], min=0.0)
        init_state9[:, 5] = torch.clamp(init_state9[:, 5], -p.max_steer, p.max_steer)
        init_state9[:, 8] = torch.clamp(init_state9[:, 8], -p.max_steer, p.max_steer)

        self.state = init_state9
        B = init_state9.size(0)
        self.F_yf = torch.zeros(B, device=self.device, dtype=self.dtype)
        self.F_yr = torch.zeros(B, device=self.device, dtype=self.dtype)
        return self.state.clone()

    def step(self, action: torch.Tensor, dt: float, compute_info: bool = True) -> torch.Tensor:
        """
        Differentiable step that updates internal self.state.
        """
        next_state, info = self.functional_step(self.state, action, dt, return_info=compute_info)
        self.state = next_state
        if compute_info and info is not None:
            self.F_yf = info["F_yf"]
            self.F_yr = info["F_yr"]
        return self.state.clone()

    # ---------- utility (differentiable) ----------
    def steering_from_kappa(self, kappa0: torch.Tensor) -> torch.Tensor:
        """
        Geometric steering: delta = atan(L * kappa)
        """
        p = self.params
        kappa0 = kappa0.to(device=self.device, dtype=self.dtype)
        L = torch.as_tensor(p.lf + p.lr, device=self.device, dtype=self.dtype)
        delta_geom = torch.atan(L * kappa0)
        return torch.clamp(delta_geom, -p.max_steer, p.max_steer)

    def max_speed_for_kappa(
        self,
        kappa: torch.Tensor,
        v_max_in: Optional[torch.Tensor] = None,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Lateral acceleration limit: v <= sqrt(mu*g / |kappa|).
        """
        p = self.params
        kappa_t = torch.as_tensor(kappa, device=self.device, dtype=self.dtype)
        mu_g = torch.as_tensor(p.mu * p.g, device=self.device, dtype=self.dtype)
        denom = torch.abs(kappa_t) + float(eps)
        v_limit = torch.sqrt(mu_g / denom)
        if v_max_in is None:
            return v_limit
        v_max_in_t = torch.as_tensor(v_max_in, device=self.device, dtype=self.dtype)
        return torch.minimum(v_limit, v_max_in_t)


def calculate_max_d_delta_ref(
    vehicle_params: VehicleParams,
    max_dk_ds: Optional[float] = None,
    max_dk_dt: Optional[float] = None,
    v_max: Optional[float] = None,
) -> float:
    """
    Safe upper bound for |d(delta_ref)/dt| from curvature change rate.
        delta ≈ atan(L*kappa)
        ddelta/dt = L/(1+(L*kappa)^2) * dkappa/dt
    Since the denominator >= 1, L*dkappa/dt is a conservative upper bound.
    """
    L = float(vehicle_params.lf + vehicle_params.lr)

    if max_dk_dt is not None:
        target_dk_dt = float(max_dk_dt)
    elif max_dk_ds is not None and v_max is not None:
        target_dk_dt = float(max_dk_ds) * float(v_max)
    else:
        raise ValueError("Specify max_dk_dt or (max_dk_ds and v_max).")

    return L * target_dk_dt



if __name__ == "__main__":

    model = BatchedDifferentiableDynamicBicycleModel(VehicleParams(), device="cpu", dtype=torch.float64)

    B = 4
    state = torch.zeros(B, 9, dtype=torch.float64)
    state[:, 3] = 10.0           # v
    state[:, 5] = 0.01           # delta
    state[:, 8] = 0.01           # delta_ref

    action = torch.tensor([[1.0, 0.02]] * B, dtype=torch.float64, requires_grad=True)  # [a_ref, delta_ref_cmd]

    next_state, info = model.functional_step(state, action, dt=0.1, return_info=True)
    loss = next_state[:, 0].sum()  # 例：xの合計
    loss.backward()
    print(action.grad)  # ← 非ゼロになればOK
