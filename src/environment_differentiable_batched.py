import numpy as np
import torch
from dataclasses import dataclass
from typing import Optional, Sequence, List, Dict, Any, Callable, Tuple

from bicycle_model_differentiable_batched import BatchedDifferentiableDynamicBicycleModel, VehicleParams
try:
    from trajectory_generator import ReferenceTrajectory
except Exception:  # pragma: no cover
    # Fallback stub for type-checking / standalone import.
    @dataclass
    class ReferenceTrajectory:  # type: ignore
        s_ref: Sequence[float]
        x_ref: Sequence[float]
        y_ref: Sequence[float]
        v_ref: Sequence[float]
        kappa_ref: Sequence[float]
        dt: float


# =========================
#  Reward weights
# =========================

@dataclass
class RewardWeights:
    # lateral / heading / speed
    w_y: float = 0.1
    w_psi: float = 0.2
    w_v_under: float = 0.05
    w_v_over: float = 0.2

    # comfort / actuation
    w_ay: float = 0.02
    w_d_delta_ref: float = 0.01
    w_dd_delta_ref: float = 0.005


    # curvature tracking
    w_kappa: float = 0.0

    # tire saturation (slip angle excess)
    w_tire_alpha_excess: float = 0.0
    tire_util_threshold: float = 0.8
    # loss type: "l1" or "l2"
    loss_y: str = "l2"
    loss_psi: str = "l2"
    loss_v: str = "l2"
    loss_ay: str = "l2"
    loss_d_delta_ref: str = "l2"
    loss_dd_delta_ref: str = "l2"


    loss_kappa: str = "l2"
    loss_tire_alpha_excess: str = "l2"
# =========================
#  Differentiable env state container
# =========================

@dataclass
class BatchedFrenetEnvState:
    """
    All tensors are batched with leading dimension B (batch size = number of reference trajectories).
    """
    vehicle_state: torch.Tensor          # (B, 9)
    s: torch.Tensor                      # (B,)
    e_y: torch.Tensor                    # (B,)
    e_psi_v: torch.Tensor                # (B,)
    step_count: torch.Tensor             # (B,) int64
    delta_ref_history: torch.Tensor      # (B, 5)


class BatchedPathTrackingEnvFrenetDifferentiable:
    """
    Differentiable batched path-tracking environment in Frenet coordinates.

    - Designed to coexist with the original env implementation.
    - Provides a PURE functional step `functional_step()` that avoids in-place tensor writes
      inside the transition, so you can safely unroll for gradient-based RL.
    - Also provides a stateful `step()` for compatibility (implemented by calling functional_step
      and replacing internal state tensors).

    Observation (B, 6 + 5 + P):
      [e_y, e_psi_v, v, a, r, v_ref_now, delta_ref_history(5), kappa_preview(P)]
    State vector (B, 8):
      [e_y, e_psi_v, v, a, delta, delta_ref, beta, r]
    """

    def __init__(
        self,
        ref_trajs: List[ReferenceTrajectory],
        kappa_preview_offsets: Optional[Sequence[float]] = None,
        vehicle_params: Optional[VehicleParams] = None,
        reward_weights: Optional[RewardWeights] = None,
        max_lateral_error: float = 5.0,
        max_steps: int = 1000,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        traj_generator: Optional[Callable[[], ReferenceTrajectory]] = None,
        angle_wrap_mode: str = "atan2",  # "atan2" (recommended) or "mod"
    ):
        self.device = torch.device(device)
        self.dtype = dtype
        self.traj_generator = traj_generator
        self.angle_wrap_mode = angle_wrap_mode

        self.B = len(ref_trajs)
        if self.B <= 0:
            raise ValueError("ref_trajs must be non-empty")

        # ---------- s_ref must be identical across batch ----------
        s_ref0 = np.asarray(ref_trajs[0].s_ref, dtype=np.float64)
        N = int(s_ref0.shape[0])
        if N < 2:
            raise ValueError("s_ref must have at least 2 points")

        for i, traj in enumerate(ref_trajs[1:], start=1):
            s_ref_i = np.asarray(traj.s_ref, dtype=np.float64)
            if not np.allclose(s_ref_i, s_ref0):
                raise ValueError(f"s_ref of trajectory {i} differs from trajectory 0")

        # NOTE: convert via list to avoid from_numpy issues with readonly buffers etc.
        self.s_ref = torch.tensor(s_ref0.tolist(), dtype=self.dtype, device=self.device)  # (N,)
        self.Ns = N
        self.s_end = torch.tensor(float(s_ref0[-1]), dtype=self.dtype, device=self.device)

        self.dt = float(ref_trajs[0].dt)

        # ---------- reference mats (B, N) ----------
        self.v_ref_mat = torch.zeros(self.B, N, dtype=self.dtype, device=self.device)
        self.kappa_ref_mat = torch.zeros(self.B, N, dtype=self.dtype, device=self.device)
        self.psi_ref_mat = torch.zeros(self.B, N, dtype=self.dtype, device=self.device)

        for b, traj in enumerate(ref_trajs):
            x_ref = np.asarray(traj.x_ref, dtype=np.float64)
            y_ref = np.asarray(traj.y_ref, dtype=np.float64)
            v_ref = np.asarray(traj.v_ref, dtype=np.float64)
            kappa_ref = np.asarray(traj.kappa_ref, dtype=np.float64)

            if not (len(x_ref) == N and len(y_ref) == N and len(v_ref) == N and len(kappa_ref) == N):
                raise ValueError("All reference arrays must have same length as s_ref")

            self.v_ref_mat[b] = torch.tensor(v_ref.tolist(), dtype=self.dtype, device=self.device)
            self.kappa_ref_mat[b] = torch.tensor(kappa_ref.tolist(), dtype=self.dtype, device=self.device)

            dx = np.diff(x_ref)
            dy = np.diff(y_ref)
            psi = np.arctan2(dy, dx)
            if len(psi) > 0:
                psi = np.concatenate([psi, psi[-1:]])
            else:
                psi = np.array([0.0], dtype=np.float64)
            self.psi_ref_mat[b] = torch.tensor(psi.tolist(), dtype=self.dtype, device=self.device)

        # preview offsets (kappa)
        if kappa_preview_offsets is None:
            self.kappa_preview_offsets = torch.arange(21, device=self.device, dtype=self.dtype) * 1.0
        else:
            self.kappa_preview_offsets = torch.tensor(kappa_preview_offsets, dtype=self.dtype, device=self.device)
        self.P = int(self.kappa_preview_offsets.shape[0])

        # delta_ref history (B,5)
        self.delta_ref_history = torch.zeros(self.B, 5, dtype=self.dtype, device=self.device)

        # vehicle model (differentiable)
        self.vehicle = BatchedDifferentiableDynamicBicycleModel(
            params=vehicle_params,
            device=str(self.device),
            dtype=self.dtype,
        )

        self.weights = reward_weights if reward_weights is not None else RewardWeights()
        self.max_lateral_error = float(max_lateral_error)
        self.max_steps = int(max_steps)

        # stateful env variables (used only by reset()/step())
        self.s = torch.zeros(self.B, dtype=self.dtype, device=self.device)
        self.e_y = torch.zeros(self.B, dtype=self.dtype, device=self.device)
        self.e_psi_v = torch.zeros(self.B, dtype=self.dtype, device=self.device)
        self.step_count = torch.zeros(self.B, dtype=torch.int64, device=self.device)

        # for debug / logging in stateful mode
        self._cache: Dict[str, torch.Tensor] = {}

    # -----------------
    # helpers
    # -----------------

    def _penalty(self, x: torch.Tensor, loss_type: str) -> torch.Tensor:
        if loss_type == "l1":
            return torch.abs(x)
        if loss_type == "l2":
            return x ** 2
        raise ValueError(f"Unknown loss_type: {loss_type} (expected 'l1' or 'l2')")

    def _wrap_angle(self, ang: torch.Tensor) -> torch.Tensor:
        if self.angle_wrap_mode == "mod":
            pi = np.pi
            return (ang + pi) % (2.0 * pi) - pi
        # differentiable and smooth (except at pi/-pi due to atan2, but much nicer than mod)
        return torch.atan2(torch.sin(ang), torch.cos(ang))

    def _ensure_vehicle_init_state(self, init_state: torch.Tensor) -> torch.Tensor:
        if not (init_state.dim() == 2 and init_state.shape[0] == self.B):
            raise ValueError(f"init_state must be (B,8) or (B,9) with B={self.B}, got {tuple(init_state.shape)}")
        if init_state.shape[1] not in (8, 9):
            raise ValueError(f"init_state last dim must be 8 or 9, got {init_state.shape[1]}")
        init_state = init_state.to(device=self.device, dtype=self.dtype)
        if init_state.shape[1] == 9:
            return init_state
        delta = init_state[:, 5:6]
        return torch.cat([init_state, delta], dim=1)

    def _interp_matrix(self, s: torch.Tensor, y_mat: torch.Tensor) -> torch.Tensor:
        """
        Linear interpolation for each batch element over a shared s_ref grid.

        Note: uses searchsorted (piecewise differentiable in s).
        """
        B = self.B
        N = self.Ns

        s_clamped = torch.clamp(s, self.s_ref[0], self.s_ref[-1] - 1e-6)
        idx = torch.searchsorted(self.s_ref, s_clamped, right=False)
        idx = torch.clamp(idx, 0, N - 2)

        s0 = self.s_ref[idx]
        s1 = self.s_ref[idx + 1]
        t = (s_clamped - s0) / (s1 - s0 + 1e-8)

        batch_idx = torch.arange(B, device=self.device)
        y0 = y_mat[batch_idx, idx]
        y1 = y_mat[batch_idx, idx + 1]
        return (1.0 - t) * y0 + t * y1

    def _interp_matrix_preview(self, s: torch.Tensor, y_mat: torch.Tensor) -> torch.Tensor:
        B = self.B
        N = self.Ns
        P = self.P

        s_q = s.unsqueeze(1) + self.kappa_preview_offsets.unsqueeze(0)
        s_q = torch.clamp(s_q, self.s_ref[0], self.s_ref[-1] - 1e-6)

        s_q_flat = s_q.reshape(-1)
        idx_q = torch.searchsorted(self.s_ref, s_q_flat, right=False)
        idx_q = torch.clamp(idx_q, 0, N - 2)

        s0_q = self.s_ref[idx_q]
        s1_q = self.s_ref[idx_q + 1]
        t_q = (s_q_flat - s0_q) / (s1_q - s0_q + 1e-8)

        batch_flat = torch.arange(B, device=self.device).unsqueeze(1).expand(B, P).reshape(-1)
        y0_q = y_mat[batch_flat, idx_q]
        y1_q = y_mat[batch_flat, idx_q + 1]
        y_flat = (1.0 - t_q) * y0_q + t_q * y1_q
        return y_flat.view(B, P)

    # -----------------
    # observation and reward (pure functions)
    # -----------------

    def compute_obs_state(
        self,
        env_state: BatchedFrenetEnvState,
        *,
        return_cache: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Pure computation of (obs_raw, state_vec) from env_state.
        """
        v = env_state.vehicle_state[:, 3]
        a = env_state.vehicle_state[:, 4]
        delta = env_state.vehicle_state[:, 5]
        beta = env_state.vehicle_state[:, 6]
        r = env_state.vehicle_state[:, 7]
        delta_ref = env_state.vehicle_state[:, 8]

        v_ref_now = self._interp_matrix(env_state.s, self.v_ref_mat)
        kappa_preview = self._interp_matrix_preview(env_state.s, self.kappa_ref_mat)
        # kappa0 is used for dynamics reward and frenet update
        if float(self.kappa_preview_offsets[0].item()) == 0.0:
            kappa0 = kappa_preview[:, 0]
        else:
            kappa0 = self._interp_matrix(env_state.s, self.kappa_ref_mat)

        e_y = env_state.e_y
        e_psi_v = self._wrap_angle(env_state.e_psi_v)

        obs_list = [
            e_y,
            e_psi_v,
            v,
            a,
            r,
            v_ref_now,
            env_state.delta_ref_history,  # (B,5)
            kappa_preview,                # (B,P)
        ]
        obs = torch.cat([x if x.dim() == 2 else x.unsqueeze(1) for x in obs_list], dim=1)
        state_vec = torch.stack([e_y, e_psi_v, v, a, delta, delta_ref, beta, r], dim=1)

        cache = {
            "v": v,
            "a": a,
            "delta": delta,
            "beta": beta,
            "r": r,
            "delta_ref": delta_ref,
            "v_ref_now": v_ref_now,
            "kappa0": kappa0,
            "kappa_preview": kappa_preview,
            "e_y": e_y,
            "e_psi_v": e_psi_v,
        }
        return obs, state_vec, cache

    def compute_reward_done_info(
        self,
        env_state: BatchedFrenetEnvState,
        action: torch.Tensor,
        *,
        d_delta_ref: Optional[torch.Tensor] = None,
        dd_delta_ref: Optional[torch.Tensor] = None,
        F_yf: Optional[torch.Tensor] = None,
        F_yr: Optional[torch.Tensor] = None,
        alpha_f: Optional[torch.Tensor] = None,
        alpha_r: Optional[torch.Tensor] = None,
        compute_info: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, Any]]]:
        """
        Pure reward/done computation.
        """
        w = self.weights
        p = self.vehicle.params

        obs_raw, _state_vec, cache = self.compute_obs_state(env_state, return_cache=True)
        v = cache["v"]
        e_y = cache["e_y"]
        e_psi_v = cache["e_psi_v"]
        v_ref_now = cache["v_ref_now"]
        delta_ref = cache["delta_ref"]
        beta = cache["beta"]
        kappa0 = cache["kappa0"]

        # geometric steer from curvature (used for logging)
        delta_geom = self.vehicle.steering_from_kappa(kappa0)

        cost_y = w.w_y * self._penalty(e_y, w.loss_y)

        e_psi_vel = self._wrap_angle(e_psi_v + beta)
        cost_psi = w.w_psi * self._penalty(e_psi_vel, w.loss_psi)

        dv = v - v_ref_now
        dv_pen = self._penalty(dv, w.loss_v)
        cost_v_under = w.w_v_under * dv_pen
        cost_v_over = w.w_v_over * dv_pen
        cost_v = torch.where(v <= v_ref_now, cost_v_under, cost_v_over)

        # lateral acceleration from tire forces
        F_yf_use = self.vehicle.F_yf if F_yf is None else F_yf
        F_yr_use = self.vehicle.F_yr if F_yr is None else F_yr
        a_y = (F_yf_use + F_yr_use) / p.m
        cost_ay = w.w_ay * self._penalty(a_y, w.loss_ay)


        cost_d_delta_ref = w.w_d_delta_ref * self._penalty(d_delta_ref, w.loss_d_delta_ref)
        cost_dd_delta_ref = w.w_dd_delta_ref * self._penalty(dd_delta_ref, w.loss_dd_delta_ref)
        # curvature tracking (path curvature from lateral accel)
        v_eff = torch.clamp(v, min=float(getattr(self.vehicle, 'v_eff_min', 1e-3)))
        v_kappa = torch.clamp(v, min=1e-3)
        kappa_hat = a_y / (v_eff * v_kappa)
        e_kappa = kappa_hat - kappa0
        cost_kappa = w.w_kappa * self._penalty(e_kappa, w.loss_kappa)

        alpha_excess_f, alpha_excess_r, _alpha_th_f, _alpha_th_r = self.vehicle.tire_alpha_excess_from_alphas(
            alpha_f=alpha_f,
            alpha_r=alpha_r,
            util_threshold=float(getattr(w, 'tire_util_threshold', 0.8)),
        )
        cost_tire_alpha_excess = w.w_tire_alpha_excess * (
            self._penalty(alpha_excess_f, w.loss_tire_alpha_excess)
            + self._penalty(alpha_excess_r, w.loss_tire_alpha_excess)
        )

        total_cost = cost_y + cost_psi + cost_v + cost_ay + cost_d_delta_ref + cost_dd_delta_ref + cost_kappa + cost_tire_alpha_excess
        reward = -total_cost

        done_lateral = torch.abs(e_y) > self.max_lateral_error
        done_pathend = env_state.s >= self.s_end
        done_steps = env_state.step_count >= self.max_steps
        done = done_lateral | done_pathend | done_steps

        info: Optional[Dict[str, Any]]
        if compute_info:
            info = {
                "cost_y": cost_y,
                "cost_psi": cost_psi,
                "cost_v": cost_v,
                "cost_ay": cost_ay,
                "cost_d_delta_ref": cost_d_delta_ref,
                "cost_dd_delta_ref": cost_dd_delta_ref,
                "cost_kappa": cost_kappa,
                "cost_tire_alpha_excess": cost_tire_alpha_excess,
                "alpha_f": alpha_f,
                "alpha_r": alpha_r,
                "kappa_hat": kappa_hat,
                "e_kappa": e_kappa,
                "a_y": a_y,
                "v_ref": v_ref_now,
                "s": env_state.s,
                "kappa0": kappa0,
                "delta_geom": delta_geom,
                "d_delta_ref": d_delta_ref,
                "dd_delta_ref": dd_delta_ref,
                "delta_ref": delta_ref,
                "done_lateral": done_lateral,
                "done_pathend": done_pathend,
                "done_steps": done_steps,
            }
        else:
            info = None

        return reward, done, info

    # -----------------
    # functional step
    # -----------------

    def functional_step(
        self,
        env_state: BatchedFrenetEnvState,
        action: torch.Tensor,
        *,
        compute_info: bool = True,
        normalize_obs: bool = True,
        obs_clip: Optional[float] = 5.0,
    ) -> Tuple[
        BatchedFrenetEnvState,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[Dict[str, Any]],
    ]:
        """
        Pure transition:
        (env_state, action) -> (next_env_state, obs_norm, obs_raw, state_vec, reward, done, info)
        """
        if not (action.shape[0] == self.B and action.shape[1] == 2):
            raise ValueError(f"action must be (B,2) with B={self.B}, got {tuple(action.shape)}")

        dt = float(self.dt)
        dt_t = torch.as_tensor(dt, dtype=self.dtype, device=self.device)

        # --- vehicle + Frenet integration (pure) ---
        delta_ref_old = env_state.vehicle_state[:, 8]

        dt_int = float(self.vehicle.dt_internal)
        if dt_int <= 0.0:
            raise ValueError(f"dt_internal must be > 0, got {dt_int}")

        n_full = int(dt // dt_int)
        dt_rem = dt - n_full * dt_int

        veh = env_state.vehicle_state
        s = env_state.s
        e_y = env_state.e_y
        e_psi_v = env_state.e_psi_v

        F_yf = None
        F_yr = None
        alpha_f = None
        alpha_r = None

        def frenet_substep(
            veh_state: torch.Tensor,
            s: torch.Tensor,
            e_y: torch.Tensor,
            e_psi_v: torch.Tensor,
            h: float,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            h_t = torch.as_tensor(h, dtype=self.dtype, device=self.device)

            v = veh_state[:, 3]
            beta = veh_state[:, 6]
            r = veh_state[:, 7]

            kappa0 = self._interp_matrix(s, self.kappa_ref_mat)

            denom = 1.0 - kappa0 * e_y
            denom = torch.clamp(denom, min=0.1)

            # 速度ベクトルの向き (e_psi_v + beta) を見るべき
            s_dot = v * torch.cos(e_psi_v + beta) / denom
            e_y_dot = v * torch.sin(e_psi_v + beta)
            e_psi_v_dot = r - kappa0 * s_dot

            s = s + s_dot * h_t
            e_y = e_y + e_y_dot * h_t
            e_psi_v = self._wrap_angle(e_psi_v + e_psi_v_dot * h_t)
            return s, e_y, e_psi_v

        for _ in range(n_full):
            veh, vinfo = self.vehicle.functional_step(veh, action, dt=dt_int, return_info=True)
            F_yf = vinfo.get("F_yf", None)
            F_yr = vinfo.get("F_yr", None)
            alpha_f = vinfo.get("alpha_f", None)
            alpha_r = vinfo.get("alpha_r", None)
            s, e_y, e_psi_v = frenet_substep(veh, s, e_y, e_psi_v, dt_int)

        if dt_rem > 0.0:
            veh, vinfo = self.vehicle.functional_step(veh, action, dt=dt_rem, return_info=True)
            F_yf = vinfo.get("F_yf", None)
            F_yr = vinfo.get("F_yr", None)
            alpha_f = vinfo.get("alpha_f", None)
            alpha_r = vinfo.get("alpha_r", None)
            s, e_y, e_psi_v = frenet_substep(veh, s, e_y, e_psi_v, dt_rem)

        next_vehicle_state = veh
        next_s = s
        next_e_y = e_y
        next_e_psi_v = e_psi_v

        # --- update delta_ref history (pure shift) ---
        next_delta_ref_history = torch.cat(
            [next_vehicle_state[:, 8:9], env_state.delta_ref_history[:, :-1]],
            dim=1,
        )

        next_step_count = env_state.step_count + 1

        next_env_state = BatchedFrenetEnvState(
            vehicle_state=next_vehicle_state,
            s=next_s,
            e_y=next_e_y,
            e_psi_v=next_e_psi_v,
            step_count=next_step_count,
            delta_ref_history=next_delta_ref_history,
        )

        # --- d_delta_ref and dd_delta_ref for reward ---
        p = self.vehicle.params
        delta_ref_new_cmd = torch.clamp(action[:, 1], -p.max_steer, p.max_steer)
        d_delta_ref = (delta_ref_new_cmd - delta_ref_old) / dt

        rate_now = (next_delta_ref_history[:, 0] - next_delta_ref_history[:, 1]) / dt
        rate_prev = (next_delta_ref_history[:, 1] - next_delta_ref_history[:, 2]) / dt
        dd_delta_ref = (rate_now - rate_prev) / dt
        valid = (next_step_count >= 3).to(dtype=self.dtype)
        dd_delta_ref = dd_delta_ref * valid

        # --- obs/state ---
        obs_raw, state_vec, _cache = self.compute_obs_state(next_env_state, return_cache=True)
        obs_out = self.normalize_obs(obs_raw, clip=obs_clip) if normalize_obs else obs_raw

        # --- reward/done/info ---
        reward, done, info = self.compute_reward_done_info(
            next_env_state,
            action=action,
            d_delta_ref=d_delta_ref,
            dd_delta_ref=dd_delta_ref,
            F_yf=F_yf,
            F_yr=F_yr,
            alpha_f=alpha_f,
            alpha_r=alpha_r,
            compute_info=compute_info,
        )

        return next_env_state, obs_out, obs_raw, state_vec, reward, done, info

    # -----------------
    # stateful API (compatible with old env)
    # -----------------

    def get_env_state(self) -> BatchedFrenetEnvState:
        return BatchedFrenetEnvState(
            vehicle_state=self.vehicle.state,
            s=self.s,
            e_y=self.e_y,
            e_psi_v=self.e_psi_v,
            step_count=self.step_count,
            delta_ref_history=self.delta_ref_history,
        )

    def set_env_state(self, env_state: BatchedFrenetEnvState) -> None:
        # replace tensors (avoid in-place modifications on existing tensors)
        self.vehicle.state = env_state.vehicle_state
        self.s = env_state.s
        self.e_y = env_state.e_y
        self.e_psi_v = env_state.e_psi_v
        self.step_count = env_state.step_count
        self.delta_ref_history = env_state.delta_ref_history

    def reset(
        self,
        init_state: torch.Tensor,
        s0: Optional[torch.Tensor] = None,
        is_perturbed: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        init_state_v = self._ensure_vehicle_init_state(init_state)
        self.vehicle.reset(init_state_v)

        self.delta_ref_history = torch.zeros(self.B, 5, dtype=self.dtype, device=self.device)

        if s0 is None:
            self.s = torch.full((self.B,), float(self.s_ref[0]), dtype=self.dtype, device=self.device)
        else:
            if s0.shape[0] != self.B:
                raise ValueError(f"s0 must have shape (B,), got {tuple(s0.shape)}")
            self.s = s0.to(device=self.device, dtype=self.dtype)

        self.e_y = torch.zeros(self.B, dtype=self.dtype, device=self.device)
        self.e_psi_v = torch.zeros(self.B, dtype=self.dtype, device=self.device)

        if is_perturbed:
            self.e_y = self.e_y + (torch.rand(self.B, device=self.device, dtype=self.dtype) * 2.0 - 1.0)

            dv_max = 10.0 / 3.6
            dv = (torch.rand(self.B, device=self.device, dtype=self.dtype) * 2.0 - 1.0) * dv_max
            v = self.vehicle.state[:, 3] + dv
            self.vehicle.state = self.vehicle.state.clone()
            self.vehicle.state[:, 3] = torch.clamp(v, min=0.0)

            dpsi_max = float(np.deg2rad(10.0))
            dpsi = (torch.rand(self.B, device=self.device, dtype=self.dtype) * 2.0 - 1.0) * dpsi_max
            self.e_psi_v = self.e_psi_v + dpsi

        self.step_count = torch.zeros(self.B, dtype=torch.int64, device=self.device)
        self._cache = {}

        env_state = self.get_env_state()
        obs_raw, state_vec, cache = self.compute_obs_state(env_state, return_cache=True)
        self._cache = cache

        obs_norm = self.normalize_obs(obs_raw, clip=5.0)
        return obs_norm, obs_raw, state_vec

    def step(
        self,
        action: torch.Tensor,
        compute_info: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[Dict[str, Any]]]:
        env_state = self.get_env_state()
        next_state, obs_norm, obs_raw, state_vec, reward, done, info = self.functional_step(
            env_state,
            action,
            compute_info=compute_info,
            normalize_obs=True,
            obs_clip=5.0,
        )
        # update stateful tensors
        self.set_env_state(next_state)
        # store cache for debug, not required for gradients
        return obs_norm, obs_raw, state_vec, reward, done, info

    # -----------------
    # observation normalization
    # -----------------

    def _init_obs_normalizer(self) -> None:
        p = self.vehicle.params
        device, dtype = self.device, self.dtype

        v_ref_min = float(self.v_ref_mat.min().item())
        v_ref_max = float(self.v_ref_mat.max().item())
        kappa_max = float(self.kappa_ref_mat.abs().max().item())

        ey_max = float(self.max_lateral_error)
        epsi_max = float(np.pi)
        a_min = float(getattr(p, "min_accel", -6.0))
        a_max = float(getattr(p, "max_accel", 3.0))
        delta_max = float(getattr(p, "max_steer", np.deg2rad(30)))

        v_min = 0.0
        v_max = max(1.0, 1.5 * v_ref_max)

        r_margin = 3.0
        r_max = max(0.5, r_margin * v_ref_max * kappa_max)

        P = int(self.P)
        obs_dim_expected = 6 + 5 + P

        obs_min = torch.empty(obs_dim_expected, device=device, dtype=dtype)
        obs_max = torch.empty(obs_dim_expected, device=device, dtype=dtype)

        i = 0
        obs_min[i], obs_max[i] = -ey_max, ey_max; i += 1
        obs_min[i], obs_max[i] = -epsi_max, epsi_max; i += 1
        obs_min[i], obs_max[i] = v_min, v_max; i += 1
        obs_min[i], obs_max[i] = a_min, a_max; i += 1
        obs_min[i], obs_max[i] = -r_max, r_max; i += 1
        obs_min[i], obs_max[i] = v_ref_min, v_ref_max; i += 1

        obs_min[i:i+5] = -delta_max
        obs_max[i:i+5] = delta_max
        i += 5

        obs_min[i:i+P] = -kappa_max
        obs_max[i:i+P] = kappa_max
        i += P

        if i != obs_dim_expected:
            raise RuntimeError(f"obs_dim mismatch in normalizer build: got {i}, expected {obs_dim_expected}")

        mid = 0.5 * (obs_max + obs_min)
        scale = 0.5 * (obs_max - obs_min)
        scale = torch.clamp(scale, min=1e-6)

        self._obs_norm_mid = mid
        self._obs_norm_scale = scale

    def normalize_obs(self, obs: torch.Tensor, clip: Optional[float] = 5.0) -> torch.Tensor:
        if not hasattr(self, "_obs_norm_mid") or not hasattr(self, "_obs_norm_scale"):
            self._init_obs_normalizer()

        if obs.size(-1) != self._obs_norm_mid.numel():
            raise ValueError(f"obs_dim mismatch: got {obs.size(-1)}, expected {self._obs_norm_mid.numel()}")

        mid = self._obs_norm_mid
        scale = self._obs_norm_scale

        obs_n = (obs - mid) / scale
        if clip is not None:
            obs_n = torch.clamp(obs_n, -float(clip), float(clip))
        return obs_n

    # -----------------
    # done mask / partial reset (PPO-like utilities)
    # -----------------

    @torch.no_grad()
    def get_done_mask(self) -> torch.Tensor:
        """Return current done mask (B,) based on internal state."""
        done_lateral = torch.abs(self.e_y) > self.max_lateral_error
        done_pathend = self.s >= self.s_end
        done_steps = self.step_count >= self.max_steps
        return done_lateral | done_pathend | done_steps

    
    def _regenerate_trajectories_for_indices(
        self,
        idx: torch.Tensor,
        init_state: torch.Tensor,
    ) -> None:
        """Regenerate reference trajectories for selected batch indices.

        Mirrors `environment_batched.py` behavior: replaces (v_ref_mat, kappa_ref_mat, psi_ref_mat)
        and updates `init_state` to match the new trajectory start pose/speed.
        """
        if self.traj_generator is None:
            return

        s_ref_base = self.s_ref.detach().cpu().numpy()

        for b in idx.tolist():
            traj = self.traj_generator()

            s_ref_new = np.asarray(traj.s_ref, dtype=np.float64)
            if s_ref_new.shape[0] != self.Ns or not np.allclose(s_ref_new, s_ref_base):
                raise ValueError(
                    f"traj_generator produced s_ref of shape {s_ref_new.shape}, "
                    "but env expects same s_ref as at init. "
                    "total_length と ds を固定しているか確認してください。"
                )

            v_ref = torch.tensor(
                np.asarray(traj.v_ref, dtype=np.float64).tolist(),
                dtype=self.dtype,
                device=self.device,
            )
            kappa_ref = torch.tensor(
                np.asarray(traj.kappa_ref, dtype=np.float64).tolist(),
                dtype=self.dtype,
                device=self.device,
            )
            self.v_ref_mat[b] = v_ref
            self.kappa_ref_mat[b] = kappa_ref

            x_ref = np.asarray(traj.x_ref, dtype=np.float64)
            y_ref = np.asarray(traj.y_ref, dtype=np.float64)
            dx = np.diff(x_ref)
            dy = np.diff(y_ref)
            psi = np.arctan2(dy, dx)
            if len(psi) > 0:
                psi = np.concatenate([psi, psi[-1:]])
            else:
                psi = np.array([0.0], dtype=np.float64)

            self.psi_ref_mat[b] = torch.tensor(psi.tolist(), dtype=self.dtype, device=self.device)

            # update init_state to match new trajectory start
            x0 = float(x_ref[0])
            y0 = float(y_ref[0])
            psi0 = float(psi[0])
            v0 = float(traj.v_ref[0])

            init_state[b, 0] = x0
            init_state[b, 1] = y0
            init_state[b, 2] = psi0
            init_state[b, 3] = v0
            init_state[b, 4:] = 0.0

        # trajectory swap => rebuild normalizer cache next time
        if hasattr(self, "_obs_norm_mid"):
            delattr(self, "_obs_norm_mid")
        if hasattr(self, "_obs_norm_scale"):
            delattr(self, "_obs_norm_scale")


    def functional_partial_reset(
        self,
        env_state: BatchedFrenetEnvState,
        init_state: torch.Tensor,
        done_mask: torch.Tensor,
        *,
        is_perturbed: bool = True,
        regenerate_traj: bool = False,
        normalize_obs: bool = True,
        obs_clip: float = 5.0,
    ) -> Tuple[BatchedFrenetEnvState, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Functional partial reset (works on an explicit env_state).

        Used during differentiable rollouts: replaces only done indices, returns updated env_state
        plus (obs_norm, obs_raw, state_vec).
        """
        device, dtype = self.device, self.dtype

        if done_mask.dim() == 2 and done_mask.shape[1] == 1:
            done_mask = done_mask.squeeze(1)
        if done_mask.dim() != 1 or done_mask.shape[0] != self.B:
            raise ValueError(f"done_mask must be (B,) or (B,1), got {tuple(done_mask.shape)} with B={self.B}")
        done_mask = done_mask.to(device=device)

        idx = torch.nonzero(done_mask, as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            obs_raw, state_vec, cache = self.compute_obs_state(env_state, return_cache=True)
            self._cache = cache
            obs_norm = self.normalize_obs(obs_raw, clip=float(obs_clip)) if normalize_obs else obs_raw
            return env_state, obs_norm, obs_raw, state_vec

        self._regenerate_trajectories_for_indices(idx, init_state)

        init_state_v = self._ensure_vehicle_init_state(init_state)

        vehicle_state = env_state.vehicle_state.clone()
        vehicle_state[idx] = init_state_v[idx].to(device=device, dtype=dtype)

        s = env_state.s.clone()
        s0_val = float(self.s_ref[0])
        s[idx] = s0_val

        e_y = env_state.e_y.clone()
        e_psi_v = env_state.e_psi_v.clone()
        e_y[idx] = 0.0
        e_psi_v[idx] = 0.0

        step_count = env_state.step_count.clone()
        step_count[idx] = 0

        delta_ref_history = env_state.delta_ref_history.clone()
        delta_ref_history[idx] = 0.0

        if is_perturbed:
            e_y[idx] += (torch.rand(idx.numel(), device=device) * 2.0 - 1.0)

            dv_max = 10.0 / 3.6
            dv = (torch.rand(idx.numel(), device=device) * 2.0 - 1.0) * dv_max
            v = vehicle_state[idx, 3] + dv
            vehicle_state[idx, 3] = torch.clamp(v, min=0.0)

            dpsi_max = torch.tensor(10.0 / 180.0 * 3.141592653589793, device=device)
            dpsi = (torch.rand(idx.numel(), device=device) * 2.0 - 1.0) * dpsi_max
            e_psi_v[idx] += dpsi

        next_env_state = BatchedFrenetEnvState(
            vehicle_state=vehicle_state,
            s=s,
            e_y=e_y,
            e_psi_v=e_psi_v,
            step_count=step_count,
            delta_ref_history=delta_ref_history,
        )

        obs_raw, state_vec, cache = self.compute_obs_state(next_env_state, return_cache=True)
        self._cache = cache
        obs_norm = self.normalize_obs(obs_raw, clip=float(obs_clip)) if normalize_obs else obs_raw
        return next_env_state, obs_norm, obs_raw, state_vec


    @torch.no_grad()
    def partial_reset(
        self,
        init_state: torch.Tensor,
        done_mask: torch.Tensor,
        is_perturbed: bool = False,
        regenerate_traj: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Stateful partial reset (PPO-like utility)."""
        device, dtype = self.device, self.dtype
        done_mask = done_mask.to(device=device)

        idx = torch.nonzero(done_mask, as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            env_state = self.get_env_state()
            obs_raw, state_vec, cache = self.compute_obs_state(env_state, return_cache=True)
            self._cache = cache
            obs_norm = self.normalize_obs(obs_raw, clip=5.0)
            return obs_norm, obs_raw, state_vec

        if regenerate_traj and self.traj_generator is not None:
            self._regenerate_trajectories_for_indices(idx, init_state)

        init_state_v = self._ensure_vehicle_init_state(init_state)
        self.vehicle.state[idx] = init_state_v[idx].to(device=device, dtype=dtype)

        s0_val = float(self.s_ref[0])
        self.s[idx] = s0_val

        self.e_y[idx] = 0.0
        self.e_psi_v[idx] = 0.0
        self.step_count[idx] = 0

        self.delta_ref_history[idx] = 0.0

        if is_perturbed:
            self.e_y[idx] += (torch.rand(idx.numel(), device=device) * 2.0 - 1.0)

            dv_max = 10.0 / 3.6
            dv = (torch.rand(idx.numel(), device=device) * 2.0 - 1.0) * dv_max
            v = self.vehicle.state[idx, 3] + dv
            self.vehicle.state[idx, 3] = torch.clamp(v, min=0.0)

            dpsi_max = torch.tensor(10.0 / 180.0 * 3.141592653589793, device=device)
            dpsi = (torch.rand(idx.numel(), device=device) * 2.0 - 1.0) * dpsi_max
            self.e_psi_v[idx] += dpsi

        env_state = self.get_env_state()
        obs_raw, state_vec, cache = self.compute_obs_state(env_state, return_cache=True)
        self._cache = cache
        obs_norm = self.normalize_obs(obs_raw, clip=5.0)
        return obs_norm, obs_raw, state_vec
