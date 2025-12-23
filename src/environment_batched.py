import numpy as np
import torch
from dataclasses import dataclass
from typing import Optional, Sequence, List, Dict, Any, Callable

from bicycle_model_batched import BatchedDynamicBicycleModel, VehicleParams
from trajectory_generator import ReferenceTrajectory


# =========================
#  報酬重み
# =========================

@dataclass
class RewardWeights:
    w_y: float = 0.1         # 横偏差
    w_psi: float = 10.0      # 速度ベクトル方向の偏差
    w_v_under: float = 0.1   # 目標より遅いとき
    w_v_over: float = 1.5    # 目標より速いとき
    w_ay: float = 0.001      # 横加速度
    w_d_delta_ref: float = 0.1  # ステア指令レート d_delta_ref の L2 ペナルティ
    # 損失タイプ ("l1" or "l2")
    loss_y: str = "l2"
    loss_psi: str = "l2"
    loss_v: str = "l2"
    loss_ay: str = "l2"
    loss_d_delta_ref: str = "l2"


# obs の列順（参考）
OBS_KEYS = [
    "e_y",
    "e_psi_v",
    "v",
    "a",
    "r",
    "v_ref",
    "delta_ref_hist_0",  # Current
    "delta_ref_hist_1",  # -1 step
    "delta_ref_hist_2",  # -2 step
    "delta_ref_hist_3",  # -3 step
    "delta_ref_hist_4",  # -4 step
    # kappa_preview x 21
]


class BatchedPathTrackingEnvFrenet:
    """
    Frenet（経路座標）系でバッチ並列に動く経路追従環境。
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
    ):
        self.device = device
        self.dtype = dtype

        self.traj_generator = traj_generator

        self.B = len(ref_trajs)
        assert self.B > 0, "ref_trajs must be non-empty"

        # ---------- s_ref はバッチ内で完全に同一と仮定 ----------
        s_ref0 = np.asarray(ref_trajs[0].s_ref, dtype=np.float64)
        N = s_ref0.shape[0]
        assert N >= 2, "s_ref must have at least 2 points"

        # 他の traj の s_ref が同一か確認
        for i, traj in enumerate(ref_trajs[1:], start=1):
            s_ref_i = np.asarray(traj.s_ref, dtype=np.float64)
            if not np.allclose(s_ref_i, s_ref0):
                raise ValueError(f"s_ref of trajectory {i} differs from trajectory 0")

        # ★ NumPy → list 経由で torch に変換（from_numpy を使わない）
        self.s_ref = torch.tensor(
            s_ref0.tolist(),
            dtype=self.dtype,
            device=self.device,
        )  # (N,)
        self.Ns = N
        self.s_end = torch.tensor(
            float(s_ref0[-1]),
            dtype=self.dtype,
            device=self.device,
        )

        self.dt = ref_trajs[0].dt

        # ---------- v_ref, kappa_ref, psi_ref を (B,N) にまとめる ----------
        self.v_ref_mat = torch.zeros(self.B, N, dtype=self.dtype, device=self.device)
        self.kappa_ref_mat = torch.zeros(self.B, N, dtype=self.dtype, device=self.device)
        self.psi_ref_mat = torch.zeros(self.B, N, dtype=self.dtype, device=self.device)

        for b, traj in enumerate(ref_trajs):
            x_ref = np.asarray(traj.x_ref, dtype=np.float64)
            y_ref = np.asarray(traj.y_ref, dtype=np.float64)
            v_ref = np.asarray(traj.v_ref, dtype=np.float64)
            kappa_ref = np.asarray(traj.kappa_ref, dtype=np.float64)

            assert len(x_ref) == N and len(y_ref) == N \
                   and len(v_ref) == N and len(kappa_ref) == N, \
                "All reference arrays must have same length as s_ref"

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

        # プレビュー距離 (kappa用)
        if kappa_preview_offsets is None:
            self.kappa_preview_offsets = torch.arange(21, device=self.device, dtype=self.dtype) * 1.0
        else:
            self.kappa_preview_offsets = torch.tensor(
                kappa_preview_offsets,
                dtype=self.dtype,
                device=self.device,
            )
        self.P = self.kappa_preview_offsets.shape[0]

        # delta_ref history buffer: (B, 5)
        self.delta_ref_history = torch.zeros(self.B, 5, dtype=self.dtype, device=self.device)

        self.vehicle = BatchedDynamicBicycleModel(
            params=vehicle_params,
            device=self.device,
            dtype=self.dtype,
        )

        self.weights = reward_weights if reward_weights is not None else RewardWeights()
        self.max_lateral_error = max_lateral_error
        self.max_steps = max_steps

        self.s = torch.zeros(self.B, dtype=self.dtype, device=self.device)
        self.e_y = torch.zeros(self.B, dtype=self.dtype, device=self.device)
        self.e_psi_v = torch.zeros(self.B, dtype=self.dtype, device=self.device)
        self.step_count = torch.zeros(self.B, dtype=torch.int64, device=self.device)

        self._cache: Dict[str, torch.Tensor] = {}

    def _penalty(self, x: torch.Tensor, loss_type: str) -> torch.Tensor:
        if loss_type == "l1":
            return torch.abs(x)
        if loss_type == "l2":
            return x ** 2
        raise ValueError(f"Unknown loss_type: {loss_type} (expected 'l1' or 'l2')")

    # ---------- vehicle init_state 互換 (B,8)/(B,9) ----------

    def _ensure_vehicle_init_state(self, init_state: torch.Tensor) -> torch.Tensor:
        assert init_state.dim() == 2 and init_state.shape[0] == self.B, \
            f"init_state must be (B,8) or (B,9) with B={self.B}, got {init_state.shape}"
        assert init_state.shape[1] in (8, 9), \
            f"init_state last dim must be 8 or 9, got {init_state.shape[1]}"

        init_state = init_state.to(device=self.device, dtype=self.dtype)
        if init_state.shape[1] == 9:
            return init_state
        delta = init_state[:, 5:6]
        return torch.cat([init_state, delta], dim=1)

    def _regenerate_trajectories_for_indices(
        self,
        idx: torch.Tensor,
        init_state: torch.Tensor,
    ):
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

            x0 = float(x_ref[0])
            y0 = float(y_ref[0])
            psi0 = float(psi[0])
            v0 = float(traj.v_ref[0])

            init_state[b, 0] = x0
            init_state[b, 1] = y0
            init_state[b, 2] = psi0
            init_state[b, 3] = v0
            init_state[b, 4:] = 0.0

            # traj差し替え後：normalize用キャッシュを作り直す
            if hasattr(self, "_obs_norm_mid"):
                delattr(self, "_obs_norm_mid")
            if hasattr(self, "_obs_norm_scale"):
                delattr(self, "_obs_norm_scale")

    # ---------- ユーティリティ ----------

    def _wrap_angle(self, ang: torch.Tensor) -> torch.Tensor:
        pi = np.pi
        return (ang + pi) % (2.0 * pi) - pi

    def _interp_matrix(self, s: torch.Tensor, y_mat: torch.Tensor) -> torch.Tensor:
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

        batch_flat = (
            torch.arange(B, device=self.device)
            .unsqueeze(1)
            .expand(B, P)
            .reshape(-1)
        )

        y0_q = y_mat[batch_flat, idx_q]
        y1_q = y_mat[batch_flat, idx_q + 1]
        y_flat = (1.0 - t_q) * y0_q + t_q * y1_q

        return y_flat.view(B, P)

    # ---------- reset ----------

    def reset(
        self,
        init_state: torch.Tensor,
        s0: Optional[torch.Tensor] = None,
        is_perturbed: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        init_state_v = self._ensure_vehicle_init_state(init_state)
        self.vehicle.reset(init_state_v)

        self.delta_ref_history.zero_()

        if s0 is None:
            self.s = torch.full((self.B,), float(self.s_ref[0]), dtype=self.dtype, device=self.device)
        else:
            assert s0.shape[0] == self.B
            self.s = s0.to(device=self.device, dtype=self.dtype)

        self.e_y.zero_()
        self.e_psi_v.zero_()

        if is_perturbed:
            self.e_y += (torch.rand(self.B, device=self.device) * 2.0 - 1.0)

            dv_max = 10.0 / 3.6
            dv = (torch.rand(self.B, device=self.device) * 2.0 - 1.0) * dv_max
            v = self.vehicle.state[:, 3] + dv
            self.vehicle.state[:, 3] = torch.clamp(v, min=0.0)

            dpsi_max = np.deg2rad(10.0)
            dpsi = (torch.rand(self.B, device=self.device) * 2.0 - 1.0) * dpsi_max
            self.e_psi_v += dpsi

        self.step_count.zero_()

        obs_raw, state = self._compute_obs_state()
        obs_norm = self.normalize_obs(obs_raw, clip=5.0)
        return obs_norm, obs_raw, state

    def partial_reset(
        self,
        init_state: torch.Tensor,
        done_mask: torch.Tensor,
        is_perturbed: bool = True,
        regenerate_traj: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device, dtype = self.device, self.dtype
        done_mask = done_mask.to(device=device)

        idx = torch.nonzero(done_mask, as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            obs_raw, state = self._compute_obs_state()
            obs_norm = self.normalize_obs(obs_raw, clip=5.0)
            return obs_norm, obs_raw, state

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

        obs_raw, state = self._compute_obs_state()
        obs_norm = self.normalize_obs(obs_raw, clip=5.0)
        return obs_norm, obs_raw, state

    # ---------- 観測（obs/state） ----------

    def _compute_obs_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        v = self.vehicle.state[:, 3]
        a = self.vehicle.state[:, 4]
        delta = self.vehicle.state[:, 5]
        beta = self.vehicle.state[:, 6]
        r = self.vehicle.state[:, 7]
        delta_ref = self.vehicle.state[:, 8]

        v_ref_now = self._interp_matrix(self.s, self.v_ref_mat)
        kappa_preview = self._interp_matrix_preview(self.s, self.kappa_ref_mat)

        if float(self.kappa_preview_offsets[0].item()) == 0.0:
            kappa0 = kappa_preview[:, 0]
        else:
            kappa0 = self._interp_matrix(self.s, self.kappa_ref_mat)

        e_y = self.e_y
        e_psi_v = self._wrap_angle(self.e_psi_v)

        obs_list = [
            e_y,
            e_psi_v,
            v,
            a,
            r,
            v_ref_now,
            self.delta_ref_history,  # (B, 5)
            kappa_preview,           # (B, P)
        ]
        obs = torch.cat([x if x.dim() == 2 else x.unsqueeze(1) for x in obs_list], dim=1)

        state = torch.stack([e_y, e_psi_v, v, a, delta, delta_ref, beta, r], dim=1)

        self._cache = {
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

        return obs, state

    # ---------- 報酬・終了判定（reward/done/info） ----------

    def _compute_reward_done_info(
        self,
        action: torch.Tensor,
        compute_info: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, Any]]]:
        B = self.B
        w = self.weights
        p = self.vehicle.params

        if not self._cache:
            raise RuntimeError("_compute_obs_state() must be called before _compute_reward_done_info().")

        v = self._cache["v"]
        e_y = self._cache["e_y"]
        e_psi_v = self._cache["e_psi_v"]
        v_ref_now = self._cache["v_ref_now"]
        delta_ref = self._cache["delta_ref"]
        beta = self._cache["beta"]

        if "kappa0" in self._cache:
            kappa0 = self._cache["kappa0"]
        else:
            kappa0 = self._interp_matrix(self.s, self.kappa_ref_mat)

        delta_geom = self.vehicle.steering_from_kappa(kappa0)

        cost_y = w.w_y * self._penalty(e_y, w.loss_y)

        e_psi_vel = self._wrap_angle(e_psi_v + beta)
        cost_psi = w.w_psi * self._penalty(e_psi_vel, w.loss_psi)

        dv = v - v_ref_now
        dv_pen = self._penalty(dv, w.loss_v)
        cost_v_under = w.w_v_under * dv_pen
        cost_v_over = w.w_v_over * dv_pen
        cost_v = torch.where(v <= v_ref_now, cost_v_under, cost_v_over)

        F_yf = self.vehicle.F_yf
        F_yr = self.vehicle.F_yr
        a_y = (F_yf + F_yr) / p.m
        cost_ay = w.w_ay * self._penalty(a_y, w.loss_ay)

        if "d_delta_ref" in self._cache:
            d_delta_ref_val = self._cache["d_delta_ref"]
        else:
            d_delta_ref_val = torch.zeros(B, device=self.device, dtype=self.dtype)

        d_delta_ref_applied = d_delta_ref_val
        cost_d_delta_ref = w.w_d_delta_ref * self._penalty(d_delta_ref_applied, w.loss_d_delta_ref)

        total_cost = cost_y + cost_psi + cost_v + cost_ay + cost_d_delta_ref
        reward = -total_cost

        done_lateral = torch.abs(e_y) > self.max_lateral_error
        done_pathend = self.s >= self.s_end
        done_steps = self.step_count >= self.max_steps
        done = done_lateral | done_pathend | done_steps

        info: Optional[Dict[str, Any]]
        if compute_info:
            info = {
                "cost_y": cost_y,
                "cost_psi": cost_psi,
                "cost_v": cost_v,
                "cost_ay": cost_ay,
                "cost_d_delta_ref": cost_d_delta_ref,
                "a_y": a_y,
                "v_ref": v_ref_now,
                "s": self.s.data.clone(),
                "kappa0": kappa0,
                "delta_geom": delta_geom,
                "d_delta_ref": d_delta_ref_applied,
                "delta_ref": delta_ref,
                "done_lateral": done_lateral,
                "done_pathend": done_pathend,
                "done_steps": done_steps,
            }
        else:
            info = None

        return reward, done, info

    # ---------- step ----------

    @torch.no_grad()
    def step(
        self,
        action: torch.Tensor,
        compute_info: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[Dict[str, Any]]]:
        assert action.shape[0] == self.B and action.shape[1] == 2, \
            f"action must be (B,2) with B={self.B}, got {action.shape}"

        delta_ref_old = self.vehicle.state[:, 8].clone()

        self.vehicle.step(action, self.dt)

        self.delta_ref_history[:, 1:] = self.delta_ref_history[:, :-1].clone()
        self.delta_ref_history[:, 0] = self.vehicle.state[:, 8]

        v = self.vehicle.state[:, 3]
        r = self.vehicle.state[:, 7]

        kappa0 = self._interp_matrix(self.s, self.kappa_ref_mat)

        denom = 1.0 - kappa0 * self.e_y
        denom = torch.clamp(denom, min=0.1)

        s_dot = v * torch.cos(self.e_psi_v) / denom
        e_y_dot = v * torch.sin(self.e_psi_v)
        e_psi_v_dot = r - kappa0 * s_dot

        dt_t = torch.as_tensor(self.dt, dtype=self.dtype, device=self.device)
        self.s = self.s + s_dot * dt_t
        self.e_y = self.e_y + e_y_dot * dt_t
        self.e_psi_v = self._wrap_angle(self.e_psi_v + e_psi_v_dot * dt_t)

        self.step_count += 1

        obs_raw, state = self._compute_obs_state()

        p = self.vehicle.params
        delta_ref_new = torch.clamp(action[:, 1], -p.max_steer, p.max_steer)
        d_delta_ref_val = (delta_ref_new - delta_ref_old) / self.dt
        self._cache["d_delta_ref"] = d_delta_ref_val

        reward, done, info = self._compute_reward_done_info(
            action=action,
            compute_info=compute_info,
        )

        if not torch.isfinite(self.vehicle.state).all():
            print("Non-finite in vehicle.state", "maxabs:", self.vehicle.state.abs().max().item())
            raise RuntimeError("Inf/NaN in vehicle.state")

        if torch.isnan(self.vehicle.state).any():
            print("NaN in vehicle.state, step_count:", self.step_count.max().item())
            raise RuntimeError("NaN in vehicle.state")

        if torch.isnan(self.s).any() or torch.isnan(self.e_y).any() or torch.isnan(self.e_psi_v).any():
            print("NaN in Frenet state")
            raise RuntimeError("NaN in Frenet state")

        obs_norm = self.normalize_obs(obs_raw, clip=5.0)

        return obs_norm, obs_raw, state, reward, done, info

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

        assert i == obs_dim_expected, (i, obs_dim_expected)

        mid = 0.5 * (obs_max + obs_min)
        scale = 0.5 * (obs_max - obs_min)
        scale = torch.clamp(scale, min=1e-6)

        self._obs_norm_mid = mid
        self._obs_norm_scale = scale

    def normalize_obs(self, obs: torch.Tensor, clip: float = 5.0) -> torch.Tensor:
        if not hasattr(self, "_obs_norm_mid") or not hasattr(self, "_obs_norm_scale"):
            self._init_obs_normalizer()

        assert obs.size(-1) == self._obs_norm_mid.numel(), \
            f"obs_dim mismatch: got {obs.size(-1)}, expected {self._obs_norm_mid.numel()}"

        mid = self._obs_norm_mid
        scale = self._obs_norm_scale

        obs_n = (obs - mid) / scale

        if clip is not None:
            obs_n = torch.clamp(obs_n, -float(clip), float(clip))

        return obs_n
