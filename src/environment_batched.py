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
    w_y: float = 0.1        # 横偏差
    w_psi: float = 10.0      # 速度ベクトル方向の偏差
    w_v_under: float = 0.1  # 目標より遅いとき
    w_v_over: float = 1.5   # 目標より速いとき
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
    "delta_ref",
    "r",
    "s",
    "v_ref",
    "kappa_0",
    "kappa_preview_1",
    "kappa_preview_2",
    "kappa_preview_3",
]


class BatchedPathTrackingEnvFrenet:
    """
    Frenet（経路座標）系でバッチ並列に動く経路追従環境。

    - 内部に BatchedDynamicBicycleModel を持つ
    - 各バッチごとに別々の ReferenceTrajectory を持つ（ただし s_ref は共通）
    - 状態:
        s         : 経路に沿った進行距離 [m]
        e_y       : 経路に対する横偏差（左を＋）[m]
        e_psi_v   : 速度ベクトルと経路方向の角度偏差 [rad]
        v, a, delta, beta, r : 車両ダイナミクス（bicycle_model 内の state）
    - step 入力:
        action: (B, 2) torch.Tensor [a_ref, d_delta_ref]
    - step 出力:
        obs   : (B, obs_dim) torch.Tensor
        state : (B, state_dim) torch.Tensor
        reward: (B,) torch.Tensor
        done  : (B,) torch.BoolTensor
        info  : None または dict[str, torch.Tensor]
    """

    def __init__(
        self,
        ref_trajs: List[ReferenceTrajectory],
        preview_distances: Sequence[float] = (10.0, 20.0, 30.0),
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
        # ここも torch.zeros から埋めていく（NumPy 行列→from_numpy は使わない）
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

            # v_ref, kappa_ref を torch にコピー
            self.v_ref_mat[b] = torch.tensor(
                v_ref.tolist(), dtype=self.dtype, device=self.device
            )
            self.kappa_ref_mat[b] = torch.tensor(
                kappa_ref.tolist(), dtype=self.dtype, device=self.device
            )

            # psi_ref を計算
            dx = np.diff(x_ref)
            dy = np.diff(y_ref)
            psi = np.arctan2(dy, dx)
            if len(psi) > 0:
                psi = np.concatenate([psi, psi[-1:]])
            else:
                psi = np.array([0.0], dtype=np.float64)

            self.psi_ref_mat[b] = torch.tensor(
                psi.tolist(), dtype=self.dtype, device=self.device
            )

        # プレビュー距離も同様に
        self.preview_distances = np.asarray(preview_distances, dtype=np.float64)
        self.preview_dists = torch.tensor(
            self.preview_distances.tolist(),
            dtype=self.dtype,
            device=self.device,
        )
        self.P = self.preview_dists.shape[0]
        assert self.P == 3, "今の OBS_KEYS は preview 3点前提になっています"

        # （以下は元のコードと同じ）
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

        # 直近に計算した obs/state と、その計算に必要な中間量をキャッシュ
        # _compute_obs_state() が更新し、_compute_reward_done_info() が参照する。
        self._cache: Dict[str, torch.Tensor] = {}

    def _penalty(self, x: torch.Tensor, loss_type: str) -> torch.Tensor:
        """loss_type に応じて L1/L2 を返す（要素ごと）"""
        if loss_type == "l1":
            return torch.abs(x)
        if loss_type == "l2":
            return x ** 2
        raise ValueError(f"Unknown loss_type: {loss_type} (expected 'l1' or 'l2')")


    # ---------- vehicle init_state 互換 (B,8)/(B,9) ----------

    def _ensure_vehicle_init_state(self, init_state: torch.Tensor) -> torch.Tensor:
        """(B,8)=[x,y,psi,v,a,delta,beta,r] を (B,9) に拡張して返す。

        新しい車両モデルは state に delta_ref を保持するため (B,9) が正。
        (B,8) の場合は delta_ref=delta で初期化する。
        """
        assert init_state.dim() == 2 and init_state.shape[0] == self.B, \
            f"init_state must be (B,8) or (B,9) with B={self.B}, got {init_state.shape}"
        assert init_state.shape[1] in (8, 9), \
            f"init_state last dim must be 8 or 9, got {init_state.shape[1]}"

        init_state = init_state.to(device=self.device, dtype=self.dtype)
        if init_state.shape[1] == 9:
            return init_state
        # (B,8) -> (B,9) by appending delta_ref=delta
        delta = init_state[:, 5:6]
        return torch.cat([init_state, delta], dim=1)

    def _regenerate_trajectories_for_indices(
        self,
        idx: torch.Tensor,
        init_state: torch.Tensor,
    ):
        """
        done になった env の index 群 idx について:
          - self.traj_generator() で新しい ReferenceTrajectory を生成
          - v_ref_mat, kappa_ref_mat, psi_ref_mat の該当行を差し替え
          - init_state[b] の初期姿勢 (x,y,psi,v) を更新
        前提:
          - traj_generator が None でない
          - 生成される s_ref が self.s_ref と同一（長さも中身も）
        """
        if self.traj_generator is None:
            return  # 何もしない

        # s_ref を numpy にして比較用に保持
        s_ref_base = self.s_ref.detach().cpu().numpy()

        for b in idx.tolist():
            traj = self.traj_generator()  # 新しい ReferenceTrajectory

            # s_ref の長さと値が一致しているか確認
            s_ref_new = np.asarray(traj.s_ref, dtype=np.float64)
            if s_ref_new.shape[0] != self.Ns or not np.allclose(s_ref_new, s_ref_base):
                raise ValueError(
                    f"traj_generator produced s_ref of shape {s_ref_new.shape}, "
                    "but env expects same s_ref as at init. "
                    "total_length と ds を固定しているか確認してください。"
                )

            # --- v_ref, kappa_ref の更新 (B,N) の b 行目を書き換える ---
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

            # --- psi_ref の更新 (x,y から計算) ---
            x_ref = np.asarray(traj.x_ref, dtype=np.float64)
            y_ref = np.asarray(traj.y_ref, dtype=np.float64)
            dx = np.diff(x_ref)
            dy = np.diff(y_ref)
            psi = np.arctan2(dy, dx)
            if len(psi) > 0:
                psi = np.concatenate([psi, psi[-1:]])
            else:
                psi = np.array([0.0], dtype=np.float64)

            self.psi_ref_mat[b] = torch.tensor(
                psi.tolist(), dtype=self.dtype, device=self.device
            )

            # --- 初期状態テンプレート init_state[b] を更新 ---
            x0 = float(x_ref[0])
            y0 = float(y_ref[0])
            psi0 = float(psi[0])
            v0 = float(traj.v_ref[0])

            init_state[b, 0] = x0
            init_state[b, 1] = y0
            init_state[b, 2] = psi0
            init_state[b, 3] = v0
            # a, delta, beta, r (and delta_ref if present) は 0 に戻す
            init_state[b, 4:] = 0.0

    # ---------- ユーティリティ ----------

    def _wrap_angle(self, ang: torch.Tensor) -> torch.Tensor:
        """角度を [-pi, pi] に wrap（要素ごと）"""
        pi = np.pi
        return (ang + pi) % (2.0 * pi) - pi

    def _interp_matrix(self, s: torch.Tensor, y_mat: torch.Tensor) -> torch.Tensor:
        """
        s: (B,)   - 各バッチの s（連続値）
        y_mat: (B, N) - 各バッチ × s_ref の値
        戻り値: (B,)  - 線形補間した値
        """
        B = self.B
        N = self.Ns

        # s を範囲内にクリップ
        s_clamped = torch.clamp(s, self.s_ref[0], self.s_ref[-1] - 1e-6)  # (B,)

        # searchsorted で各 s が属する区間の左端インデックスを求める
        idx = torch.searchsorted(self.s_ref, s_clamped, right=False)  # (B,)
        idx = torch.clamp(idx, 0, N - 2)  # 区間 [idx, idx+1]

        s0 = self.s_ref[idx]       # (B,)
        s1 = self.s_ref[idx + 1]   # (B,)
        t = (s_clamped - s0) / (s1 - s0 + 1e-8)  # (B,)

        batch_idx = torch.arange(B, device=self.device)

        y0 = y_mat[batch_idx, idx]       # (B,)
        y1 = y_mat[batch_idx, idx + 1]   # (B,)

        return (1.0 - t) * y0 + t * y1   # (B,)

    def _interp_matrix_preview(self, s: torch.Tensor, y_mat: torch.Tensor) -> torch.Tensor:
        """
        s: (B,)
        y_mat: (B, N)
        戻り値: (B, P)  各バッチ×各プレビュー距離に対する線形補間値
        """
        B = self.B
        N = self.Ns
        P = self.P

        # s_q: (B, P)
        s_q = s.unsqueeze(1) + self.preview_dists.unsqueeze(0)
        s_q = torch.clamp(s_q, self.s_ref[0], self.s_ref[-1] - 1e-6)

        # flatten して searchsorted
        s_q_flat = s_q.reshape(-1)  # (B*P,)
        idx_q = torch.searchsorted(self.s_ref, s_q_flat, right=False)  # (B*P,)
        idx_q = torch.clamp(idx_q, 0, N - 2)

        s0_q = self.s_ref[idx_q]        # (B*P,)
        s1_q = self.s_ref[idx_q + 1]
        t_q = (s_q_flat - s0_q) / (s1_q - s0_q + 1e-8)

        # バッチインデックスを flatten
        batch_flat = (
            torch.arange(B, device=self.device)
            .unsqueeze(1)
            .expand(B, P)
            .reshape(-1)
        )  # (B*P,)

        y0_q = y_mat[batch_flat, idx_q]
        y1_q = y_mat[batch_flat, idx_q + 1]
        y_flat = (1.0 - t_q) * y0_q + t_q * y1_q  # (B*P,)

        return y_flat.view(B, P)  # (B, P)

    # ---------- reset ----------

    def reset(
        self,
        init_state: torch.Tensor,
        s0: Optional[torch.Tensor] = None,
        is_perturbed: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        バッチ全体をリセット。

        Args:
            init_state: (B, 8) または (B, 9)
                (B, 8) = [x, y, psi, v, a, delta, beta, r]
                (B, 9) = [x, y, psi, v, a, delta, beta, r, delta_ref]
            s0: (B,) 経路上の初期位置。None の場合は全バッチ s_ref[0]
            is_perturbed: True なら e_y, v, e_psi_v にランダムノイズを加える。

        Returns:
            obs:   (B, obs_dim) Tensor
            state: (B, state_dim) Tensor
        """
        # 車両状態リセット（(B,8)/(B,9) 互換）
        init_state_v = self._ensure_vehicle_init_state(init_state)
        self.vehicle.reset(init_state_v)

        # s0 初期化
        if s0 is None:
            self.s = torch.full(
                (self.B,), float(self.s_ref[0]), dtype=self.dtype, device=self.device
            )
        else:
            assert s0.shape[0] == self.B
            self.s = s0.to(device=self.device, dtype=self.dtype)

        # e_y, e_psi_v 初期化
        self.e_y.zero_()
        self.e_psi_v.zero_()

        # ノイズ付与
        if is_perturbed:
            # 横偏差ノイズ ±1 m
            self.e_y += (torch.rand(self.B, device=self.device) * 2.0 - 1.0)

            # 速度ノイズ ±10 km/h
            dv_max = 10.0 / 3.6
            dv = (torch.rand(self.B, device=self.device) * 2.0 - 1.0) * dv_max
            v = self.vehicle.state[:, 3] + dv
            self.vehicle.state[:, 3] = torch.clamp(v, min=0.0)

            # 速度方向偏差ノイズ ±10 deg
            dpsi_max = np.deg2rad(10.0)
            dpsi = (torch.rand(self.B, device=self.device) * 2.0 - 1.0) * dpsi_max
            self.e_psi_v += dpsi

        self.step_count.zero_()

        obs, state = self._compute_obs_state()
        return obs, state
    
    def partial_reset(
        self,
        init_state: torch.Tensor,
        done_mask: torch.Tensor,
        is_perturbed: bool = True,
        regenerate_traj: bool = False,
    ):
        """
        done_mask が True のバッチだけ reset する。

        Args:
            init_state: (B, 8) 初期状態テンプレート（外側で保持している Tensor）
                        regenerate_traj=True の場合、この中身も更新される
            done_mask: (B,) bool テンソル
            is_perturbed: True のとき、e_y, v, e_psi_v にノイズ付与
            regenerate_traj: True のとき、traj_generator を用いて
                             対応する env の参照経路も振り直す
        Returns:
            obs:   (B, obs_dim) Tensor
            state: (B, state_dim) Tensor
        """
        device, dtype = self.device, self.dtype
        done_mask = done_mask.to(device=device)

        # どの index をリセットするか
        idx = torch.nonzero(done_mask, as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            # 何もリセットしない場合でも、最新の obs を返せるようにしておく
            obs, state = self._compute_obs_state()
            return obs, state

        # --- 必要なら traj を振り直す ---
        if regenerate_traj and self.traj_generator is not None:
            self._regenerate_trajectories_for_indices(idx, init_state)

        # --- vehicle.state を初期値に戻す ---
        init_state_v = self._ensure_vehicle_init_state(init_state)
        self.vehicle.state[idx] = init_state_v[idx].to(device=device, dtype=dtype)

        # --- Frenet 状態を初期化 ---
        # s は共通 s_ref の先頭値
        s0_val = float(self.s_ref[0])
        self.s[idx] = s0_val

        # e_y, e_psi_v, step_count をゼロ
        self.e_y[idx] = 0.0
        self.e_psi_v[idx] = 0.0
        self.step_count[idx] = 0

        # ノイズを入れるならここで
        if is_perturbed:
            # 横偏差ノイズ ±1m
            self.e_y[idx] += (torch.rand(idx.numel(), device=device) * 2.0 - 1.0)

            # 速度ノイズ ±10km/h
            dv_max = 10.0 / 3.6
            dv = (torch.rand(idx.numel(), device=device) * 2.0 - 1.0) * dv_max
            v = self.vehicle.state[idx, 3] + dv
            self.vehicle.state[idx, 3] = torch.clamp(v, min=0.0)

            # 速度方向偏差ノイズ ±10deg
            dpsi_max = torch.tensor(10.0 / 180.0 * 3.141592653589793, device=device)
            dpsi = (torch.rand(idx.numel(), device=device) * 2.0 - 1.0) * dpsi_max
            self.e_psi_v[idx] += dpsi

        # リセット後の obs/state を返す
        obs, state = self._compute_obs_state()
        return obs, state


    # ---------- 観測（obs/state） ----------

    def _compute_obs_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Frenet 状態と車両状態から obs, state を計算し、必要な中間量をキャッシュする。

        呼び出し順序の前提:
          1) _compute_obs_state() を呼ぶ（キャッシュを更新）
          2) _compute_reward_done_info(action, ...) を呼ぶ（キャッシュを参照）
        """
        # 車両状態を取り出し
        v = self.vehicle.state[:, 3]
        a = self.vehicle.state[:, 4]
        delta = self.vehicle.state[:, 5]
        beta = self.vehicle.state[:, 6]
        r = self.vehicle.state[:, 7]
        delta_ref = self.vehicle.state[:, 8]

        # 経路情報をバッチで補間
        v_ref_now = self._interp_matrix(self.s, self.v_ref_mat)
        kappa0 = self._interp_matrix(self.s, self.kappa_ref_mat)
        kappa_preview = self._interp_matrix_preview(self.s, self.kappa_ref_mat)

        # 速度ベクトル方向の偏差は e_psi_v として状態で保持
        e_y = self.e_y
        e_psi_v = self._wrap_angle(self.e_psi_v)

        # ---- obs / state ----
        # obs: 経路情報を含む（delta ではなく delta_ref を観測に出す）
        obs = torch.stack(
            [
                e_y,
                e_psi_v,
                v,
                a,
                delta_ref,
                r,
                self.s,
                v_ref_now,
                kappa0,
                kappa_preview[:, 0],
                kappa_preview[:, 1],
                kappa_preview[:, 2],
            ],
            dim=1,
        )  # (B, 12)

        # state: 経路情報は落として beta を入れ、delta_ref も含める
        state = torch.stack(
            [e_y, e_psi_v, v, a, delta, delta_ref, beta, r], dim=1
        )  # (B, 8)

        # ---- cache ----
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
        """直近の _compute_obs_state() が作ったキャッシュを使って reward/done/info を計算する。"""
        B = self.B
        w = self.weights
        p = self.vehicle.params

        if not self._cache:
            raise RuntimeError("_compute_obs_state() must be called before _compute_reward_done_info().")

        # キャッシュから取り出し
        v = self._cache["v"]
        e_y = self._cache["e_y"]
        e_psi_v = self._cache["e_psi_v"]
        v_ref_now = self._cache["v_ref_now"]
        delta_ref = self._cache["delta_ref"]
        beta = self._cache["beta"]

        # ★追加：現在の s に対応する曲率 kappa(s)
        # （_compute_obs_state() 側でキャッシュしていればそれを優先）
        if "kappa0" in self._cache:
            kappa0 = self._cache["kappa0"]
        else:
            kappa0 = self._interp_matrix(self.s, self.kappa_ref_mat)

        delta_geom = self.vehicle.steering_from_kappa(kappa0)



        # ---- 報酬計算 ----
        # cost_y = w.w_y * (e_y ** 2)
        # cost_psi = w.w_psi * (e_psi_v ** 2)
        cost_y = w.w_y * self._penalty(e_y, w.loss_y)

        # 速度ベクトル方向の偏差 = e_psi_v (車体向き偏差) + beta (スリップ角)
        e_psi_vel = self._wrap_angle(e_psi_v + beta)
        cost_psi = w.w_psi * self._penalty(e_psi_vel, w.loss_psi)

        dv = v - v_ref_now
        # cost_v_under = w.w_v_under * (dv ** 2)
        # cost_v_over = w.w_v_over * (dv ** 2)
        dv_pen = self._penalty(dv, w.loss_v)
        cost_v_under = w.w_v_under * dv_pen
        cost_v_over = w.w_v_over * dv_pen
        cost_v = torch.where(v <= v_ref_now, cost_v_under, cost_v_over)

        F_yf = self.vehicle.F_yf
        F_yr = self.vehicle.F_yr
        a_y = (F_yf + F_yr) / p.m
        # cost_ay = w.w_ay * (a_y ** 2)
        cost_ay = w.w_ay * self._penalty(a_y, w.loss_ay)

        # d_delta_ref（ステア指令レート）のペナルティ（適用値で計算）
        action = action.to(device=self.device, dtype=self.dtype)
        assert action.shape == (B, 2), f"action shape must be ({B},2), got {action.shape}"
        d_delta_ref_applied = torch.clamp(action[:, 1], -p.max_steer_rate, p.max_steer_rate)
        # cost_d_delta_ref = w.w_d_delta_ref * (d_delta_ref_applied ** 2)
        cost_d_delta_ref = w.w_d_delta_ref * self._penalty(d_delta_ref_applied, w.loss_d_delta_ref)

        total_cost = cost_y + cost_psi + cost_v + cost_ay + cost_d_delta_ref
        reward = -total_cost

        # ---- done 判定 ----
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
                "s": self.s.data.clone(),  # ★追加：現在 s の曲率
                "kappa0": kappa0,  # ★追加：現在 s の曲率
                "delta_geom": delta_geom,  # ★追加：現在 s の曲率
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[Dict[str, Any]]]:
        """
        action: (B, 2) [a_ref, d_delta_ref] の Tensor
        Returns:
            obs   : (B, obs_dim)
            state : (B, state_dim)
            reward: (B,)
            done  : (B,)
            info  : None または dict[str, Tensor]
        """
        assert action.shape[0] == self.B and action.shape[1] == 2, \
            f"action must be (B,2) with B={self.B}, got {action.shape}"

        # 1. 車両ダイナミクス更新（バッチ）
        self.vehicle.step(action, self.dt)

        # 2. Frenet 状態更新
        v = self.vehicle.state[:, 3]   # (B,)
        r = self.vehicle.state[:, 7]

        # 現在 s に対する κ(s) を補間
        kappa0 = self._interp_matrix(self.s, self.kappa_ref_mat)  # (B,)

        denom = 1.0 - kappa0 * self.e_y
        denom = torch.clamp(denom, min=0.1)  # 特異点回避

        s_dot = v * torch.cos(self.e_psi_v) / denom
        e_y_dot = v * torch.sin(self.e_psi_v)
        e_psi_v_dot = r - kappa0 * s_dot

        dt_t = torch.as_tensor(self.dt, dtype=self.dtype, device=self.device)
        self.s = self.s + s_dot * dt_t
        self.e_y = self.e_y + e_y_dot * dt_t
        self.e_psi_v = self._wrap_angle(self.e_psi_v + e_psi_v_dot * dt_t)

        # ステップ数カウント
        self.step_count += 1

        # 3. obs/state を計算（内部キャッシュ更新）→ reward/done/info を計算
        obs, state = self._compute_obs_state()
        reward, done, info = self._compute_reward_done_info(
            action=action,
            compute_info=compute_info,
        )

        if not torch.isfinite(self.vehicle.state).all():
            print("Non-finite in vehicle.state",
                "maxabs:", self.vehicle.state.abs().max().item())
            raise RuntimeError("Inf/NaN in vehicle.state")


        if torch.isnan(self.vehicle.state).any():
            print("NaN in vehicle.state, step_count:", self.step_count.max().item())
            raise RuntimeError("NaN in vehicle.state")

        if torch.isnan(self.s).any() or torch.isnan(self.e_y).any() or torch.isnan(self.e_psi_v).any():
            print("NaN in Frenet state")
            raise RuntimeError("NaN in Frenet state")

        return obs, state, reward, done, info
