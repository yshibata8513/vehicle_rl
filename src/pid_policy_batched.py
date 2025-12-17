import torch
from dataclasses import dataclass
from typing import Optional

from bicycle_model import VehicleParams


@dataclass
class PIDGains:
    # 速度制御（v_ref - v）
    kv_p: float = 1.0
    kv_i: float = 0.1
    kv_d: float = 0.0

    # 姿勢（速度ベクトル）偏差 e_psi_v 用
    kpsi_p: float = 1.5
    kpsi_i: float = 0.0
    kpsi_d: float = 0.1

    # 横偏差 e_y 用
    ky_p: float = 0.1
    ky_i: float = 0.0
    ky_d: float = 0.0


class BatchedPIDPolicy:
    """
    BatchedPathTrackingEnvFrenet 向けのテスト用バッチ PID ポリシー。

    ・obs テンソル（B, obs_dim）から
        e_y      : obs[:, 0]
        e_psi_v  : obs[:, 1]
        v        : obs[:, 2]
        v_ref    : obs[:, 7]
      を取り出して制御入力を計算。

    ・出力は (B, 2) テンソル
        [:, 0] : a_ref  (目標加速度)
        [:, 1] : delta_ref (目標舵角)
    """

    # obs のカラム index（BatchedPathTrackingEnvFrenet に合わせる）
    IDX_EY = 0
    IDX_EPSI_V = 1
    IDX_V = 2
    IDX_VREF = 7

    def __init__(
        self,
        vehicle_params: VehicleParams,
        dt: float,
        batch_size: int,
        gains: Optional[PIDGains] = None,
        i_limit_accel: float = 10.0,   # 積分項の簡易リミット（加速度用）
        i_limit_steer: float = 10.0,   # 積分項の簡易リミット（ステア用）
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.p = vehicle_params
        self.dt = float(dt)
        self.gains = gains if gains is not None else PIDGains()
        self.B = batch_size
        self.device = device
        self.dtype = dtype

        self.i_limit_accel = float(i_limit_accel)
        self.i_limit_steer = float(i_limit_steer)

        # 積分・微分用の内部状態（バッチ分）
        self.e_v_int = torch.zeros(self.B, dtype=dtype, device=device)
        self.e_v_prev = torch.zeros(self.B, dtype=dtype, device=device)

        self.e_psi_int = torch.zeros(self.B, dtype=dtype, device=device)
        self.e_psi_prev = torch.zeros(self.B, dtype=dtype, device=device)

        self.e_y_int = torch.zeros(self.B, dtype=dtype, device=device)
        self.e_y_prev = torch.zeros(self.B, dtype=dtype, device=device)

    def reset(self, mask: Optional[torch.Tensor] = None):
        """
        内部状態をリセット。

        Args:
            mask: (B,) の bool Tensor。
                  None のときは全バッチをリセット。
                  指定したときは mask == True の要素のみリセット。
                  （例: done になった env だけ積分をクリアしたい場合）
        """
        if mask is None:
            self.e_v_int.zero_()
            self.e_v_prev.zero_()
            self.e_psi_int.zero_()
            self.e_psi_prev.zero_()
            self.e_y_int.zero_()
            self.e_y_prev.zero_()
        else:
            assert mask.shape[0] == self.B
            # True のところだけ 0 に
            self.e_v_int[mask] = 0.0
            self.e_v_prev[mask] = 0.0
            self.e_psi_int[mask] = 0.0
            self.e_psi_prev[mask] = 0.0
            self.e_y_int[mask] = 0.0
            self.e_y_prev[mask] = 0.0

    @torch.no_grad()
    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs: (B, obs_dim) の Tensor
             カラム構成は BatchedPathTrackingEnvFrenet の obs を想定。

        戻り値:
            actions: (B, 2) Tensor [a_ref, delta_ref]
        """
        assert obs.shape[0] == self.B, f"obs batch size must be {self.B}, got {obs.shape[0]}"
        assert obs.shape[1] > self.IDX_VREF, "obs dim is too small for expected indices"

        g = self.gains
        dt = self.dt

        # ------------------
        #  1) 速度 PID
        # ------------------
        v = obs[:, self.IDX_V]        # (B,)
        v_ref = obs[:, self.IDX_VREF] # (B,)
        e_v = v_ref - v               # (B,) 目標より遅いとき正

        # 積分・微分
        self.e_v_int += e_v * dt
        self.e_v_int = torch.clamp(self.e_v_int, -self.i_limit_accel, self.i_limit_accel)
        e_v_dot = (e_v - self.e_v_prev) / dt
        self.e_v_prev = e_v

        a_cmd = (
            g.kv_p * e_v
            + g.kv_i * self.e_v_int
            + g.kv_d * e_v_dot
        )  # (B,)

        # 加速度クリップ
        a_ref = torch.clamp(
            a_cmd,
            min=self.p.min_accel,
            max=self.p.max_accel,
        )  # (B,)

        # ------------------
        #  2) ステア PID
        # ------------------
        e_psi_v = obs[:, self.IDX_EPSI_V]   # (B,)
        e_y = obs[:, self.IDX_EY]           # (B,)

        # (1) 姿勢偏差 PID
        self.e_psi_int += e_psi_v * dt
        self.e_psi_int = torch.clamp(self.e_psi_int, -self.i_limit_steer, self.i_limit_steer)
        e_psi_dot = (e_psi_v - self.e_psi_prev) / dt
        self.e_psi_prev = e_psi_v

        # e_psi_v > 0（左を向きすぎ）→ 右に切りたい → -符号
        delta_psi = -(
            g.kpsi_p * e_psi_v
            + g.kpsi_i * self.e_psi_int
            + g.kpsi_d * e_psi_dot
        )  # (B,)

        # (2) 横偏差 PID
        self.e_y_int += e_y * dt
        self.e_y_int = torch.clamp(self.e_y_int, -self.i_limit_steer, self.i_limit_steer)
        e_y_dot = (e_y - self.e_y_prev) / dt
        self.e_y_prev = e_y

        # e_y > 0（左にずれている）→ 右に切りたい → -符号
        delta_y = -(
            g.ky_p * e_y
            + g.ky_i * self.e_y_int
            + g.ky_d * e_y_dot
        )  # (B,)

        # 姿勢と横偏差の補正を足し合わせる
        delta_cmd = delta_psi + delta_y   # (B,)

        # ステア角クリップ
        delta_ref = torch.clamp(
            delta_cmd,
            min=-self.p.max_steer,
            max=self.p.max_steer,
        )  # (B,)

        # (B,2) にまとめて返す
        actions = torch.stack([a_ref, delta_ref], dim=1)
        return actions
