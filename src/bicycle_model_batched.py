import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional


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

    tau_a: float = 0.3     # 加速度アクチュエータ時定数 [s]
    tau_delta: float = 0.3 # ステアアクチュエータ時定数 [s]

    max_steer_deg: float = 30.0  # 最大舵角 [deg]
    max_accel: float = 3.0       # 最大加速 [m/s^2]
    min_accel: float = -6.0      # 最大減速 [m/s^2]（負）

    @property
    def max_steer(self) -> float:
        # ラジアンに変換
        return self.max_steer_deg * 3.141592653589793 / 180.0


# =========================
#  バッチ対応 DynamicBicycleModel（PyTorch）
# =========================

class BatchedDynamicBicycleModel(nn.Module):
    """
    x, y, psi, v, a, delta, beta, r を状態にもつ線形動的 2輪モデル（バッチ対応）

    state: (B, 8)
        [0] x
        [1] y
        [2] psi
        [3] v
        [4] a
        [5] delta
        [6] beta
        [7] r

    action: (B, 2)
        [0] a_ref
        [1] delta_ref
    """

    def __init__(
        self,
        params: Optional[VehicleParams] = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.params = params if params is not None else VehicleParams()
        self.device = device
        self.dtype = dtype


    @property
    def batch_size(self) -> int:
        return self.state.shape[0]

    # --------------------
    # 初期化
    # --------------------
    def reset(self, init_state: torch.Tensor) -> torch.Tensor:
        """
        init_state: shape = (B, 8)
          各列は [x, y, psi, v, a, delta, beta, r] を表す。

        このテンソルで内部状態 self.state を上書きする。
        バッチサイズ B は init_state の 0 次元から自動で決まる。
        """
        assert init_state.dim() == 2, f"init_state must be 2D (B, 8), got {init_state.shape}"
        assert init_state.size(1) == 8, f"last dim of init_state must be 8, got {init_state.size(1)}"

        # device / dtype を合わせる
        init_state = init_state.to(device=self.device, dtype=self.dtype)
        B = init_state.size(0)

        # もし既存の state とバッチサイズが違えば作り直す

        # ここで新しいテンソルを割り当てる
        self.state = init_state.clone()
        self.F_yf = torch.zeros(B, dtype=self.dtype, device=self.device)
        self.F_yr = torch.zeros(B, dtype=self.dtype, device=self.device)

        return self.state.clone()

    # --------------------
    # step: バッチ一括更新
    # --------------------
    @torch.no_grad()
    def step(self, action: torch.Tensor, dt: float) -> torch.Tensor:
        """
        action: (B, 2) = [a_ref, delta_ref]
        dt: float （全バッチ共通）

        戻り値: state (B, 8) の最新状態
        """
        p = self.params
        B = self.batch_size

        assert action.shape == (B, 2), f"action shape must be ({B}, 2), got {action.shape}"

        # 状態展開
        x = self.state[:, 0]
        y = self.state[:, 1]
        psi = self.state[:, 2]
        v = self.state[:, 3]
        a = self.state[:, 4]
        delta = self.state[:, 5]
        beta = self.state[:, 6]
        r = self.state[:, 7]

        # --- 入力 a_ref, delta_ref をクリップ ---
        a_ref = torch.clamp(action[:, 0], p.min_accel, p.max_accel)
        delta_ref = torch.clamp(action[:, 1], -p.max_steer, p.max_steer)

        # --- アクチュエータ一次遅れ ---
        a_dot = (a_ref - a) / p.tau_a
        delta_dot = (delta_ref - delta) / p.tau_delta

        # --- タイヤモデル用の有効速度 ---
        v_eff = torch.clamp(v, min=0.1)

        # --- スリップ角 ---
        alpha_f = beta + p.lf * r / v_eff - delta
        alpha_r = beta - p.lr * r / v_eff

        # --- タイヤ力（線形モデル） ---
        F_yf = -p.Cf * alpha_f
        F_yr = -p.Cr * alpha_r

        # 保存（後で横加速度などに使いたいとき用）
        self.F_yf = F_yf
        self.F_yr = F_yr

        # --- β と r のダイナミクス ---
        beta_dot = (F_yf + F_yr) / (p.m * v_eff) - r
        r_dot = (p.lf * F_yf - p.lr * F_yr) / p.Iz

        # --- 縦方向 ---
        v_dot = a

        # --- 位置と姿勢 ---
        psi_eff = psi + beta
        x_dot = v * torch.cos(psi_eff)
        y_dot = v * torch.sin(psi_eff)
        psi_dot = r

        # --- 状態更新（オイラー） ---
        dt_t = torch.as_tensor(dt, dtype=self.dtype, device=self.device)

        x = x + x_dot * dt_t
        y = y + y_dot * dt_t
        psi = psi + psi_dot * dt_t
        v = v + v_dot * dt_t
        a = a + a_dot * dt_t
        delta = delta + delta_dot * dt_t
        beta = beta + beta_dot * dt_t
        r = r + r_dot * dt_t

        # --- 後処理（クリップなど） ---
        v = torch.clamp(v, min=0.0)
        delta = torch.clamp(delta, -p.max_steer, p.max_steer)

        # 状態をまとめて保存
        self.state[:, 0] = x
        self.state[:, 1] = y
        self.state[:, 2] = psi
        self.state[:, 3] = v
        self.state[:, 4] = a
        self.state[:, 5] = delta
        self.state[:, 6] = beta
        self.state[:, 7] = r

        return self.state.clone()
