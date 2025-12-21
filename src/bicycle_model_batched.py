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

    max_steer_deg: float = 30.0      # 最大舵角 [deg]
    max_steer_rate_deg: float = 180.0  # 最大舵角指令レート [deg/s]
    max_accel: float = 3.0           # 最大加速 [m/s^2]
    min_accel: float = -6.0          # 最大減速 [m/s^2]（負）

    @property
    def max_steer(self) -> float:
        # ラジアンに変換
        return self.max_steer_deg * 3.141592653589793 / 180.0

    @property
    def max_steer_rate(self) -> float:
        # ラジアン/秒に変換
        return self.max_steer_rate_deg * 3.141592653589793 / 180.0


# =========================
#  バッチ対応 DynamicBicycleModel（PyTorch）
# =========================

class BatchedDynamicBicycleModel(nn.Module):
    """
    x, y, psi, v, a, delta, beta, r, delta_ref を状態にもつ線形動的 2輪モデル（バッチ対応）

    state: (B, 9)
        [0] x
        [1] y
        [2] psi
        [3] v
        [4] a
        [5] delta
        [6] beta
        [7] r
        [8] delta_ref  (ステア指令角 [rad])

    action: (B, 2)
        [0] a_ref
        [1] d_delta_ref (delta_ref の時間微分指令 [rad/s])

    NOTE:
      - 従来は action に delta_ref を直接入れていたが、舵角振動抑制のために
        action を「delta_ref のレート入力」に変更した。
      - delta_ref 自体は状態として保持し、step 内で積分して更新する。
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
        init_state: shape = (B, 9)
          各列は [x, y, psi, v, a, delta, beta, r, delta_ref] を表す。

        互換のため、(B, 8) も受け付ける。その場合 delta_ref は delta と同じ値で初期化する。

        このテンソルで内部状態 self.state を上書きする。
        バッチサイズ B は init_state の 0 次元から自動で決まる。
        """
        assert init_state.dim() == 2, f"init_state must be 2D (B, 8|9), got {init_state.shape}"
        assert init_state.size(1) in (8, 9), f"last dim of init_state must be 8 or 9, got {init_state.size(1)}"

        # device / dtype を合わせる
        init_state = init_state.to(device=self.device, dtype=self.dtype)

        # (B, 8) の場合は delta_ref を付与して (B, 9) に拡張
        if init_state.size(1) == 8:
            delta = init_state[:, 5:6]
            init_state = torch.cat([init_state, delta], dim=1)

        B = init_state.size(0)

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
        action: (B, 2) = [a_ref, d_delta_ref]
        dt: float （全バッチ共通）

        戻り値: state (B, 9) の最新状態
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
        delta_ref = self.state[:, 8]

        # --- 入力 a_ref, d_delta_ref をクリップ ---
        a_ref = torch.clamp(action[:, 0], p.min_accel, p.max_accel)
        d_delta_ref = torch.clamp(action[:, 1], -p.max_steer_rate, p.max_steer_rate)

        # --- delta_ref の更新（舵角指令レートを積分） ---
        dt_t = torch.as_tensor(dt, dtype=self.dtype, device=self.device)
        delta_ref = delta_ref + d_delta_ref * dt_t
        delta_ref = torch.clamp(delta_ref, -p.max_steer, p.max_steer)

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
        self.state[:, 8] = delta_ref

        return self.state.clone()

    @torch.no_grad()
    def steering_from_kappa(self, kappa0: torch.Tensor) -> torch.Tensor:
        """
        曲率 kappa0 から幾何学（キネマティック）に必要な舵角を推定して返す。

        前提:
        - 低スリップ近似（β≈0）かつ自転車モデルの幾何学関係
        - ホイールベース L = lf + lr
        - 曲率 κ と舵角 δ の関係: κ = tan(δ) / L  =>  δ = atan(L * κ)

        Args:
            kappa0: (B,) 現在の s に対応する曲率 [1/m]

        Returns:
            delta_geom: (B,) 幾何学的舵角 [rad]（±max_steer でクリップ）
        """
        p = self.params
        kappa0 = kappa0.to(device=self.device, dtype=self.dtype)

        L = torch.as_tensor(p.lf + p.lr, dtype=self.dtype, device=self.device)
        delta_geom = torch.atan(L * kappa0)

        # 物理上限でクリップ
        delta_geom = torch.clamp(delta_geom, -p.max_steer, p.max_steer)
        return delta_geom



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
