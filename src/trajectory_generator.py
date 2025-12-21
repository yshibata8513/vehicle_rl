import numpy as np
from dataclasses import dataclass
from typing import Optional

# 既にどこかで定義済みの想定
@dataclass
class ReferenceTrajectory:
    s_ref: np.ndarray
    x_ref: np.ndarray
    y_ref: np.ndarray
    kappa_ref: np.ndarray
    v_ref: np.ndarray
    dt: float


def generate_random_reference_trajectory_arc_mix(
    total_length: float = 2000.0,   # 経路全長 [m]
    ds: float = 1.0,                # s の離散ステップ [m]
    dt: float = 0.05,               # 推奨シミュレーション刻み [s]
    v_min_kph: float = 20.0,        # 目標速度の下限 [km/h]
    v_max_kph: float = 60.0,        # 目標速度の上限 [km/h]
    R_min: float = 100.0,           # 最小曲率半径 [m]（|kappa| <= 1/R_min）
    R_max: float = 500.0,           # 最大曲率半径 [m]（より緩いカーブ用）
    seg_len_min: float = 80.0,      # 各セグメントの最短長さ [m]
    seg_len_max: float = 250.0,     # 各セグメントの最長長さ [m]
    transition_length: float = 30.0,# 曲率遷移区間の長さの上限 [m]
    kappa_step_max: float = 0.002,  # 隣接セグメント間の曲率差の最大値 [1/m]
    straight_prob: float = 0.5,     # 直線セグメントを選ぶ確率
    seed: Optional[int] = None,
) -> ReferenceTrajectory:
    """
    直線＋円弧＋線形曲率遷移をランダムに組み合わせた ReferenceTrajectory を生成する。

    - 曲率は |kappa| <= 1/R_min に制限
    - 直線は kappa=0
    - 円弧は一定曲率 kappa = ±1/R, R ∈ [R_min, R_max]
    - その間を一定長さの線形曲率遷移（クロソイド風）でつなぐ
    """

    rng = np.random.default_rng(seed)

    # s グリッド
    n_points = int(total_length / ds) + 1
    s_ref = np.arange(n_points) * ds

    # セグメントリスト: (seg_type, L_seg, kappa_start, kappa_end)
    # seg_type: "const" or "linear"
    segments = []
    prev_kappa = 0.0
    s_accum = 0.0
    kappa_abs_max = 1.0 / R_min

    while s_accum < total_length:
        L_avail = total_length - s_accum

        # 残りが小さすぎるときは最後のセグメントに吸収
        if L_avail < seg_len_min * 0.5:
            if segments:
                seg_type, L, ks, ke = segments[-1]
                segments[-1] = (seg_type, L + L_avail, ks, ke)
            else:
                segments.append(("const", L_avail, 0.0, 0.0))
            s_accum = total_length
            break

        # ターゲット曲率を決める（直線か円弧）
        if rng.random() < straight_prob:
            kappa_target_raw = 0.0
        else:
            R = rng.uniform(R_min, R_max)
            kappa_mag = 1.0 / R
            sign = rng.choice([-1.0, 1.0])
            kappa_target_raw = sign * kappa_mag

        # 曲率のステップを制限（dκを小さくする）
        delta = kappa_target_raw - prev_kappa
        if abs(delta) > kappa_step_max:
            delta = np.sign(delta) * kappa_step_max

        # 絶対値も制限（R_min 以下にならないように）
        kappa_target = np.clip(prev_kappa + delta, -kappa_abs_max, kappa_abs_max)

        # セグメント長をランダムに決定
        L_const = float(rng.uniform(seg_len_min, seg_len_max))

        # 曲率が変わるなら遷移区間をつくる
        L_tr = 0.0
        if abs(kappa_target - prev_kappa) > 1e-6:
            L_tr = min(transition_length, max(0.0, L_avail - seg_len_min * 0.5))

        # 残り長さに収まるように調整
        if L_const + L_tr > L_avail:
            L_const = max(seg_len_min * 0.5, L_avail - L_tr)

        # 遷移セグメント（線形に κ を変化）
        if L_tr > 1e-6 and abs(kappa_target - prev_kappa) > 1e-6:
            segments.append(("linear", L_tr, prev_kappa, kappa_target))
            s_accum += L_tr

        # 一定曲率セグメント
        segments.append(("const", L_const, kappa_target, kappa_target))
        s_accum += L_const
        prev_kappa = kappa_target

    # -------------------------------------------------
    # κ(s) を s_ref 上にサンプリング
    # -------------------------------------------------
    kappa_ref = np.zeros_like(s_ref)
    seg_idx = 0
    seg_s = 0.0  # 現在のセグメント内の s

    for i in range(n_points):
        # セグメント切り替え
        while seg_idx < len(segments) and seg_s >= segments[seg_idx][1] - 1e-9:
            seg_s -= segments[seg_idx][1]
            seg_idx += 1

        if seg_idx >= len(segments):
            kappa = prev_kappa
        else:
            seg_type, L_seg, ks, ke = segments[seg_idx]
            if L_seg <= 0:
                kappa = ke
            elif seg_type == "const":
                kappa = ks
            else:  # "linear"
                ratio = seg_s / L_seg
                kappa = ks + (ke - ks) * ratio

        kappa_ref[i] = kappa
        seg_s += ds

    # -------------------------------------------------
    # κ(s) から ψ(s) と (x(s), y(s)) を積分で計算
    # -------------------------------------------------
    psi = np.zeros_like(s_ref)
    x_ref = np.zeros_like(s_ref)
    y_ref = np.zeros_like(s_ref)

    for i in range(1, n_points):
        psi[i] = psi[i - 1] + kappa_ref[i - 1] * ds
        x_ref[i] = x_ref[i - 1] + np.cos(psi[i - 1]) * ds
        y_ref[i] = y_ref[i - 1] + np.sin(psi[i - 1]) * ds

    # -------------------------------------------------
    # 目標速度プロファイル（区間全体で一定、20〜60 km/h のランダム）
    # -------------------------------------------------
    v_kph = rng.uniform(v_min_kph, v_max_kph)
    v_ms = v_kph / 3.6
    v_ref = np.full_like(s_ref, v_ms)

    return ReferenceTrajectory(
        s_ref=s_ref,
        x_ref=x_ref,
        y_ref=y_ref,
        kappa_ref=kappa_ref,
        v_ref=v_ref,
        dt=dt,
    )


def calculate_max_curvature_rates(
    transition_length: float,
    kappa_step_max: float,
    v_max_kph: float,
) -> tuple[float, float]:
    """
    generate_random_reference_trajectory_arc_mix のパラメータから
    理論上の最大曲率変化率を算出する。

    Args:
        transition_length: 曲率遷移区間の長さ [m]
        kappa_step_max: 隣接セグメント間の曲率差の最大値 [1/m]
        v_max_kph: 目標速度の上限 [km/h]

    Returns:
        max_dk_ds: 空間に対する最大曲率変化率 [1/m^2] (dκ/ds)
        max_dk_dt: 時間に対する最大曲率変化率 [1/(m*s)] (dκ/dt)
                   ※ ステアリング動作速度の要件見積もりに使用
    """
    if transition_length <= 1e-6:
        raise ValueError("transition_length must be greater than 0 to calculate rates.")

    # 1. 空間微分 (dκ/ds) の最大値
    # 線形遷移区間において、距離 transition_length で最大 kappa_step_max 変化する
    max_dk_ds = kappa_step_max / transition_length

    # 2. 時間微分 (dκ/dt) の最大値
    # dκ/dt = (dκ/ds) * (ds/dt) = (dκ/ds) * v
    # 速度が最大のときに、時間あたりの変化率は最大になる
    v_max_ms = v_max_kph / 3.6
    max_dk_dt = max_dk_ds * v_max_ms

    return max_dk_ds, max_dk_dt
