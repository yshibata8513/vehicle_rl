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
    mu_ref: np.ndarray
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
    kappa_step_max: float = 0.5, # 隣接セグメント間の曲率差の最大値 [1/m]
    straight_prob: float = 0.5,     # 直線セグメントを選ぶ確率
    seed: Optional[int] = None,
    # --- 追加（ショーツ式） ---
    P_allow: float = 0.6,           # 許容遠心加速度変化率（横方向ジャーク）[m/s^3]
    t_min: float = 3.0,             # 緩和走行時間の下限 [s]（0.0 で無効化）
    a_lat_max_g: float = 0.30,   # 許容横G（例: 0.15〜0.25 など）
    mu_min: float = 0.1,            # 摩擦係数の最小値 [-]
    mu_max: float = 1.0,            # 摩擦係数の最大値 [-]
) -> ReferenceTrajectory:
    """
    直線＋円弧＋線形曲率遷移をランダムに組み合わせた ReferenceTrajectory を生成する。

    - 曲率は |kappa| <= 1/R_min に制限
    - 直線は kappa=0
    - 円弧は一定曲率 kappa = ±1/R, R ∈ [R_min, R_max]
    - その間を一定長さの線形曲率遷移（クロソイド風）でつなぐ

    変更点:
    - セグメントごとに目標速度 v_ref をふり直す。
    - 「遷移セグメント」とそれに続く「一定曲率セグメント」は同じ曲率ターゲットを使う
      （= 遷移の終端 κ と一定区間の κ が一致する。元実装の意図を明確化しつつ保持）。
    - 遷移長 L_tr はショーツ式（P_allow）および緩和走行時間下限（t_min）に基づいて動的決定する。
      線形曲率遷移で P = v^3 * (dκ/ds), かつ dκ/ds = Δκ/L より L >= v^3|Δκ|/P
    """
    rng = np.random.default_rng(seed)

    # s グリッド
    n_points = int(total_length / ds) + 1
    s_ref = np.arange(n_points) * ds

    # セグメントリスト: (seg_type, L_seg, kappa_start, kappa_end, v_ms)
    # seg_type: "const" or "linear"
    segments = []
    prev_kappa = 0.0
    s_accum = 0.0
    kappa_abs_max = 1.0 / R_min

    is_first_segment = True

    while s_accum < total_length:
        L_avail = total_length - s_accum

        # 残りが小さすぎるときは最後のセグメントに吸収
        if L_avail < seg_len_min * 0.5:
            if segments:
                seg_type, L, ks, ke, v_ms = segments[-1]
                segments[-1] = (seg_type, L + L_avail, ks, ke, v_ms)
            else:
                v_kph = rng.uniform(v_min_kph, v_max_kph)
                v_ms = v_kph / 3.6
                segments.append(("const", L_avail, 0.0, 0.0, v_ms))
            s_accum = total_length
            break

        # ターゲット曲率を決める（直線か円弧）
        if is_first_segment or rng.random() < straight_prob:
            kappa_target_raw = 0.0
            is_first_segment = False
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

        # セグメントごとの速度をサンプル（遷移+一定曲率で同一速度）
        # 上限は曲率と許容横Gから決める: a_y = v^2*|kappa| <= a_lat_max
        G0 = 9.80665
        if abs(kappa_target) > 1e-9:
            a_lat_max = a_lat_max_g * G0
            v_cap_ms = np.sqrt(a_lat_max / abs(kappa_target))
            v_cap_kph = min(v_max_kph, v_cap_ms * 3.6)
        else:
            v_cap_kph = v_max_kph

        # 上限が下限を下回る場合は上限に張り付け
        if v_cap_kph <= v_min_kph:
            v_kph = v_cap_kph
        else:
            v_kph = rng.uniform(v_min_kph, v_cap_kph)
        v_ms = v_kph / 3.6


        # セグメント長をランダムに決定
        L_const = float(rng.uniform(seg_len_min, seg_len_max))

        # 曲率が変わるなら遷移区間をつくる（ショーツ式で動的決定）
        L_tr = 0.0
        delta_kappa = kappa_target - prev_kappa
        if abs(delta_kappa) > 1e-6:
            # ショーツ式: L >= v^3 * |Δκ| / P
            P_eff = max(float(P_allow), 1e-9)
            L_tr_req = (v_ms ** 3) * abs(delta_kappa) / P_eff

            # 任意: 緩和走行時間下限 t_min（L >= v*t）
            if t_min > 1e-9:
                L_tr_req = max(L_tr_req, v_ms * t_min)

            # 既存の「遷移長上限」や残り長さ制約に合わせてクリップ
            L_tr_cap = min(transition_length, max(0.0, L_avail - seg_len_min * 0.5))
            L_tr = min(L_tr_req, L_tr_cap)

            # もし L_tr が不足するなら、Δκ を縮めて P 条件を満たすようにする
            if L_tr < L_tr_req - 1e-6:
                if L_tr <= 1e-9:
                    # 遷移区間が取れないなら曲率は変えない
                    kappa_target = prev_kappa
                    delta_kappa = 0.0
                    L_tr = 0.0
                else:
                    # この L_tr で許容される最大 |Δκ|（ジャーク制約）
                    delta_kappa_max_by_P = (P_eff * L_tr) / (v_ms ** 3)

                    # t_min が効いている場合でも、P制約は満たしたいので Δκ をP側に合わせて縮める
                    delta_kappa_eff = min(abs(delta_kappa), delta_kappa_max_by_P)
                    kappa_target = prev_kappa + np.sign(delta_kappa) * delta_kappa_eff
                    kappa_target = float(np.clip(kappa_target, -kappa_abs_max, kappa_abs_max))
                    delta_kappa = kappa_target - prev_kappa

        # 残り長さに収まるように調整
        if L_const + L_tr > L_avail:
            L_const = max(seg_len_min * 0.5, L_avail - L_tr)

        # 遷移セグメント（線形に κ を変化）: 終端は kappa_target
        if L_tr > 1e-6 and abs(kappa_target - prev_kappa) > 1e-6:
            segments.append(("linear", L_tr, prev_kappa, kappa_target, v_ms))
            s_accum += L_tr

        # 一定曲率セグメント: κ は遷移終端と同じ kappa_target
        segments.append(("const", L_const, kappa_target, kappa_target, v_ms))
        s_accum += L_const
        prev_kappa = kappa_target

    # -------------------------------------------------
    # κ(s), v(s) を s_ref 上にサンプリング
    # -------------------------------------------------
    kappa_ref = np.zeros_like(s_ref)
    v_ref = np.zeros_like(s_ref)
    seg_idx = 0
    seg_s = 0.0  # 現在のセグメント内の s

    for i in range(n_points):
        # セグメント切り替え
        while seg_idx < len(segments) and seg_s >= segments[seg_idx][1] - 1e-9:
            seg_s -= segments[seg_idx][1]
            seg_idx += 1

        if seg_idx >= len(segments):
            kappa = prev_kappa
            v_ms = segments[-1][4] if segments else (rng.uniform(v_min_kph, v_max_kph) / 3.6)
        else:
            seg_type, L_seg, ks, ke, v_ms = segments[seg_idx]
            if L_seg <= 0:
                kappa = ke
            elif seg_type == "const":
                kappa = ks
            else:  # "linear"
                ratio = seg_s / L_seg
                kappa = ks + (ke - ks) * ratio

        kappa_ref[i] = kappa
        v_ref[i] = v_ms
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
    # μ(s) を生成（トラジェクトリ全体で一定のランダム値）
    # -------------------------------------------------
    mu_value = rng.uniform(mu_min, mu_max)
    mu_ref = np.full_like(s_ref, mu_value)
    
    return ReferenceTrajectory(
        s_ref=s_ref,
        x_ref=x_ref,
        y_ref=y_ref,
        kappa_ref=kappa_ref,
        v_ref=v_ref,
        mu_ref=mu_ref,
        dt=dt,
    )


def calculate_max_curvature_rates(
    transition_length: float,
    kappa_step_max: float,
    v_max_kph: float,
    # --- 追加（ショーツ式を使う場合のオプション） ---
    P_allow: Optional[float] = None,
    v_min_kph: Optional[float] = None,
    t_min: float = 0.0,
) -> tuple[float, float]:
    """
    generate_random_reference_trajectory_arc_mix のパラメータから
    理論上の最大曲率変化率を算出する。

    互換性のため従来式（transition_length と kappa_step_max）も残しつつ、
    P_allow が与えられた場合はショーツ式ベースの上限推定も行う。

    Args:
        transition_length: 曲率遷移区間の長さ [m]（上限）
        kappa_step_max: 隣接セグメント間の曲率差の最大値 [1/m]
        v_max_kph: 目標速度の上限 [km/h]

        P_allow: 許容遠心加速度変化率（横方向ジャーク）[m/s^3]
        v_min_kph: 目標速度の下限 [km/h]（ショーツ式の最悪ケースは低速側なので必要）
        t_min: 緩和走行時間の下限 [s]（0.0 で無効化）

    Returns:
        max_dk_ds: 空間に対する最大曲率変化率 [1/m^2] (dκ/ds)
        max_dk_dt: 時間に対する最大曲率変化率 [1/(m*s)] (dκ/dt)
                   ※ ステアリング動作速度の要件見積もりに使用
    """
    # --- ショーツ式ベースの推定（P_allow が指定された場合） ---
    if P_allow is not None:
        if v_min_kph is None:
            raise ValueError("v_min_kph must be provided when P_allow is specified.")
        v_min_ms = v_min_kph / 3.6
        if v_min_ms <= 1e-9:
            raise ValueError("v_min_kph must be greater than 0 when P_allow is specified.")

        P_eff = max(float(P_allow), 1e-9)

        # L = max(v^3*Δκ/P, v*t_min) を想定すると
        # dκ/ds = Δκ/L <= min(P/v^3, Δκ/(v*t_min))
        # Δκ の上限を kappa_step_max とみなしたときの最大値
        if t_min > 1e-9:
            max_dk_ds = min(P_eff / (v_min_ms ** 3), kappa_step_max / (v_min_ms * t_min))
            max_dk_dt = min(P_eff / (v_min_ms ** 2), kappa_step_max / t_min)
        else:
            max_dk_ds = P_eff / (v_min_ms ** 3)
            max_dk_dt = P_eff / (v_min_ms ** 2)

        return float(max_dk_ds), float(max_dk_dt)

    # --- 従来の上限推定（互換維持） ---
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

    return float(max_dk_ds), float(max_dk_dt)
