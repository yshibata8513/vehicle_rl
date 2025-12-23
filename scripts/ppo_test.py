import os
import datetime
import torch
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append("../src")
from bicycle_model_batched import VehicleParams, calculate_max_d_delta_ref
from environment_batched import BatchedPathTrackingEnvFrenet, RewardWeights
from trajectory_generator import generate_random_reference_trajectory_arc_mix, calculate_max_curvature_rates
from ppo_agent_batched import PPOAgentBatched


EVAL_MODE = False
CKPT_PATH = None
# CKPT_PATH = "./checkpoints/20251219_020201_update0200.pt"
# CKPT_PATH = "./checkpoints/20251219_032229_update0050.pt"

# ===== 実行ごとの一意な ID（学習開始時刻） =====
RUN_ID = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def save_checkpoint(agent, update, run_id=RUN_ID, ckpt_dir=CHECKPOINT_DIR):
    """
    agent のモデル・optimizer をチェックポイント保存する。
    ファイル名: {run_id}_update{update:04d}.pt
    """
    ckpt_path = os.path.join(ckpt_dir, f"{run_id}_update{update:04d}.pt")
    torch.save(
        {
            "run_id": run_id,
            "update": update,
            "model_state_dict": agent.net.state_dict(),
            "optimizer_state_dict": agent.optim.state_dict(),
        },
        ckpt_path,
    )
    print(f"[Checkpoint] Saved: {ckpt_path}")
    return ckpt_path


def load_checkpoint(agent, ckpt_path, device="cpu"):
    """
    チェックポイントを読み込み、agent に反映する。
    戻り値: (run_id, update)
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    agent.net.load_state_dict(ckpt["model_state_dict"])
    agent.optim.load_state_dict(ckpt["optimizer_state_dict"])
    run_id = ckpt.get("run_id", "unknown")
    update = ckpt.get("update", -1)
    print(f"[Checkpoint] Loaded: {ckpt_path} (run_id={run_id}, update={update})")
    return run_id, update


def save_ref_traj_plots(traj, run_id, out_dir="ref_traj_plots", prefix="eval_ref_traj"):
    """
    ReferenceTrajectory 1本について
      1) x-y プロット画像
      2) s横軸で kappa_ref / v_ref の2段プロット画像
    を作って保存する。

    Returns:
        (xy_path, prof_path)
    """
    os.makedirs(out_dir, exist_ok=True)

    # できるだけ一意になるように先頭点と長さも入れる（indexが無くても衝突しにくい）
    tag = f"L{float(traj.s_ref[-1]):.0f}_x0{float(traj.x_ref[0]):.2f}_y0{float(traj.y_ref[0]):.2f}"
    base = f"{prefix}_{run_id}_{tag}"

    # 1) x-y
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot(traj.x_ref, traj.y_ref, label="ref")
    ax.set_title("Reference XY")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.axis("equal")
    ax.grid(True)
    ax.legend(loc="best")
    plt.tight_layout()
    xy_path = os.path.join(out_dir, f"{base}_xy.png")
    plt.savefig(xy_path, dpi=150, bbox_inches="tight")
    plt.close()

    # 2) profiles (kappa_ref / v_ref vs s)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(traj.s_ref, traj.kappa_ref, label="kappa_ref")
    axes[0].set_ylabel("kappa_ref [1/m]")
    axes[0].grid(True)
    axes[0].legend(loc="best")

    axes[1].plot(traj.s_ref, traj.v_ref, label="v_ref")
    axes[1].set_xlabel("s [m]")
    axes[1].set_ylabel("v_ref [m/s]")
    axes[1].grid(True)
    axes[1].legend(loc="best")

    plt.tight_layout()
    prof_path = os.path.join(out_dir, f"{base}_profiles.png")
    plt.savefig(prof_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved: {xy_path}")
    print(f"Saved: {prof_path}")
    return xy_path, prof_path


# ===== 共通設定 =====
dt = 0.05
B = 64              # 学習用バッチサイズ
device = "cuda"
dtype = torch.float32
is_perturbed = False

# Trajectory Generation Params
R_MIN = 100.0


# STAGE3
weights = RewardWeights(
    w_y=0.1,
    w_psi=10.0,
    w_v_under=0.1,
    w_v_over=1.5,
    w_ay=0.001,
    w_d_delta_ref=.1,
    loss_y="l2",
    loss_psi="l2",
    loss_v="l2",
    loss_ay="l2",
    loss_d_delta_ref="l2",
)

veh_params = VehicleParams()


def make_train_traj(seed=None):
    # train 用の経路条件（いままで trafin_ref_trajs を作っていた条件と同じにする）
    return generate_random_reference_trajectory_arc_mix(
        total_length=2000.0,
        ds=1.0,
        dt=dt,
        v_min_kph=40.0,
        v_max_kph=60.0,
        R_min=R_MIN,
        R_max=400.0,
        seed=None,   # reset ごとにランダム
    )


# ===== 学習用 ref_traj / env =====
train_ref_trajs = [
    make_train_traj()
    for i in range(B)
]

# Kappa preview offsets: 0m to 20m, step=1.0m (21 points)
kappa_preview_offsets = [float(i) for i in range(21)]

train_env = BatchedPathTrackingEnvFrenet(
    ref_trajs=train_ref_trajs,
    vehicle_params=veh_params,
    reward_weights=weights,  # 明示的に渡す
    kappa_preview_offsets=kappa_preview_offsets,
    max_steps=2000,
    device=device,
    dtype=dtype,
    traj_generator=make_train_traj,  # ここを追加
)


# 初期状態テンプレート（学習・評価で使い回し）
def make_init_state(ref_trajs, device, dtype):
    B_local = len(ref_trajs)
    init_state_local = torch.zeros(B_local, 8, dtype=dtype, device=device)
    for b, traj in enumerate(ref_trajs):
        x_ref = traj.x_ref
        y_ref = traj.y_ref
        x0 = x_ref[0]
        y0 = y_ref[0]
        dx = x_ref[1] - x_ref[0]
        dy = y_ref[1] - y_ref[0]
        psi0 = float(torch.atan2(torch.tensor(dy), torch.tensor(dx)))
        v0 = float(traj.v_ref[0])
        init_state_local[b, 0] = x0
        init_state_local[b, 1] = y0
        init_state_local[b, 2] = psi0
        init_state_local[b, 3] = v0
    return init_state_local


init_state = make_init_state(train_ref_trajs, device, dtype)
obs, _, state = train_env.reset(init_state, is_perturbed=is_perturbed)  # ★ reset は (obs_norm, obs_raw, state)
obs_dim = obs.shape[1]
action_dim = 2

# ===== PPO エージェント =====

# Calculate action limits
# 1. Curvature rates based on trajectory generation parameters
max_dk_ds, max_dk_dt = calculate_max_curvature_rates(
    transition_length=30.0,
    kappa_step_max=0.002,
    v_max_kph=60.0,
)

# 2. Required steering rate (geometric)
max_d_delta_geom = calculate_max_d_delta_ref(
    vehicle_params=veh_params,
    max_dk_dt=max_dk_dt,
)

# 3. Set limits
#    d_delta_ref needs margin -> 2.0x (For reference / checking)
limit_d_delta = 2.0 * max_d_delta_geom

#    a_ref limit -> 0.3G
limit_accel = 0.3 * 9.80665

#    delta_ref limit -> max_steer (from vehicle params)
#    Proposed: 3 * steering_from_kappa(1/R_min)
max_kappa = 1.0 / R_MIN
delta_geom_max = train_env.vehicle.steering_from_kappa(torch.tensor(max_kappa))
limit_delta = 3.0 * float(delta_geom_max)
# Clamp to physical max steer just in case
limit_delta = min(limit_delta, veh_params.max_steer)

delta_geom_max_f = float(delta_geom_max.detach().cpu().item())

print(f"[Limits] max_dk_dt    = {max_dk_dt:.4f} [1/(m*s)]")
print(f"[Limits] max_d_delta  = {max_d_delta_geom:.4f} [rad/s] ({np.degrees(max_d_delta_geom):.1f} deg/s)")
print(f"[Limits] limit_d_delta= {limit_d_delta:.4f} [rad/s] ({np.degrees(limit_d_delta):.1f} deg/s) (Reference)")
print(f"[Limits] delta_geom_max={delta_geom_max_f:.4f} [rad] ({np.degrees(delta_geom_max_f):.1f} deg) (@ R={R_MIN}m)")
print(f"[Limits] limit_delta  = {limit_delta:.4f} [rad] ({np.degrees(limit_delta):.1f} deg)")
print(f"[Limits] limit_accel  = {limit_accel:.4f} [m/s^2]")

action_min = np.array([-limit_accel, -limit_delta], dtype=np.float32)
action_max = np.array([limit_accel, limit_delta], dtype=np.float32)

agent = PPOAgentBatched(
    obs_dim=obs_dim,
    action_dim=action_dim,
    num_envs=B,
    rollout_steps=64,
    device=device,
    action_min=action_min,
    action_max=action_max,
)

if CKPT_PATH is not None:
    load_checkpoint(agent, CKPT_PATH, device="cuda")


# ===== 評価用 env（小さめバッチ + 固定seed） =====
EVAL_ENVS = 8
eval_ref_trajs = [
    make_train_traj(seed=1000 + i)
    for i in range(EVAL_ENVS)
]

# ---- eval_ref_trajs 作成直後に先頭4本を保存 ----
for i, traj in enumerate(eval_ref_trajs[:4]):
    save_ref_traj_plots(traj, i)

eval_env = BatchedPathTrackingEnvFrenet(
    ref_trajs=eval_ref_trajs,
    vehicle_params=veh_params,
    reward_weights=weights,  # ← これを追加
    kappa_preview_offsets=kappa_preview_offsets,
    max_steps=2000,
    device=device,
    dtype=dtype,
)
eval_init_state = make_init_state(eval_ref_trajs, device, dtype)


@torch.no_grad()
def evaluate_policy(
    agent,
    env,
    init_state,
    ref_trajs,              # ← 参照経路リストを渡す
    num_episodes=3,
    max_steps=1000,
    use_partial_reset=True,
    save_xy=False,
    xy_filename=None,
    num_xy_envs=4,
):
    B_eval = env.B

    # ---- XY & 時系列ログ用の準備（最初の episode の先頭 num_xy_envs 分だけ）----
    num_xy_envs = min(num_xy_envs, B_eval)
    if save_xy:
        # XY
        xy_x_hist = [[] for _ in range(num_xy_envs)]
        xy_y_hist = [[] for _ in range(num_xy_envs)]
        xy_done = torch.zeros(B_eval, dtype=torch.bool, device=env.device)

        # 時系列用
        ts_v_hist = [[] for _ in range(num_xy_envs)]
        ts_vref_hist = [[] for _ in range(num_xy_envs)]
        ts_ax_hist = [[] for _ in range(num_xy_envs)]
        ts_ay_hist = [[] for _ in range(num_xy_envs)]
        ts_delta_hist = [[] for _ in range(num_xy_envs)]
        ts_ey_hist = [[] for _ in range(num_xy_envs)]
        ts_epsi_hist = [[] for _ in range(num_xy_envs)]
        # ★追加：d_delta_ref（ステア指令レート）
        ts_ddelta_ref_hist = [[] for _ in range(num_xy_envs)]
        ts_delta_geom_hist = [[] for _ in range(num_xy_envs)]
        ts_s_hist = [[] for _ in range(num_xy_envs)]

    ep_returns = []
    ep_lengths = []
    ep_ey_mean = []
    ep_epsi_mean = []
    ep_dv_mean = []
    ep_cost_y_mean = []
    ep_cost_psi_mean = []
    ep_cost_v_mean = []
    ep_cost_ay_mean = []
    # ★追加：d_delta_ref の誤差（|d_delta_ref|）とコスト
    ep_ddelta_ref_mean = []
    ep_cost_ddelta_ref_mean = []

    for ep in range(num_episodes):
        obs, _, state = env.reset(init_state, is_perturbed=False)  # ★ reset は (obs_norm, obs_raw, state)
        done = torch.zeros(B_eval, dtype=torch.bool, device=env.device)

        ep_ret = torch.zeros(B_eval, dtype=torch.float32, device=env.device)
        ep_len = torch.zeros(B_eval, dtype=torch.int32, device=env.device)

        ey_sum = torch.zeros(B_eval, dtype=torch.float32, device=env.device)
        epsi_sum = torch.zeros(B_eval, dtype=torch.float32, device=env.device)
        dv_sum = torch.zeros(B_eval, dtype=torch.float32, device=env.device)

        cost_y_sum = torch.zeros(B_eval, dtype=torch.float32, device=env.device)
        cost_psi_sum = torch.zeros(B_eval, dtype=torch.float32, device=env.device)
        cost_v_sum = torch.zeros(B_eval, dtype=torch.float32, device=env.device)
        cost_ay_sum = torch.zeros(B_eval, dtype=torch.float32, device=env.device)

        # ★追加
        ddelta_ref_sum = torch.zeros(B_eval, dtype=torch.float32, device=env.device)
        cost_ddelta_ref_sum = torch.zeros(B_eval, dtype=torch.float32, device=env.device)

        for t in range(max_steps):
            actions, log_probs, values = agent.act_batch(obs)
            next_obs, next_obs_raw, next_state, reward, done_step, info = env.step(
                actions, compute_info=True
            )

            delta_geom = info.get("delta_geom", None)  # (B,) 期待

            not_done = ~done

            # 累積報酬 / 長さ
            ep_ret[not_done] += reward[not_done]
            ep_len[not_done] += 1

            # 誤差
            ey = next_obs_raw[:, 0]        # e_y
            epsi = next_obs_raw[:, 1]      # e_psi_v
            v = next_obs_raw[:, 2]         # v [m/s]
            a_long = next_obs_raw[:, 3]    # 縦加速度 a_x
            v_ref_now = info["v_ref"]      # (B,)
            a_lat = info["a_y"]            # (B,) 横加速度 a_y
            s = info["s"]

            # ★ 舵角は obs_raw の index ではなく vehicle.state から取る（obs構成変更に強い）
            delta = env.vehicle.state[:, 5]  # actual delta [rad]

            dv = (v - v_ref_now).abs()

            ey_sum[not_done] += ey[not_done].abs()
            epsi_sum[not_done] += epsi[not_done].abs()
            dv_sum[not_done] += dv[not_done]

            # コスト
            cost_y = info["cost_y"]
            cost_psi = info["cost_psi"]
            cost_v = info["cost_v"]
            cost_ay = info["cost_ay"]

            cost_y_sum[not_done] += cost_y[not_done]
            cost_psi_sum[not_done] += cost_psi[not_done]
            cost_v_sum[not_done] += cost_v[not_done]
            cost_ay_sum[not_done] += cost_ay[not_done]

            # ★追加：d_delta_ref 誤差（|d_delta_ref|）とコスト
            if "d_delta_ref" in info:
                d_delta_ref = info["d_delta_ref"]
            else:
                d_delta_ref = actions[:, 1]
            if "cost_d_delta_ref" in info:
                cost_d_delta_ref = info["cost_d_delta_ref"]
            else:
                cost_d_delta_ref = torch.zeros_like(reward)

            ddelta_ref_sum[not_done] += d_delta_ref[not_done].abs()
            cost_ddelta_ref_sum[not_done] += cost_d_delta_ref[not_done]

            # ---- XY & 時系列ログ（ep==0 のときだけ & その env がまだ初回エピソード未完了なら）----
            if save_xy and ep == 0:
                veh_state = env.vehicle.state  # (B, 9) 想定（delta_ref を含む）

                for b in range(num_xy_envs):
                    if not xy_done[b]:
                        # XY
                        xy_x_hist[b].append(veh_state[b, 0].item())
                        xy_y_hist[b].append(veh_state[b, 1].item())

                        # 時系列ログ
                        ts_v_hist[b].append(v[b].item())
                        ts_vref_hist[b].append(v_ref_now[b].item())
                        ts_ax_hist[b].append(a_long[b].item())
                        ts_ay_hist[b].append(a_lat[b].item())
                        ts_delta_hist[b].append(delta[b].item())
                        ts_ey_hist[b].append(ey[b].item())
                        ts_epsi_hist[b].append(epsi[b].item())
                        # ★追加
                        ts_ddelta_ref_hist[b].append(d_delta_ref[b].item())
                        if delta_geom is not None:
                            ts_delta_geom_hist[b].append(delta_geom[b].item())
                        ts_s_hist[b].append(info["s"][b].item())

            done = done | done_step

            # partial reset で数値安定（ただし done は維持）
            if use_partial_reset and done_step.any():
                obs, _, _ = env.partial_reset(
                    init_state=init_state,
                    done_mask=done_step,
                    is_perturbed=False,
                    regenerate_traj=False,
                )
            else:
                obs = next_obs

            # XY/時系列ログ側の done 状態も更新（1エピソード分だけ取りたい）
            if save_xy and ep == 0:
                xy_done |= done_step

            if done.all():
                break

        mask_valid = ep_len > 0
        ep_returns.append(ep_ret[mask_valid].cpu())
        ep_lengths.append(ep_len[mask_valid].cpu())
        ep_ey_mean.append((ey_sum[mask_valid] / ep_len[mask_valid].float()).cpu())
        ep_epsi_mean.append((epsi_sum[mask_valid] / ep_len[mask_valid].float()).cpu())
        ep_dv_mean.append((dv_sum[mask_valid] / ep_len[mask_valid].float()).cpu())
        ep_cost_y_mean.append((cost_y_sum[mask_valid] / ep_len[mask_valid].float()).cpu())
        ep_cost_psi_mean.append((cost_psi_sum[mask_valid] / ep_len[mask_valid].float()).cpu())
        ep_cost_v_mean.append((cost_v_sum[mask_valid] / ep_len[mask_valid].float()).cpu())
        ep_cost_ay_mean.append((cost_ay_sum[mask_valid] / ep_len[mask_valid].float()).cpu())

        ep_ddelta_ref_mean.append((ddelta_ref_sum[mask_valid] / ep_len[mask_valid].float()).cpu())
        ep_cost_ddelta_ref_mean.append((cost_ddelta_ref_sum[mask_valid] / ep_len[mask_valid].float()).cpu())

    # ---- 集計 ----
    ep_returns = torch.cat(ep_returns)
    ep_lengths = torch.cat(ep_lengths)
    ep_ey_mean = torch.cat(ep_ey_mean)
    ep_epsi_mean = torch.cat(ep_epsi_mean)
    ep_dv_mean = torch.cat(ep_dv_mean)
    ep_cost_y_mean = torch.cat(ep_cost_y_mean)
    ep_cost_psi_mean = torch.cat(ep_cost_psi_mean)
    ep_cost_v_mean = torch.cat(ep_cost_v_mean)
    ep_cost_ay_mean = torch.cat(ep_cost_ay_mean)

    ep_ddelta_ref_mean = torch.cat(ep_ddelta_ref_mean)
    ep_cost_ddelta_ref_mean = torch.cat(ep_cost_ddelta_ref_mean)

    ep_return_per_step = ep_returns / ep_lengths.float()

    result = {
        "return_mean": ep_returns.mean().item(),
        "return_std": ep_returns.std().item(),
        "return_per_step_mean": ep_return_per_step.mean().item(),
        "return_per_step_std": ep_return_per_step.std().item(),
        "len_mean": ep_lengths.float().mean().item(),
        "ey_mean": ep_ey_mean.mean().item(),
        "e_psi_v_mean": ep_epsi_mean.mean().item(),
        "dv_mean": ep_dv_mean.mean().item(),
        "cost_y_mean": ep_cost_y_mean.mean().item(),
        "cost_psi_mean": ep_cost_psi_mean.mean().item(),
        "cost_v_mean": ep_cost_v_mean.mean().item(),
        "cost_ay_mean": ep_cost_ay_mean.mean().item(),
        # ★追加：d_delta_ref の“誤差”とコスト
        "d_delta_ref_abs_mean": ep_ddelta_ref_mean.mean().item(),
        "cost_d_delta_ref_mean": ep_cost_ddelta_ref_mean.mean().item(),
    }

    # ---- XY プロット保存（4 走行を1画像に） + 時系列プロット ----
    if save_xy:
        import os
        import math
        import matplotlib.pyplot as plt

        # XY 4本を 1画像
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        axes = axes.ravel()

        for i in range(num_xy_envs):
            ax = axes[i]
            traj = ref_trajs[i]
            ax.plot(traj.x_ref, traj.y_ref, "--", label=f"ref {i}")
            ax.plot(xy_x_hist[i], xy_y_hist[i], label=f"traj {i}")
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            ax.set_title(f"Eval XY env {i}")
            ax.axis("equal")
            ax.grid(True)
            ax.legend()

        plt.tight_layout()
        if xy_filename is None:
            xy_filename = "ppo_eval_xy.png"
        plt.savefig(xy_filename, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved eval XY plot: {xy_filename}")

        # ---- 追加：各走行ごとの時系列プロット（4枚）----
        dt_local = float(env.dt)
        base, ext = os.path.splitext(xy_filename)
        if ext == "":
            ext = ".png"

        for i in range(num_xy_envs):
            T = len(ts_v_hist[i])
            if T == 0:
                continue
            t_axis = ts_s_hist[i]  # ← s 軸

            # ★ 7 段プロット: v, a_x, a_y, delta, e_y, e_psi_v, d_delta_ref
            fig, axes = plt.subplots(7, 1, figsize=(10, 16), sharex=True)

            # 1) 車速 & 目標車速
            axes[0].plot(t_axis, ts_v_hist[i], label="v [m/s]")
            axes[0].plot(t_axis, ts_vref_hist[i], label="v_ref [m/s]")
            axes[0].set_ylabel("Speed")
            axes[0].grid(True)
            axes[0].legend(loc="upper right")

            # 2) 縦加速度
            axes[1].plot(t_axis, ts_ax_hist[i], label="a_x [m/s^2]")
            axes[1].set_ylabel("Long. accel")
            axes[1].grid(True)
            axes[1].legend(loc="upper right")

            # 3) 横加速度
            axes[2].plot(t_axis, ts_ay_hist[i], label="a_y [m/s^2]")
            axes[2].set_ylabel("Lat. accel")
            axes[2].grid(True)
            axes[2].legend(loc="upper right")

            # 4) 舵角（表示名はそのまま）
            delta_deg = [math.degrees(d) for d in ts_delta_hist[i]]
            axes[3].plot(t_axis, delta_deg, label="delta [deg]")

            # ★追加：delta_geom を重ねる
            if len(ts_delta_geom_hist[i]) == len(t_axis) and len(ts_delta_geom_hist[i]) > 0:
                delta_geom_deg = [math.degrees(d) for d in ts_delta_geom_hist[i]]
                axes[3].plot(t_axis, delta_geom_deg, label="delta_geom [deg]")

            axes[3].set_ylabel("Steering")
            axes[3].set_ylim(-0.5, 0.5)
            axes[3].grid(True)
            axes[3].legend(loc="upper right")

            # 5) 横偏差 e_y
            axes[4].plot(t_axis, ts_ey_hist[i], label="e_y [m]")
            axes[4].set_ylabel("e_y")
            axes[4].grid(True)
            axes[4].legend(loc="upper right")

            # 6) 速度方向偏差 e_psi_v
            axes[5].plot(t_axis, ts_epsi_hist[i], label="e_psi_v [rad]")
            axes[5].set_ylabel("e_psi_v")
            axes[5].grid(True)
            axes[5].legend(loc="upper right")

            # ★7) d_delta_ref（ステア指令レート）
            ddelta_deg_s = [math.degrees(d) for d in ts_ddelta_ref_hist[i]]
            axes[6].plot(t_axis, ddelta_deg_s, label="d_delta_ref [deg/s]")
            axes[6].set_xlabel("time [s]")
            axes[6].set_ylabel("d_delta_ref")
            axes[6].grid(True)
            axes[6].legend(loc="upper right")

            fig.suptitle(f"Eval time-series (env {i})")
            plt.tight_layout()

            ts_filename = f"{base}_env{i}_timeseries{ext}"
            plt.savefig(ts_filename, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Saved eval time-series plot: {ts_filename}")

    return result


# ===== 学習ループ =====
num_updates = 1000
eval_interval = 10
plot_env_idx = 0  # プロットする env index（学習環境側）

for update in range(num_updates):
    if EVAL_MODE == False:
        agent.buffer.reset()
        obs_t = obs.clone()

        # 10 回に 1 回 x-y ログ
        log_xy = ((update + 1) % 10 == 0)
        if log_xy:
            x_hist = []
            y_hist = []

        for t in range(agent.rollout_steps):
            actions, log_probs, values = agent.act_batch(obs_t)
            next_obs, next_obs_raw, next_state, reward, done_step, info = train_env.step(
                actions, compute_info=False
            )

            agent.store_step(
                obs_batch=obs_t,
                actions=actions,
                rewards=reward,
                dones=done_step,
                values=values,
                log_probs=log_probs,
            )

            # partial reset
            if done_step.any():
                obs_t, _, _ = train_env.partial_reset(
                    init_state=init_state,
                    done_mask=done_step,
                    is_perturbed=is_perturbed,
                    regenerate_traj=True,
                )
            else:
                obs_t = next_obs

        # PPO 更新
        agent.update(last_obs=obs_t, last_done=done_step)
        obs = obs_t.detach()

    # ===== 評価 =====
    if (update + 1) % eval_interval == 0:
        eval_stats = evaluate_policy(
            agent,
            eval_env,
            eval_init_state,
            ref_trajs=eval_ref_trajs,
            num_episodes=3,
            max_steps=3000,
            use_partial_reset=True,
            save_xy=True,
            xy_filename=f"ppo_eval_xy_update_{update+1:03d}.png",
            num_xy_envs=4,
        )

        print(f"\n========== Eval {update+1} ==========")
        print(f"  return_mean          : {eval_stats['return_mean']:.3f}")
        print(f"  return_per_step_mean : {eval_stats['return_per_step_mean']:.4f}")
        print(f"  episode_len_mean     : {eval_stats['len_mean']:.1f}")
        print(f"  mean |e_y|           : {eval_stats['ey_mean']:.4f}")
        print(f"  mean |e_psi_v|       : {eval_stats['e_psi_v_mean']:.4f}")
        print(f"  mean |Δv|            : {eval_stats['dv_mean']:.4f}")
        print(f"  mean cost_y          : {eval_stats['cost_y_mean']:.6f}")
        print(f"  mean cost_psi        : {eval_stats['cost_psi_mean']:.6f}")
        print(f"  mean cost_v          : {eval_stats['cost_v_mean']:.6f}")
        print(f"  mean cost_ay         : {eval_stats['cost_ay_mean']:.6f}")
        print(f"  mean cost_d_delta_ref: {eval_stats['cost_d_delta_ref_mean']:.6f}")
        print(f"==============================\n")

        if EVAL_MODE:
            break

        # 例）「平均ステップ数が max_steps 近くまで伸びたら」セーブしたい場合
        if eval_stats["len_mean"] > eval_env.max_steps - 30:
            save_checkpoint(agent, update=update+1)
