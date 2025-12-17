

import os
import datetime
import torch
import matplotlib.pyplot as plt

import sys
sys.path.append("../src")
from bicycle_model_batched import VehicleParams
from environment_batched import BatchedPathTrackingEnvFrenet, RewardWeights
from trajectory_generator import generate_random_reference_trajectory_arc_mix
from ppo_agent_batched import PPOAgentBatched


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


# ===== 共通設定 =====
dt = 0.05
B = 64              # 学習用バッチサイズ
device = "cuda"
dtype = torch.float32
is_perturbed = False

weights = RewardWeights(
    w_y=0.1,
    w_psi=10.0,
    w_v_under=0.1,
    w_v_over=1.5,
    w_ay=0.001,
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
        R_min=100.0,
        R_max=400.0,
        seed=None,   # reset ごとにランダム
    )

# ===== 学習用 ref_traj / env =====
train_ref_trajs = [
    # generate_random_reference_trajectory_arc_mix(
    #     total_length=2000.0,
    #     ds=1.0,
    #     dt=dt,
    #     v_min_kph=50.0,
    #     v_max_kph=80.0,
    #     R_min=100.0,
    #     R_max=400.0,
    #     seed=i,
    # )
    make_train_traj()
    for i in range(B)
]

train_env = BatchedPathTrackingEnvFrenet(
    ref_trajs=train_ref_trajs,
    vehicle_params=veh_params,
    reward_weights=weights,  # 明示的に渡す
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
obs, state = train_env.reset(init_state, is_perturbed=is_perturbed)
obs_dim = obs.shape[1]
action_dim = 2

# ===== PPO エージェント =====
agent = PPOAgentBatched(
    obs_dim=obs_dim,
    action_dim=action_dim,
    num_envs=B,
    rollout_steps=64,
    device=device,
)

# agent = PPOAgentBatched(
#     obs_dim=obs_dim,
#     action_dim=action_dim,
#     num_envs=B,
#     rollout_steps=64,      # とりあえずそのままでもOK（後で128に増やしても良い）
#     lr=1e-4,               # 3e-4 -> 1e-4 に下げて一歩あたりをマイルドに
#     gamma=0.99,            # そのままでOK
#     lam=0.95,              # そのままでOK
#     clip_eps=0.1,          # 0.2 -> 0.1 にして「大きな方針更新」を抑える
#     epochs=5,              # 10 -> 5 にして過学習＆崩壊を少し抑制
#     batch_size=256,        # 64*64=4096 サンプル / update なので 16 ミニバッチ
#     vf_coef=0.5,           # そのままでOK
#     ent_coef=0.001,        # 0.01 はやや強めなので少し弱める（安定寄り）
#     max_grad_norm=0.5,     # そのままでOK（NaN対策にも効く）
#     device=device,
#     dtype=dtype,
# )


# ===== 評価用 env（小さめバッチ + 固定seed） =====
EVAL_ENVS = 8
eval_ref_trajs = [
    # generate_random_reference_trajectory_arc_mix(
    #     total_length=2000.0,
    #     ds=1.0,
    #     dt=dt,
    #     v_min_kph=50.0,
    #     v_max_kph=80.0,
    #     R_min=100.0,
    #     R_max=400.0,
    #     seed=1000 + i,   # 学習とは別のシード
    # )
    make_train_traj(seed=1000 + i)
    for i in range(EVAL_ENVS)
]

eval_env = BatchedPathTrackingEnvFrenet(
    ref_trajs=eval_ref_trajs,
    vehicle_params=veh_params,
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
        #  - v, v_ref
        #  - a_x（縦加速度）
        #  - a_y（横加速度）
        #  - delta（舵角）
        #  - e_y（横偏差）
        #  - e_psi_v（速度方向偏差）
        ts_v_hist = [[] for _ in range(num_xy_envs)]
        ts_vref_hist = [[] for _ in range(num_xy_envs)]
        ts_ax_hist = [[] for _ in range(num_xy_envs)]
        ts_ay_hist = [[] for _ in range(num_xy_envs)]
        ts_delta_hist = [[] for _ in range(num_xy_envs)]
        ts_ey_hist = [[] for _ in range(num_xy_envs)]
        ts_epsi_hist = [[] for _ in range(num_xy_envs)]

    ep_returns = []
    ep_lengths = []
    ep_ey_mean = []
    ep_epsi_mean = []
    ep_dv_mean = []
    ep_cost_y_mean = []
    ep_cost_psi_mean = []
    ep_cost_v_mean = []
    ep_cost_ay_mean = []

    for ep in range(num_episodes):
        obs, state = env.reset(init_state, is_perturbed=False)
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

        for t in range(max_steps):
            actions, log_probs, values = agent.act_batch(obs)
            next_obs, state, reward, done_step, info = env.step(
                actions, compute_info=True
            )

            not_done = ~done

            # 累積報酬 / 長さ
            ep_ret[not_done] += reward[not_done]
            ep_len[not_done] += 1

            # 誤差
            ey = obs[:, 0]        # e_y
            epsi = obs[:, 1]      # e_psi_v
            v = obs[:, 2]         # v [m/s]
            a_long = obs[:, 3]    # 縦加速度 a_x
            delta = obs[:, 4]     # 舵角
            v_ref_now = info["v_ref"]      # (B,)
            a_lat = info["a_y"]            # (B,) 横加速度 a_y

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

            # ---- XY & 時系列ログ（ep==0 のときだけ & その env がまだ初回エピソード未完了なら）----
            if save_xy and ep == 0:
                veh_state = env.vehicle.state  # (B,8)

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

            done = done | done_step

            # partial reset で数値安定（ただし done は維持）
            if use_partial_reset and done_step.any():
                obs, _ = env.partial_reset(
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
        ep_cost_y_mean.append(
            (cost_y_sum[mask_valid] / ep_len[mask_valid].float()).cpu()
        )
        ep_cost_psi_mean.append(
            (cost_psi_sum[mask_valid] / ep_len[mask_valid].float()).cpu()
        )
        ep_cost_v_mean.append(
            (cost_v_sum[mask_valid] / ep_len[mask_valid].float()).cpu()
        )
        ep_cost_ay_mean.append(
            (cost_ay_sum[mask_valid] / ep_len[mask_valid].float()).cpu()
        )

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
                continue  # 念のため
            t_axis = [k * dt_local for k in range(T)]

            # 6 段プロット: v, a_x, a_y, delta, e_y, e_psi_v
            fig, axes = plt.subplots(6, 1, figsize=(10, 14), sharex=True)

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

            # 4) 舵角
            delta_deg = [math.degrees(d) for d in ts_delta_hist[i]]
            axes[3].plot(t_axis, delta_deg, label="delta [deg]")
            axes[3].set_ylabel("Steering")
            axes[3].grid(True)
            axes[3].legend(loc="upper right")

            # 5) 横偏差 e_y
            axes[4].plot(t_axis, ts_ey_hist[i], label="e_y [m]")
            axes[4].set_ylabel("e_y")
            axes[4].grid(True)
            axes[4].legend(loc="upper right")

            # 6) 速度方向偏差 e_psi_v
            axes[5].plot(t_axis, ts_epsi_hist[i], label="e_psi_v [rad]")
            axes[5].set_xlabel("time [s]")
            axes[5].set_ylabel("e_psi_v")
            axes[5].grid(True)
            axes[5].legend(loc="upper right")

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
    agent.buffer.reset()
    obs_t = obs.clone()

    # 10 回に 1 回 x-y ログ
    log_xy = ((update + 1) % 10 == 0)
    if log_xy:
        x_hist = []
        y_hist = []

    for t in range(agent.rollout_steps):
        actions, log_probs, values = agent.act_batch(obs_t)
        next_obs, state, reward, done_step, info = train_env.step(actions, compute_info=False)

        agent.store_step(
            obs_batch=obs_t,
            actions=actions,
            rewards=reward,
            dones=done_step,
            values=values,
            log_probs=log_probs,
        )

        # # x-y ログ
        # if log_xy:
        #     x_hist.append(train_env.vehicle.state[plot_env_idx, 0].item())
        #     y_hist.append(train_env.vehicle.state[plot_env_idx, 1].item())

        # partial reset
        if done_step.any():
            obs_t, _ = train_env.partial_reset(
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
        print(f"==============================\n")

        # 例）「平均ステップ数が max_steps 近くまで伸びたら」セーブしたい場合
        if eval_stats["len_mean"] > eval_env.max_steps - 30:
            save_checkpoint(agent, update=update+1)
