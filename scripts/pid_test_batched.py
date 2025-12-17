import numpy as np
import matplotlib
matplotlib.use("Agg")  # GUIバックエンドを使わず画像として保存
import matplotlib.pyplot as plt
import torch

import sys
sys.path.append("../src")

from bicycle_model_batched import VehicleParams
from environment_batched import BatchedPathTrackingEnvFrenet   # 名前は環境に合わせて
from pid_policy_batched import BatchedPIDPolicy                 # さっき定義したバッチPID
from trajectory_generator import generate_random_reference_trajectory_arc_mix


# =========================
#  設定
# =========================
dt = 0.05
BATCH_SIZE = 4
device = "cpu"
dtype = torch.float32

# =========================
#  ランダム経路生成（バッチ分）
# =========================
ref_trajs = [
    generate_random_reference_trajectory_arc_mix(
        total_length=2000.0,
        ds=1.0,
        dt=dt,
        v_min_kph=20.0,
        v_max_kph=60.0,
        R_min=50.0,
        R_max=400.0,
        # seed=i,   # バッチごとに違うシード
        seed=None,
    )
    for i in range(BATCH_SIZE)
]

veh_params = VehicleParams()

# =========================
#  バッチ環境・PIDポリシー生成
# =========================
env = BatchedPathTrackingEnvFrenet(
    ref_trajs=ref_trajs,
    vehicle_params=veh_params,
    max_steps=2000,
    device=device,
    dtype=dtype,
)

# 初期状態（各バッチで経路の s=0 の位置からスタート）
init_state_np = np.zeros((BATCH_SIZE, 8), dtype=np.float32)  # [x,y,psi,v,a,delta,beta,r]

for b, traj in enumerate(ref_trajs):
    x_ref = traj.x_ref
    y_ref = traj.y_ref
    # s=0 の点とその接線方向
    x0 = x_ref[0]
    y0 = y_ref[0]
    dx = x_ref[1] - x_ref[0]
    dy = y_ref[1] - y_ref[0]
    psi0 = float(np.arctan2(dy, dx))
    v0 = float(traj.v_ref[0])

    init_state_np[b, 0] = x0
    init_state_np[b, 1] = y0
    init_state_np[b, 2] = psi0
    init_state_np[b, 3] = v0
    # a, delta, beta, r は 0 のまま

init_state = torch.as_tensor(init_state_np, dtype=dtype, device=device)

# env.reset: Frenetの e_y, e_psi_v にノイズを入れるかどうか
# obs, state = env.reset(init_state, is_perturbed=True)
obs, state = env.reset(init_state, is_perturbed=False)

policy = BatchedPIDPolicy(
    vehicle_params=veh_params,
    dt=dt,
    batch_size=BATCH_SIZE,
    device=device,
    dtype=dtype,
)

policy.reset()  # 内部PID状態初期化

# =========================
#  シミュレーションループ
# =========================
t_hist = []                          # 共通の時間
x_hist = [[] for _ in range(BATCH_SIZE)]
y_hist = [[] for _ in range(BATCH_SIZE)]
v_hist = [[] for _ in range(BATCH_SIZE)]
a_hist = [[] for _ in range(BATCH_SIZE)]
delta_hist = [[] for _ in range(BATCH_SIZE)]
beta_hist = [[] for _ in range(BATCH_SIZE)]
r_hist = [[] for _ in range(BATCH_SIZE)]
ey_hist = [[] for _ in range(BATCH_SIZE)]
epsi_hist = [[] for _ in range(BATCH_SIZE)]

reward_hist = [[] for _ in range(BATCH_SIZE)]
cost_y_hist = [[] for _ in range(BATCH_SIZE)]
cost_psi_hist = [[] for _ in range(BATCH_SIZE)]
cost_v_hist = [[] for _ in range(BATCH_SIZE)]
cost_ay_hist = [[] for _ in range(BATCH_SIZE)]

done = torch.zeros(BATCH_SIZE, dtype=torch.bool, device=device)
step = 0
max_steps_safety = 10000

while (not bool(done.all())) and step < max_steps_safety:
    t = step * dt
    t_hist.append(t)

    veh_state = env.vehicle.state  # (B, 8) [x,y,psi,v,a,delta,beta,r]
    x = veh_state[:, 0].tolist()
    y = veh_state[:, 1].tolist()
    v = veh_state[:, 3].tolist()
    a = veh_state[:, 4].tolist()
    delta = veh_state[:, 5].tolist()
    beta = veh_state[:, 6].tolist()
    r = veh_state[:, 7].tolist()

    e_y = obs[:, 0].tolist()        # obs[:, IDX_EY]
    e_psi_v = obs[:, 1].tolist()    # obs[:, IDX_EPSI_V]

    # ログに追加
    for b in range(BATCH_SIZE):
        x_hist[b].append(x[b])
        y_hist[b].append(y[b])
        v_hist[b].append(v[b])
        a_hist[b].append(a[b])
        delta_hist[b].append(delta[b])
        beta_hist[b].append(beta[b])
        r_hist[b].append(r[b])
        ey_hist[b].append(e_y[b])
        epsi_hist[b].append(e_psi_v[b])

    # policy で行動決定（バッチ）
    actions = policy.act(obs)  # (B,2) [a_ref, delta_ref]

    # env 1ステップ
    obs, state, reward, done_step, info = env.step(actions, compute_info=True)

    # 報酬/コストログ
    reward_list = reward.tolist()
    cost_y_list = info["cost_y"].tolist()
    cost_psi_list = info["cost_psi"].tolist()
    cost_v_list = info["cost_v"].tolist()
    cost_ay_list = info["cost_ay"].tolist()

    for b in range(BATCH_SIZE):
        reward_hist[b].append(reward_list[b])
        cost_y_hist[b].append(cost_y_list[b])
        cost_psi_hist[b].append(cost_psi_list[b])
        cost_v_hist[b].append(cost_v_list[b])
        cost_ay_hist[b].append(cost_ay_list[b])

    # done 更新（OR をとって今までの done も保持）
    done = done | done_step

    # 終了した env の PID 内部状態だけリセットしておきたいなら
    if done_step.any():
        policy.reset(mask=done_step)

    step += 1

# ログを numpy 配列化（時間は共通）
t_hist = np.array(t_hist)
# 各バッチは shape (T,)
x_hist = [np.array(h) for h in x_hist]
y_hist = [np.array(h) for h in y_hist]
v_hist = [np.array(h) for h in v_hist]
a_hist = [np.array(h) for h in a_hist]
delta_hist = [np.array(h) for h in delta_hist]
beta_hist = [np.array(h) for h in beta_hist]
r_hist = [np.array(h) for h in r_hist]
ey_hist = [np.array(h) for h in ey_hist]
epsi_hist = [np.array(h) for h in epsi_hist]

reward_hist = [np.array(h) for h in reward_hist]
cost_y_hist = [np.array(h) for h in cost_y_hist]
cost_psi_hist = [np.array(h) for h in cost_psi_hist]
cost_v_hist = [np.array(h) for h in cost_v_hist]
cost_ay_hist = [np.array(h) for h in cost_ay_hist]

# =========================
#  図1: x-y 軌跡（バッチ4本）
# =========================
fig1 = plt.figure()
for b, traj in enumerate(ref_trajs):
    plt.plot(traj.x_ref, traj.y_ref, linestyle="--", label=f"ref path {b}")
    plt.plot(x_hist[b], y_hist[b], label=f"vehicle {b}")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.title("Vehicle trajectories (x-y, batch=4)")
plt.axis("equal")
plt.grid(True)
plt.legend()
fig1.tight_layout()
fig1.savefig("traj_xy.png", dpi=150)

# =========================
#  図2: 状態量の時系列（ここでは env 0 を表示）
# =========================
idx_plot = 0  # 可視化するバッチ index（変えたければここをいじる）

fig2 = plt.figure(figsize=(10, 8))

plt.subplot(3, 1, 1)
plt.plot(t_hist, ey_hist[idx_plot], label="e_y [m]")
plt.plot(t_hist, epsi_hist[idx_plot], label="e_psi_v [rad]")
plt.ylabel("tracking errors")
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(t_hist, v_hist[idx_plot], label="v [m/s]")
plt.plot(t_hist, a_hist[idx_plot], label="a [m/s^2]")
plt.ylabel("speed / accel")
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(t_hist, np.rad2deg(delta_hist[idx_plot]), label="delta [deg]")
plt.plot(t_hist, np.rad2deg(beta_hist[idx_plot]), label="beta [deg]")
plt.plot(t_hist, r_hist[idx_plot], label="r [rad/s]")
plt.xlabel("time [s]")
plt.ylabel("angles / yaw rate")
plt.grid(True)
plt.legend()

fig2.tight_layout()
fig2.savefig("states_timeseries_env0.png", dpi=150)

# =========================
#  図3: コスト成分と報酬の時系列（env 0）
# =========================
fig3 = plt.figure(figsize=(10, 6))
plt.plot(t_hist, cost_y_hist[idx_plot], label="cost_y (lateral error)")
plt.plot(t_hist, cost_psi_hist[idx_plot], label="cost_psi (heading error)")
plt.plot(t_hist, cost_v_hist[idx_plot], label="cost_v (speed)")
plt.plot(t_hist, cost_ay_hist[idx_plot], label="cost_ay (lat. accel)")
plt.plot(t_hist, reward_hist[idx_plot], label="reward (negative total cost)")
plt.xlabel("time [s]")
plt.ylabel("cost / reward")
plt.title(f"Cost components and reward (env {idx_plot})")
plt.grid(True)
plt.legend()

fig3.tight_layout()
fig3.savefig("cost_reward_env0.png", dpi=150)

# plt.show() は呼ばない（ヘッドレス環境想定）
