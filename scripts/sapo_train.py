
"""
SAPO training script (batched, differentiable environment).

This script is designed to coexist with the existing PPO training codebase.
It uses:
  - environment_differentiable_batched.BatchedPathTrackingEnvFrenetDifferentiable
  - sapo_agent_batched.SAPOAgentBatched

Notes:
  - The environment must provide a differentiable `functional_step` for actor updates.
  - Critic/alpha updates are performed on detached rollout data (stopgrad), per SAPO.

Run example:
  python sapo_train.py --device cuda --num-envs 128 --updates 2000

"""
from __future__ import annotations

import os
import time
import json
import argparse
import datetime
import math
from dataclasses import asdict
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import torch

# If your project keeps sources under ../src like the existing ppo_test.py,
# keep this path tweak. Otherwise, remove it and rely on PYTHONPATH.
import sys
sys.path.append("../src")

from bicycle_model_differentiable_batched import VehicleParams
from environment_differentiable_batched import BatchedPathTrackingEnvFrenetDifferentiable, RewardWeights
from sapo_agent_batched import SAPOAgentBatched, SAPOConfig

try:
    from trajectory_generator import (
        generate_random_reference_trajectory_arc_mix,
        calculate_max_curvature_rates,
    )
except Exception as e:  # pragma: no cover
    generate_random_reference_trajectory_arc_mix = None
    calculate_max_curvature_rates = None
    _traj_import_error = e


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_train_traj(
    *,
    dt: float,
    total_length: float,
    ds: float,
    v_min_kph: float,
    v_max_kph: float,
    R_min: float,
    R_max: float,
    seed: Optional[int] = None,
):
    if generate_random_reference_trajectory_arc_mix is None:
        raise RuntimeError(
            "trajectory_generator is not available. "
            "Make sure trajectory_generator.py is on PYTHONPATH (e.g., under ../src). "
            f"Import error: {_traj_import_error}"
        )
    return generate_random_reference_trajectory_arc_mix(
        total_length=total_length,
        ds=ds,
        dt=dt,
        v_min_kph=v_min_kph,
        v_max_kph=v_max_kph,
        R_min=R_min,
        R_max=R_max,
        seed=seed,
    )


def make_init_state(ref_trajs, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Matches ppo_test.py: init at the first point, heading from first segment.
    Returns (B, 8) vehicle state: [x,y,psi,v, beta, r, a, delta] (others 0)
    """
    B = len(ref_trajs)
    init_state = torch.zeros(B, 8, dtype=dtype, device=device)
    for b, traj in enumerate(ref_trajs):
        x_ref = traj.x_ref
        y_ref = traj.y_ref
        x0 = float(x_ref[0])
        y0 = float(y_ref[0])

        # heading from first segment
        dx = float(x_ref[1] - x_ref[0])
        dy = float(y_ref[1] - y_ref[0])
        psi0 = float(torch.atan2(torch.tensor(dy, dtype=dtype), torch.tensor(dx, dtype=dtype)))
        v0 = float(traj.v_ref[0])

        init_state[b, 0] = x0
        init_state[b, 1] = y0
        init_state[b, 2] = psi0
        init_state[b, 3] = v0
    return init_state


def build_action_limits(veh_params: VehicleParams, *, limit_accel: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    SAPO agent uses action squashing into [action_min, action_max].
    Action convention: [a_ref, delta_ref_cmd] (rad).
    """
    a_min = float(getattr(veh_params, "min_accel", -6.0))
    a_max = float(getattr(veh_params, "max_accel", 3.0))
    max_steer = float(getattr(veh_params, "max_steer", np.deg2rad(getattr(veh_params, "max_steer_deg", 30.0))))

    if limit_accel is not None:
        a_min = max(a_min, -float(limit_accel))
        a_max = min(a_max, float(limit_accel))

    action_min = np.array([a_min, -max_steer], dtype=np.float32)
    action_max = np.array([a_max,  max_steer], dtype=np.float32)
    return action_min, action_max


def save_checkpoint(
    *,
    ckpt_path: str,
    agent: SAPOAgentBatched,
    update: int,
    args: argparse.Namespace,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    payload: Dict[str, Any] = {
        "update": update,
        "args": vars(args),
        "sapo_cfg": asdict(agent.cfg),
        "actor": agent.actor.state_dict(),
        "critic1": agent.critic1.state_dict(),
        "critic2": agent.critic2.state_dict(),
        "log_alpha": agent.log_alpha.detach().cpu(),
        "opt_actor": agent.actor_opt.state_dict(),
        "opt_critic": agent.critic_opt.state_dict(),
        "opt_alpha": agent.alpha_opt.state_dict(),
    }
    if extra:
        payload["extra"] = extra
    torch.save(payload, ckpt_path)


def load_checkpoint(
    ckpt_path: str,
    agent: SAPOAgentBatched,
    map_location: str = "cpu",
) -> Dict[str, Any]:
    ckpt = torch.load(ckpt_path, map_location=map_location)
    agent.actor.load_state_dict(ckpt["actor"])
    agent.critic1.load_state_dict(ckpt["critic1"])
    agent.critic2.load_state_dict(ckpt["critic2"])
    with torch.no_grad():
        agent.log_alpha.copy_(torch.as_tensor(ckpt["log_alpha"], device=agent.device, dtype=agent.dtype))
    agent.actor_opt.load_state_dict(ckpt["opt_actor"])
    agent.critic_opt.load_state_dict(ckpt["opt_critic"])
    agent.alpha_opt.load_state_dict(ckpt["opt_alpha"])
    return ckpt


@torch.no_grad()
def evaluate_policy(
    env: BatchedPathTrackingEnvFrenetDifferentiable,
    agent: SAPOAgentBatched,
    *,
    steps: int = 1000,
) -> Dict[str, float]:
    """
    Quick batched evaluation: deterministic actor (mean action), roll for `steps`.
    Returns mean return per env and some simple stats.
    """
    B = env.B
    # Clone env state by resetting to the same initial condition used in training
    # (caller should do env.reset prior to calling evaluate if they want a specific start)
    obs_raw, _state_vec, _cache = env.compute_obs_state(env.get_env_state(), return_cache=True)
    obs = env.normalize_obs(obs_raw)

    returns = torch.zeros(B, device=env.device, dtype=env.dtype)
    lengths = torch.zeros(B, device=env.device, dtype=env.dtype)
    alive = torch.ones(B, device=env.device, dtype=env.dtype)

    for _ in range(int(steps)):
        a = agent.actor.act(obs, deterministic=True)
        obs, _obs_raw, _state_vec, r, done, _info = env.step(a, compute_info=False)
        returns += alive * r
        lengths += alive
        alive = alive * (1.0 - 1.0*done)

        # if all done, stop early
        if float(alive.sum().item()) <= 0.0:
            break

    # Avoid divide-by-zero
    len_mean = float(lengths.mean().item())
    ret_mean = float(returns.mean().item())
    ret_per_step = float((returns / torch.clamp(lengths, min=1.0)).mean().item())
    done_frac = float((1.0 - 1.0*alive).mean().item())

    return {
        "return_mean": ret_mean,
        "return_per_step_mean": ret_per_step,
        "len_mean": len_mean,
        "done_frac": done_frac,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    parser.add_argument("--seed", type=int, default=0)

    # env / traj
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--traj-length", type=float, default=2000.0)
    parser.add_argument("--ds", type=float, default=1.0)
    parser.add_argument("--v-min-kph", type=float, default=50.0)
    parser.add_argument("--v-max-kph", type=float, default=80.0)
    parser.add_argument("--R-min", type=float, default=100.0)
    parser.add_argument("--R-max", type=float, default=500.0)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--kappa-preview", type=int, default=21)

    # reward weights (defaults match ppo_test STAGE3-ish)
    parser.add_argument("--w-y", type=float, default=0.1)
    parser.add_argument("--w-psi", type=float, default=10.0)
    parser.add_argument("--w-v-under", type=float, default=0.5)
    parser.add_argument("--w-v-over", type=float, default=1.5)
    parser.add_argument("--w-ay", type=float, default=0.001)
    parser.add_argument("--w-d-delta-ref", type=float, default=10.0)
    parser.add_argument("--w-dd-delta-ref", type=float, default=0.0)

    # SAPO hyperparams
    parser.add_argument("--updates", type=int, default=2000)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--td-lambda", type=float, default=0.95)
    parser.add_argument("--critic-updates", type=int, default=16)
    parser.add_argument("--minibatch-size", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=256)

    parser.add_argument("--lr-actor", type=float, default=3e-4)
    parser.add_argument("--lr-critic", type=float, default=3e-4)
    parser.add_argument("--lr-alpha", type=float, default=3e-4)

    parser.add_argument("--target-entropy", type=float, default=None)
    parser.add_argument("--alpha-init", type=float, default=0.2)
    parser.add_argument("--actor-grad-clip", type=float, default=5.0)
    parser.add_argument("--critic-grad-clip", type=float, default=10.0)

    # housekeeping
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--eval-interval", type=int, default=20)
    parser.add_argument("--eval-steps", type=int, default=1000)
    parser.add_argument("--reset-interval", type=int, default=0, help="if >0, reset env every N updates")
    parser.add_argument("--ckpt-interval", type=int, default=50)
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints_sapo")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--run-id", type=str, default=None)

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    set_seed(args.seed)

    os.makedirs(args.ckpt_dir, exist_ok=True)

    run_id = args.run_id
    if run_id is None:
        run_id = datetime.datetime.now().strftime("sapo_%Y%m%d_%H%M%S")

    # ---- Build reward weights ----
    weights = RewardWeights(
        w_y=float(args.w_y),
        w_psi=float(args.w_psi),
        w_v_under=float(args.w_v_under),
        w_v_over=float(args.w_v_over),
        w_ay=float(args.w_ay),
        w_d_delta_ref=float(args.w_d_delta_ref),
        w_dd_delta_ref=float(args.w_dd_delta_ref),
        loss_y="l2",
        loss_psi="l2",
        loss_v="l2",
        loss_ay="l2",
        loss_d_delta_ref="l2",
        loss_dd_delta_ref="l2",
    )

    veh_params = VehicleParams()

    # ---- Build reference trajectories ----
    ref_trajs = [
        make_train_traj(
            dt=float(args.dt),
            total_length=float(args.traj_length),
            ds=float(args.ds),
            v_min_kph=float(args.v_min_kph),
            v_max_kph=float(args.v_max_kph),
            R_min=float(args.R_min),
            R_max=float(args.R_max),
            seed=None,
        )
        for _ in range(int(args.num_envs))
    ]

    kappa_preview_offsets = [float(i) for i in range(int(args.kappa_preview))]

    env = BatchedPathTrackingEnvFrenetDifferentiable(
        ref_trajs=ref_trajs,
        kappa_preview_offsets=kappa_preview_offsets,
        vehicle_params=veh_params,
        reward_weights=weights,
        max_steps=int(args.max_steps),
        device=str(device),
        dtype=dtype,
        angle_wrap_mode="atan2",
    )

    init_state = make_init_state(ref_trajs, device=device, dtype=dtype)
    obs_norm, _obs_raw, _state_vec = env.reset(init_state, is_perturbed=False)
    obs_dim = int(obs_norm.shape[1])
    action_dim = 2

    # action bounds
    action_min, action_max = build_action_limits(veh_params)

    # ---- SAPO agent ----
    sapo_cfg = SAPOConfig(
        horizon=int(args.horizon),
        gamma=float(args.gamma),
        td_lambda=float(args.td_lambda),
        critic_updates=int(args.critic_updates),
        minibatch_size=int(args.minibatch_size),
        hidden_dim=int(args.hidden_dim),
        actor_lr=float(args.lr_actor),
        critic_lr=float(args.lr_critic),
        alpha_lr=float(args.lr_alpha),
        target_entropy=(None if args.target_entropy is None else float(args.target_entropy)),
        init_log_alpha=float(math.log(max(float(args.alpha_init), 1e-8))),
        grad_clip=float(args.actor_grad_clip),
    )

    agent = SAPOAgentBatched(
        obs_dim=obs_dim,
        action_dim=action_dim,
        num_envs=int(args.num_envs),
        action_min=action_min,
        action_max=action_max,
        device=str(device),
        dtype=dtype,
        cfg=sapo_cfg,
    )

    start_update = 0
    if args.resume:
        ckpt = load_checkpoint(args.resume, agent, map_location=str(device))
        start_update = int(ckpt.get("update", 0)) + 1
        print(f"[Resume] Loaded checkpoint: {args.resume} (start_update={start_update})")

    # Save a run config for reproducibility
    cfg_path = os.path.join(args.ckpt_dir, f"{run_id}_config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_id": run_id,
                "args": vars(args),
                "sapo_cfg": asdict(agent.cfg),
                "action_min": action_min.tolist(),
                "action_max": action_max.tolist(),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"[Config] Saved: {cfg_path}")

    # ---- Training loop ----
    t0 = time.time()
    for update in range(start_update, int(args.updates)):
        if args.reset_interval and (update % int(args.reset_interval) == 0) and update != start_update:
            env.reset(init_state, is_perturbed=False)

        logs = agent.update(env)

        if (update % int(args.log_interval)) == 0:
            elapsed = time.time() - t0
            msg = (
                f"[{run_id}] upd {update:05d} | "
                f"actor {logs.get('actor_loss', float('nan')):+.4e} | "
                f"critic {logs.get('critic_loss', float('nan')):+.4e} | "
                f"alpha {logs.get('alpha', float('nan')):.4f} | "
                f"H {logs.get('mean_entropy', float('nan')):.3f} | "
                f"r {logs.get('mean_reward', float('nan')):.4f} | "
                f"{elapsed/60.0:.1f} min"
            )
            print(msg)

        if args.eval_interval and (update % int(args.eval_interval) == 0) and update != start_update:
            # evaluate from a fresh reset for comparability
            env.reset(init_state, is_perturbed=False)
            stats = evaluate_policy(env, agent, steps=int(args.eval_steps))
            print(
                f"[Eval] upd {update:05d} | "
                f"ret {stats['return_mean']:.3f} | "
                f"ret/step {stats['return_per_step_mean']:.4f} | "
                f"len {stats['len_mean']:.1f} | "
                f"done_frac {stats['done_frac']:.2f}"
            )

        if args.ckpt_interval and (update % int(args.ckpt_interval) == 0) and update != start_update:
            ckpt_path = os.path.join(args.ckpt_dir, f"{run_id}_update{update:05d}.pt")
            save_checkpoint(ckpt_path=ckpt_path, agent=agent, update=update, args=args, extra={"run_id": run_id})
            print(f"[Checkpoint] Saved: {ckpt_path}")

    # final checkpoint
    ckpt_path = os.path.join(args.ckpt_dir, f"{run_id}_final.pt")
    save_checkpoint(ckpt_path=ckpt_path, agent=agent, update=int(args.updates) - 1, args=args, extra={"run_id": run_id})
    print(f"[Checkpoint] Saved final: {ckpt_path}")


if __name__ == "__main__":
    main()
