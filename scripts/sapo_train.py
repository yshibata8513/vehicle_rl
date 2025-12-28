
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



@torch.no_grad()
def collect_env_debug(env: BatchedPathTrackingEnvFrenetDifferentiable) -> Dict[str, float]:
    """Lightweight environment diagnostics (mean/max), useful when training diverges."""
    env_state = env.get_env_state()
    obs_raw, _state_vec, cache = env.compute_obs_state(env_state, return_cache=True)
    v = cache["v"]
    a = cache["a"]
    r = cache["r"]
    e_y = cache["e_y"]
    e_psi_v = cache["e_psi_v"]
    v_ref = cache["v_ref_now"]
    dv = v - v_ref
    # done mask (only depends on state)
    try:
        done = env.get_done_mask()
    except Exception:
        done = (torch.abs(env.e_y) > env.max_lateral_error) | (env.s >= env.s_end) | (env.step_count >= env.max_steps)

    def _mean(x: torch.Tensor) -> float:
        return float(x.detach().mean().cpu().item())

    def _maxabs(x: torch.Tensor) -> float:
        return float(x.detach().abs().max().cpu().item())

    out = {
        "done_frac": _mean(done.to(dtype=env.dtype)),
        "s_mean": _mean(env.s),
        "s_max": float(env.s.detach().max().cpu().item()),
        "ey_abs_mean": _mean(torch.abs(e_y)),
        "ey_abs_max": _maxabs(e_y),
        "epsi_abs_mean": _mean(torch.abs(e_psi_v)),
        "epsi_abs_max": _maxabs(e_psi_v),
        "v_mean": _mean(v),
        "v_max": float(v.detach().max().cpu().item()),
        "dv_abs_mean": _mean(torch.abs(dv)),
        "a_mean": _mean(a),
        "r_abs_mean": _mean(torch.abs(r)),
    }
    return out


def _append_jsonl(path: str, record: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


@torch.no_grad()
def evaluate_policy_detailed(
    agent: SAPOAgentBatched,
    env: BatchedPathTrackingEnvFrenetDifferentiable,
    init_state: torch.Tensor,
    *,
    ref_trajs,
    num_episodes: int = 3,
    max_steps: int = 3000,
    use_partial_reset: bool = True,
    save_xy: bool = False,
    xy_filename: Optional[str] = None,
    num_xy_envs: int = 4,
) -> Dict[str, float]:
    """
    PPO の evaluate_policy と同様に、
      - return / step
      - episode length
      - |e_y|, |e_psi_v|, |Δv|
      - コスト分解 (cost_y, cost_psi, cost_v, cost_ay, cost_d_delta_ref, cost_dd_delta_ref)
    を集計し、必要なら XY と時系列プロットを保存します。
    """
    B = env.B
    obs, _obs_raw, _state_vec = env.reset(init_state, is_perturbed=False)

    # per-env episode counters
    ep_count = torch.zeros(B, dtype=torch.int64, device=env.device)
    finished = torch.zeros(B, dtype=torch.bool, device=env.device)

    # current episode accumulators (per env)
    cur_ret = torch.zeros(B, dtype=env.dtype, device=env.device)
    cur_len = torch.zeros(B, dtype=env.dtype, device=env.device)

    cur_ey = torch.zeros(B, dtype=env.dtype, device=env.device)
    cur_epsi = torch.zeros(B, dtype=env.dtype, device=env.device)
    cur_dv = torch.zeros(B, dtype=env.dtype, device=env.device)

    cur_cost_y = torch.zeros(B, dtype=env.dtype, device=env.device)
    cur_cost_psi = torch.zeros(B, dtype=env.dtype, device=env.device)
    cur_cost_v = torch.zeros(B, dtype=env.dtype, device=env.device)
    cur_cost_ay = torch.zeros(B, dtype=env.dtype, device=env.device)
    cur_cost_ddelta = torch.zeros(B, dtype=env.dtype, device=env.device)
    cur_cost_dddelta = torch.zeros(B, dtype=env.dtype, device=env.device)
    cur_ddelta_abs = torch.zeros(B, dtype=env.dtype, device=env.device)
    cur_dddelta_abs = torch.zeros(B, dtype=env.dtype, device=env.device)

    # episode results (stack later)
    ep_returns: List[torch.Tensor] = []
    ep_lengths: List[torch.Tensor] = []
    ep_ey_mean: List[torch.Tensor] = []
    ep_epsi_mean: List[torch.Tensor] = []
    ep_dv_mean: List[torch.Tensor] = []
    ep_cost_y_mean: List[torch.Tensor] = []
    ep_cost_psi_mean: List[torch.Tensor] = []
    ep_cost_v_mean: List[torch.Tensor] = []
    ep_cost_ay_mean: List[torch.Tensor] = []
    ep_cost_ddelta_mean: List[torch.Tensor] = []
    ep_cost_dddelta_mean: List[torch.Tensor] = []
    ep_ddelta_mean: List[torch.Tensor] = []
    ep_dddelta_mean: List[torch.Tensor] = []

    # ---- XY & time-series logs (first episode only) ----
    num_xy_envs = min(int(num_xy_envs), B)
    if save_xy:
        xy_x_hist = [[] for _ in range(num_xy_envs)]
        xy_y_hist = [[] for _ in range(num_xy_envs)]
        xy_done = torch.zeros(B, dtype=torch.bool, device=env.device)

        ts_v_hist = [[] for _ in range(num_xy_envs)]
        ts_vref_hist = [[] for _ in range(num_xy_envs)]
        ts_ax_hist = [[] for _ in range(num_xy_envs)]
        ts_ay_hist = [[] for _ in range(num_xy_envs)]
        ts_delta_hist = [[] for _ in range(num_xy_envs)]
        ts_delta_ref_hist = [[] for _ in range(num_xy_envs)]
        ts_delta_geom_hist = [[] for _ in range(num_xy_envs)]
        ts_ey_hist = [[] for _ in range(num_xy_envs)]
        ts_epsi_hist = [[] for _ in range(num_xy_envs)]
        ts_ddelta_ref_hist = [[] for _ in range(num_xy_envs)]
        ts_dd_delta_ref_hist = [[] for _ in range(num_xy_envs)]
        ts_s_hist = [[] for _ in range(num_xy_envs)]

    for _t in range(int(max_steps)):
        # Do not accumulate further episodes once we already collected enough for that env.
        finished = ep_count >= int(num_episodes)

        # deterministic policy for evaluation
        action = agent.actor.act(obs, deterministic=True)
        obs_next, _obs_raw_next, _state_vec_next, r, done, info = env.step(action, compute_info=True)

        # normalize shapes
        if r.dim() != 1:
            r_step = r.view(-1)
        else:
            r_step = r
        done_mask = done.bool().view(-1)

        # ignore finished envs for stats; but keep them numerically stable by force-resetting
        active = (~finished).to(dtype=env.dtype)
        cur_ret += active * r_step
        cur_len += active

        # metrics from current state/info (after env.step)
        v = env.vehicle.state[:, 3]
        e_y = env.e_y
        e_psi_v = env.e_psi_v
        v_ref = info.get("v_ref", torch.zeros_like(v))
        dv = v - v_ref

        cur_ey += active * torch.abs(e_y)
        cur_epsi += active * torch.abs(e_psi_v)
        cur_dv += active * torch.abs(dv)

        cur_cost_y += active * info.get("cost_y", torch.zeros_like(v))
        cur_cost_psi += active * info.get("cost_psi", torch.zeros_like(v))
        cur_cost_v += active * info.get("cost_v", torch.zeros_like(v))
        cur_cost_ay += active * info.get("cost_ay", torch.zeros_like(v))
        cur_cost_ddelta += active * info.get("cost_d_delta_ref", torch.zeros_like(v))
        cur_cost_dddelta += active * info.get("cost_dd_delta_ref", torch.zeros_like(v))

        d_delta_ref = info.get("d_delta_ref", torch.zeros_like(v))
        dd_delta_ref = info.get("dd_delta_ref", torch.zeros_like(v))
        cur_ddelta_abs += active * torch.abs(d_delta_ref)
        cur_dddelta_abs += active * torch.abs(dd_delta_ref)

        # XY/time-series (first episode only)
        if save_xy:
            for i in range(num_xy_envs):
                if not xy_done[i]:
                    x = float(env.vehicle.state[i, 0].detach().cpu().item())
                    y = float(env.vehicle.state[i, 1].detach().cpu().item())
                    xy_x_hist[i].append(x)
                    xy_y_hist[i].append(y)

                    ts_v_hist[i].append(float(v[i].detach().cpu().item()))
                    ts_vref_hist[i].append(float(v_ref[i].detach().cpu().item()))
                    ts_ax_hist[i].append(float(env.vehicle.state[i, 4].detach().cpu().item()))
                    ts_ay_hist[i].append(float(info.get("a_y", torch.zeros_like(v))[i].detach().cpu().item()))
                    ts_delta_hist[i].append(float(env.vehicle.state[i, 5].detach().cpu().item()))
                    ts_delta_ref_hist[i].append(float(env.vehicle.state[i, 8].detach().cpu().item()))
                    ts_delta_geom_hist[i].append(float(info.get("delta_geom", torch.zeros_like(v))[i].detach().cpu().item()))
                    ts_ey_hist[i].append(float(e_y[i].detach().cpu().item()))
                    ts_epsi_hist[i].append(float(e_psi_v[i].detach().cpu().item()))
                    ts_ddelta_ref_hist[i].append(float(d_delta_ref[i].detach().cpu().item()))
                    ts_dd_delta_ref_hist[i].append(float(dd_delta_ref[i].detach().cpu().item()))
                    ts_s_hist[i].append(float(env.s[i].detach().cpu().item()))

        # finalize episodes that ended this step (only for active envs)
        done_event = done_mask & (~finished)
        if done_event.any():
            idx = torch.nonzero(done_event, as_tuple=False).view(-1)

            # avoid div-by-zero
            denom = torch.clamp(cur_len[idx], min=1.0)
            ep_returns.append(cur_ret[idx].detach().cpu())
            ep_lengths.append(cur_len[idx].detach().cpu())

            ep_ey_mean.append((cur_ey[idx] / denom).detach().cpu())
            ep_epsi_mean.append((cur_epsi[idx] / denom).detach().cpu())
            ep_dv_mean.append((cur_dv[idx] / denom).detach().cpu())

            ep_cost_y_mean.append((cur_cost_y[idx] / denom).detach().cpu())
            ep_cost_psi_mean.append((cur_cost_psi[idx] / denom).detach().cpu())
            ep_cost_v_mean.append((cur_cost_v[idx] / denom).detach().cpu())
            ep_cost_ay_mean.append((cur_cost_ay[idx] / denom).detach().cpu())
            ep_cost_ddelta_mean.append((cur_cost_ddelta[idx] / denom).detach().cpu())
            ep_cost_dddelta_mean.append((cur_cost_dddelta[idx] / denom).detach().cpu())
            ep_ddelta_mean.append((cur_ddelta_abs[idx] / denom).detach().cpu())
            ep_dddelta_mean.append((cur_dddelta_abs[idx] / denom).detach().cpu())

            # increment episode counter and clear accumulators for those envs
            ep_count[idx] += 1

            cur_ret[idx] = 0.0
            cur_len[idx] = 0.0
            cur_ey[idx] = 0.0
            cur_epsi[idx] = 0.0
            cur_dv[idx] = 0.0
            cur_cost_y[idx] = 0.0
            cur_cost_psi[idx] = 0.0
            cur_cost_v[idx] = 0.0
            cur_cost_ay[idx] = 0.0
            cur_cost_ddelta[idx] = 0.0
            cur_cost_dddelta[idx] = 0.0
            cur_ddelta_abs[idx] = 0.0
            cur_dddelta_abs[idx] = 0.0

            # mark XY logging done for those envs (first episode only)
            if save_xy:
                xy_done = xy_done | done_event

        # keep finished envs stable by force resetting them every step
        done_for_reset = done_mask | (ep_count >= int(num_episodes))
        if use_partial_reset and done_for_reset.any():
            obs_next, _obs_raw_r, _state_vec_r = env.partial_reset(
                init_state=init_state,
                done_mask=done_for_reset,
                is_perturbed=False,
                regenerate_traj=False,
            )

        obs = obs_next

        # stop early if all envs finished required episodes
        if bool((ep_count >= int(num_episodes)).all().item()):
            break

    if len(ep_returns) == 0:
        return {
            "return_mean": float("nan"),
            "return_std": float("nan"),
            "return_per_step_mean": float("nan"),
            "return_per_step_std": float("nan"),
            "len_mean": float("nan"),
            "ey_mean": float("nan"),
            "e_psi_v_mean": float("nan"),
            "dv_mean": float("nan"),
            "cost_y_mean": float("nan"),
            "cost_psi_mean": float("nan"),
            "cost_v_mean": float("nan"),
            "cost_ay_mean": float("nan"),
            "d_delta_ref_abs_mean": float("nan"),
            "cost_d_delta_ref_mean": float("nan"),
            "dd_delta_ref_abs_mean": float("nan"),
            "cost_dd_delta_ref_mean": float("nan"),
        }

    ep_returns_t = torch.cat(ep_returns)
    ep_lengths_t = torch.cat(ep_lengths).to(dtype=torch.float32)
    ep_ret_per_step = ep_returns_t / torch.clamp(ep_lengths_t, min=1.0)

    ep_ey_mean_t = torch.cat(ep_ey_mean)
    ep_epsi_mean_t = torch.cat(ep_epsi_mean)
    ep_dv_mean_t = torch.cat(ep_dv_mean)

    ep_cost_y_mean_t = torch.cat(ep_cost_y_mean)
    ep_cost_psi_mean_t = torch.cat(ep_cost_psi_mean)
    ep_cost_v_mean_t = torch.cat(ep_cost_v_mean)
    ep_cost_ay_mean_t = torch.cat(ep_cost_ay_mean)
    ep_cost_ddelta_mean_t = torch.cat(ep_cost_ddelta_mean)
    ep_cost_dddelta_mean_t = torch.cat(ep_cost_dddelta_mean)
    ep_ddelta_mean_t = torch.cat(ep_ddelta_mean)
    ep_dddelta_mean_t = torch.cat(ep_dddelta_mean)

    result = {
        "return_mean": float(ep_returns_t.mean().item()),
        "return_std": float(ep_returns_t.std().item()),
        "return_per_step_mean": float(ep_ret_per_step.mean().item()),
        "return_per_step_std": float(ep_ret_per_step.std().item()),
        "len_mean": float(ep_lengths_t.mean().item()),
        "ey_mean": float(ep_ey_mean_t.mean().item()),
        "e_psi_v_mean": float(ep_epsi_mean_t.mean().item()),
        "dv_mean": float(ep_dv_mean_t.mean().item()),
        "cost_y_mean": float(ep_cost_y_mean_t.mean().item()),
        "cost_psi_mean": float(ep_cost_psi_mean_t.mean().item()),
        "cost_v_mean": float(ep_cost_v_mean_t.mean().item()),
        "cost_ay_mean": float(ep_cost_ay_mean_t.mean().item()),
        "d_delta_ref_abs_mean": float(ep_ddelta_mean_t.mean().item()),
        "cost_d_delta_ref_mean": float(ep_cost_ddelta_mean_t.mean().item()),
        "dd_delta_ref_abs_mean": float(ep_dddelta_mean_t.mean().item()),
        "cost_dd_delta_ref_mean": float(ep_cost_dddelta_mean_t.mean().item()),
    }

    # ---- plot saving (XY + time-series) ----
    if save_xy:
        import os as _os
        import math as _math
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        axes = axes.ravel()

        for i in range(min(num_xy_envs, 4)):
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
            xy_filename = "sapo_eval_xy.png"
        _os.makedirs(_os.path.dirname(xy_filename) or ".", exist_ok=True)
        plt.savefig(xy_filename, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved eval XY plot: {xy_filename}")

        # time-series per env
        dt_local = float(env.dt)
        base, ext = _os.path.splitext(xy_filename)
        if ext == "":
            ext = ".png"

        for i in range(num_xy_envs):
            T = len(ts_v_hist[i])
            if T == 0:
                continue
            t_axis = [k * dt_local for k in range(T)]

            fig, axes = plt.subplots(8, 1, figsize=(10, 16), sharex=True)

            axes[0].plot(t_axis, ts_v_hist[i], label="v [m/s]")
            axes[0].plot(t_axis, ts_vref_hist[i], label="v_ref [m/s]")
            axes[0].set_ylabel("Speed")
            axes[0].grid(True)
            axes[0].legend(loc="upper right")

            axes[1].plot(t_axis, ts_ax_hist[i], label="a_x [m/s^2]")
            axes[1].set_ylabel("a_x")
            axes[1].grid(True)
            axes[1].legend(loc="upper right")

            axes[2].plot(t_axis, ts_ay_hist[i], label="a_y [m/s^2]")
            axes[2].set_ylabel("a_y")
            axes[2].grid(True)
            axes[2].legend(loc="upper right")

            delta_deg = [_math.degrees(d) for d in ts_delta_hist[i]]
            delta_ref_deg = [_math.degrees(d) for d in ts_delta_ref_hist[i]]
            axes[3].plot(t_axis, delta_deg, label="delta [deg]")
            axes[3].plot(t_axis, delta_ref_deg, label="delta_ref [deg]")
            if len(ts_delta_geom_hist[i]) == len(t_axis):
                delta_geom_deg = [_math.degrees(d) for d in ts_delta_geom_hist[i]]
                axes[3].plot(t_axis, delta_geom_deg, label="delta_geom [deg]")
            axes[3].set_ylabel("Steering")
            axes[3].grid(True)
            axes[3].legend(loc="upper right")

            axes[4].plot(t_axis, ts_ey_hist[i], label="e_y [m]")
            axes[4].set_ylabel("e_y")
            axes[4].grid(True)
            axes[4].legend(loc="upper right")

            axes[5].plot(t_axis, ts_epsi_hist[i], label="e_psi_v [rad]")
            axes[5].set_ylabel("e_psi_v")
            axes[5].grid(True)
            axes[5].legend(loc="upper right")

            ddelta_deg_s = [_math.degrees(d) for d in ts_ddelta_ref_hist[i]]
            axes[6].plot(t_axis, ddelta_deg_s, label="d_delta_ref [deg/s]")
            axes[6].set_ylabel("d_delta_ref")
            axes[6].grid(True)
            axes[6].legend(loc="upper right")

            dddelta_deg_s2 = [_math.degrees(d) for d in ts_dd_delta_ref_hist[i]]
            axes[7].plot(t_axis, dddelta_deg_s2, label="dd_delta_ref [deg/s^2]")
            axes[7].set_xlabel("time [s]")
            axes[7].set_ylabel("jerk")
            axes[7].grid(True)
            axes[7].legend(loc="upper right")

            fig.suptitle(f"Eval time-series (env {i})")
            plt.tight_layout()

            ts_filename = f"{base}_env{i}_timeseries{ext}"
            plt.savefig(ts_filename, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Saved eval time-series plot: {ts_filename}")

    return result


def plot_training_curves(
    history: List[Dict[str, float]],
    *,
    out_path: str,
) -> None:
    if len(history) == 0:
        return
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    xs = [h["update"] for h in history]
    def _get(k: str):
        return [h.get(k, float("nan")) for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()

    axes[0].plot(xs, _get("mean_reward"))
    axes[0].set_title("mean_reward")
    axes[0].grid(True)

    axes[1].plot(xs, _get("actor_loss"))
    axes[1].set_title("actor_loss")
    axes[1].grid(True)

    axes[2].plot(xs, _get("critic_loss"))
    axes[2].set_title("critic_loss")
    axes[2].grid(True)

    axes[3].plot(xs, _get("alpha"))
    axes[3].set_title("alpha")
    axes[3].grid(True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    parser.add_argument("--seed", type=int, default=0)

    # env / traj
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--traj-length", type=float, default=2000.0)
    parser.add_argument("--ds", type=float, default=1.0)
    parser.add_argument("--v-min-kph", type=float, default=50.0)
    parser.add_argument("--v-max-kph", type=float, default=80.0)
    parser.add_argument("--R-min", type=float, default=30.0)
    parser.add_argument("--R-max", type=float, default=60.0)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--kappa-preview", type=int, default=21)

    # reward weights (defaults match ppo_test STAGE3-ish)
    parser.add_argument("--w-y", type=float, default=0.1)
    parser.add_argument("--w-psi", type=float, default=10.0)
    parser.add_argument("--w-v-under", type=float, default=0.05)
    parser.add_argument("--w-v-over", type=float, default=0.5)
    parser.add_argument("--w-ay", type=float, default=0.1)
    parser.add_argument("--w-kappa", type=float, default=0.001)
    parser.add_argument("--w-d-delta-ref", type=float, default=1.0)
    parser.add_argument("--w-dd-delta-ref", type=float, default=0.1)
    parser.add_argument("--w-tire-alpha-excess", type=float, default=0.1)

    # SAPO hyperparams
    parser.add_argument("--updates", type=int, default=2000)
    parser.add_argument("--horizon", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.999)
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

    parser.add_argument("--eval-envs", type=int, default=8, help="number of eval environments (fixed trajectories)")
    parser.add_argument("--eval-episodes", type=int, default=3, help="episodes per eval env")
    parser.add_argument("--eval-max-steps", type=int, default=3000, help="max steps per eval loop")
    parser.add_argument("--plot-envs", type=int, default=4, help="how many eval envs to plot (<=4 recommended)")
    parser.add_argument("--save-plots", type=int, default=1, choices=[0, 1], help="1: save eval XY/time-series + training curves")
    parser.add_argument("--log-jsonl", type=int, default=1, choices=[0, 1], help="1: write {run_id}_progress.jsonl")
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
        w_kappa=float(args.w_kappa),
        w_tire_alpha_excess=float(args.w_tire_alpha_excess),
        loss_y="l1",
        loss_psi="l1",
        loss_v="l1",
        loss_ay="l1",
        loss_d_delta_ref="l1",
        loss_dd_delta_ref="l1",
        loss_kappa="l1",
        loss_tire_alpha_excess="l1",
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
        traj_generator=lambda: make_train_traj(
            dt=float(args.dt),
            total_length=float(args.traj_length),
            ds=float(args.ds),
            v_min_kph=float(args.v_min_kph),
            v_max_kph=float(args.v_max_kph),
            R_min=float(args.R_min),
            R_max=float(args.R_max),
            seed=None,
        ),
        angle_wrap_mode="atan2",
    )

    init_state = make_init_state(ref_trajs, device=device, dtype=dtype)
    obs_norm, _obs_raw, _state_vec = env.reset(init_state, is_perturbed=False)
    obs_dim = int(obs_norm.shape[1])
    action_dim = 2

    # action bounds
    action_min, action_max = build_action_limits(veh_params)

    # ---- Logging / eval setup ----
    progress_jsonl = os.path.join(args.ckpt_dir, f"{run_id}_progress.jsonl")

    # ---- Eval env (fixed trajectories for comparable metrics) ----
    eval_ref_trajs = [
        make_train_traj(
            dt=float(args.dt),
            total_length=float(args.traj_length),
            ds=float(args.ds),
            v_min_kph=float(args.v_min_kph),
            v_max_kph=float(args.v_max_kph),
            R_min=float(args.R_min),
            R_max=float(args.R_max),
            seed=1000 + i,
        )
        for i in range(max(1, int(args.eval_envs)))
    ]
    eval_env = BatchedPathTrackingEnvFrenetDifferentiable(
        ref_trajs=eval_ref_trajs,
        kappa_preview_offsets=kappa_preview_offsets,
        vehicle_params=veh_params,
        reward_weights=weights,
        max_steps=int(args.max_steps),
        device=str(device),
        dtype=dtype,
        angle_wrap_mode="atan2",
    )
    eval_init_state = make_init_state(eval_ref_trajs, device=device, dtype=dtype)
    eval_env.reset(eval_init_state, is_perturbed=False)

    print(f"[Env] obs_dim={obs_dim} action_dim={action_dim} B(train)={env.B} B(eval)={eval_env.B} dt={env.dt}")
    print(f"[Action] min={action_min.tolist()} max={action_max.tolist()}")
    if int(args.log_jsonl) == 1:
        print(f"[Log] progress: {progress_jsonl}")

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

        # rollout 中に done になった env は PPO (ppo_test.py) と同様に partial reset して継続
        agent.init_state = init_state
        agent.reset_is_perturbed = False
        agent.regenerate_traj_on_done = True

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
    history: List[Dict[str, float]] = []
    t0 = time.time()
    for update in range(start_update, int(args.updates)):
        if args.reset_interval and (update % int(args.reset_interval) == 0) and update != start_update:
            env.reset(init_state, is_perturbed=False)

        logs = agent.update(env)

        if (update % int(args.log_interval)) == 0:
            elapsed = time.time() - t0
            env_dbg = collect_env_debug(env)

            msg = (
                f"[{run_id}] upd {update:05d} | "
                f"actor {logs.get('actor_loss', float('nan')):+.4e} | "
                f"critic {logs.get('critic_loss', float('nan')):+.4e} | "
                f"alpha {logs.get('alpha', float('nan')):.4f} | "
                f"H {logs.get('mean_entropy', float('nan')):.3f} | "
                f"r {logs.get('mean_reward', float('nan')):.4f} | "
                f"done {env_dbg.get('done_frac', float('nan')):.2f} | "
                f"|ey| {env_dbg.get('ey_abs_mean', float('nan')):.3f} | "
                f"|dv| {env_dbg.get('dv_abs_mean', float('nan')):.3f} | "
                f"{elapsed/60.0:.1f} min"
            )
            print(msg)

            history.append(
                {
                    "update": float(update),
                    "mean_reward": float(logs.get("mean_reward", float("nan"))),
                    "actor_loss": float(logs.get("actor_loss", float("nan"))),
                    "critic_loss": float(logs.get("critic_loss", float("nan"))),
                    "alpha": float(logs.get("alpha", float("nan"))),
                }
            )

            if int(args.log_jsonl) == 1:
                rec: Dict[str, Any] = {
                    "run_id": run_id,
                    "update": int(update),
                    "elapsed_sec": float(elapsed),
                    **{k: float(v) for k, v in logs.items()},
                    "env": env_dbg,
                }
                _append_jsonl(progress_jsonl, rec)

        if args.eval_interval and (update % int(args.eval_interval) == 0) and update != start_update:
            xy_path = os.path.join(args.ckpt_dir, f"{run_id}_eval_xy_update_{update:05d}.png")
            stats = evaluate_policy_detailed(
                agent,
                eval_env,
                eval_init_state,
                ref_trajs=eval_ref_trajs,
                num_episodes=int(args.eval_episodes),
                max_steps=int(args.eval_max_steps),
                use_partial_reset=True,
                save_xy=bool(int(args.save_plots)),
                xy_filename=xy_path if bool(int(args.save_plots)) else None,
                num_xy_envs=int(args.plot_envs),
            )

            print(f"\n========== Eval {update:05d} ==========")
            print(f"  return_mean           : {stats['return_mean']:.3f}")
            print(f"  return_per_step_mean  : {stats['return_per_step_mean']:.4f}")
            print(f"  episode_len_mean      : {stats['len_mean']:.1f}")
            print(f"  mean |e_y|            : {stats['ey_mean']:.4f}")
            print(f"  mean |e_psi_v|        : {stats['e_psi_v_mean']:.4f}")
            print(f"  mean |Δv|             : {stats['dv_mean']:.4f}")
            print(f"  mean cost_y           : {stats['cost_y_mean']:.6f}")
            print(f"  mean cost_psi         : {stats['cost_psi_mean']:.6f}")
            print(f"  mean cost_v           : {stats['cost_v_mean']:.6f}")
            print(f"  mean cost_ay          : {stats['cost_ay_mean']:.6f}")
            print(f"  mean cost_d_delta_ref : {stats['cost_d_delta_ref_mean']:.6f}")
            print(f"  mean cost_dd_delta_ref: {stats['cost_dd_delta_ref_mean']:.6f}")
            print(f"==============================\n")

            if int(args.log_jsonl) == 1:
                _append_jsonl(progress_jsonl, {"run_id": run_id, "update": int(update), "eval": stats})

            if bool(int(args.save_plots)):
                curve_path = os.path.join(args.ckpt_dir, f"{run_id}_training_curves.png")
                plot_training_curves(history, out_path=curve_path)

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
