
import math
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.distributions as D


# -----------------------------
# Utilities
# -----------------------------

def _tanh_squash(
    z: torch.Tensor,
    action_scale: torch.Tensor,
    action_bias: torch.Tensor,
) -> torch.Tensor:
    return torch.tanh(z) * action_scale + action_bias


def _squashed_gaussian_log_prob(
    dist: D.Normal,
    z: torch.Tensor,
    action_scale: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    log π(a|s) for a = tanh(z) * scale + bias.
    Uses the standard tanh change-of-variables correction + linear scale correction.
    """
    # log p(z)
    log_prob_z = dist.log_prob(z)

    # log |det d(tanh(z))/dz| = sum log(1 - tanh(z)^2)
    # numerically stable: log(1 - tanh^2) = 2*(log(2) - z - softplus(-2z))
    log_det_jacobian_tanh = 2.0 * (math.log(2.0) - z - nn.functional.softplus(-2.0 * z))

    # scale is per-dim; a = tanh(z) * scale + bias => |da/dtanh| = scale
    log_det_jacobian_scale = torch.log(action_scale + eps)

    log_prob = log_prob_z - log_det_jacobian_tanh - log_det_jacobian_scale
    return log_prob.sum(dim=-1, keepdim=True)



def _detach_dataclass_like(obj: Any) -> Any:
    """
    Recursively detach torch.Tensors in a dataclass-like object (has __dict__).
    Used to 'stopgrad' environment state after the actor update, matching SAPO pseudocode. citeturn1view1
    """
    if torch.is_tensor(obj):
        return obj.detach()
    if isinstance(obj, (tuple, list)):
        return type(obj)(_detach_dataclass_like(x) for x in obj)
    if isinstance(obj, dict):
        return {k: _detach_dataclass_like(v) for k, v in obj.items()}
    if hasattr(obj, "__dict__"):
        # reconstruct by copying attributes (works for simple dataclasses)
        cls = obj.__class__
        try:
            kwargs = {k: _detach_dataclass_like(v) for k, v in obj.__dict__.items()}
            return cls(**kwargs)
        except Exception:
            # fallback: shallow copy and in-place detach attributes
            for k, v in obj.__dict__.items():
                setattr(obj, k, _detach_dataclass_like(v))
            return obj
    return obj

# -----------------------------
# Networks
# -----------------------------

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, *, layer_norm: bool = True):
        super().__init__()
        layers: List[nn.Module] = []
        last = in_dim
        for _ in range(2):
            layers.append(nn.Linear(last, hidden_dim))
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.SiLU())
            last = hidden_dim
        layers.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SquashedGaussianActor(nn.Module):
    """
    State-dependent (per-state) variance squashed Gaussian policy:
      z ~ N(mu(s), std(s)), a = tanh(z) scaled to [action_min, action_max].

    This matches SAPO design choice III (squashed Normal with state-dependent variance). citeturn2view0
    """
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int,
        action_min: Optional[torch.Tensor] = None,
        action_max: Optional[torch.Tensor] = None,
        log_std_min: float = -10.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)


        self.register_buffer("action_min", action_min)
        self.register_buffer("action_max", action_max)

        self.register_buffer("action_scale", (action_max - action_min) * 0.5)
        self.register_buffer("action_bias", (action_max + action_min) * 0.5)

        self.body = MLP(obs_dim, hidden_dim, hidden_dim, layer_norm=True)
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          action (B, A), log_pi (B,1), entropy_proxy (B,1) where entropy_proxy ≈ -log_pi.
        """
        h = self.body(obs)
        mu = self.mu_head(h)
        log_std = self.log_std_head(h).clamp(self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        dist = D.Normal(mu, std)

        # reparameterized sample
        z = dist.rsample()
        action = _tanh_squash(z, self.action_scale, self.action_bias)
        log_pi = _squashed_gaussian_log_prob(dist, z, self.action_scale)
        entropy_proxy = -log_pi  # common proxy in SAC-style max-ent RL

        return action, log_pi, entropy_proxy

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        h = self.body(obs)
        mu = self.mu_head(h)
        if deterministic:
            z = mu
        else:
            log_std = self.log_std_head(h).clamp(self.log_std_min, self.log_std_max)
            std = torch.exp(log_std)
            z = (D.Normal(mu, std)).sample()
        return _tanh_squash(z, self.action_scale, self.action_bias)


class CriticV(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int):
        super().__init__()
        self.v = MLP(obs_dim, hidden_dim, 1, layer_norm=True)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.v(obs)


# -----------------------------
# SAPO Agent
# -----------------------------

@dataclass
class SAPOConfig:
    # rollout / horizon
    horizon: int = 32
    gamma: float = 0.99
    td_lambda: float = 0.95

    # optimization
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    grad_clip: float = 1.0

    # critic updates per iteration (paper uses many mini-epochs) citeturn1view1
    critic_updates: int = 16
    minibatch_size: int = 1024

    # entropy temperature / target entropy
    init_log_alpha: float = 0.0
    target_entropy: Optional[float] = None  # default: -action_dim

    # entropy normalization (SAPO design choice II) citeturn2view0
    entropy_norm_denom: Optional[float] = None  # default: action_dim

    # value target clipping (stability knob; SAPO pseudocode mentions clipped targets) citeturn1view1
    value_target_clip: Optional[float] = 100.0

    # network sizes
    hidden_dim: int = 256


class SAPOAgentBatched:
    """
    Soft Analytic Policy Optimization (SAPO) agent for differentiable environments.

    Key idea:
      - Update the stochastic actor by backpropagating through the differentiable simulator
        using a max-entropy short-horizon objective. citeturn2view0turn5view0
      - Train a soft value function (ensemble of two V critics) by TD learning on detached data,
        using the clipped double critic trick and no target networks. citeturn2view0

    This implementation is designed to coexist with the existing PPO codebase, but does not
    reuse PPO's GAE/ratio objective.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        num_envs: int,
        *,
        action_min: Optional[torch.Tensor] = None,
        action_max: Optional[torch.Tensor] = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        cfg: Optional[SAPOConfig] = None,
    ):
        self.device = device
        self.dtype = dtype
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.num_envs = int(num_envs)
        self.cfg = cfg if cfg is not None else SAPOConfig()

        self.actor = SquashedGaussianActor(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=self.cfg.hidden_dim,
            action_min=torch.as_tensor(action_min, device=device, dtype=dtype),
            action_max=torch.as_tensor(action_max, device=device, dtype=dtype),
        ).to(device=device, dtype=dtype)

        # double critic (value ensemble)
        self.critic1 = CriticV(obs_dim, self.cfg.hidden_dim).to(device=device, dtype=dtype)
        self.critic2 = CriticV(obs_dim, self.cfg.hidden_dim).to(device=device, dtype=dtype)

        # entropy temperature parameter (log α)
        self.log_alpha = torch.nn.Parameter(
            torch.tensor(float(self.cfg.init_log_alpha), device=device, dtype=dtype)
        )

        self.actor_opt = torch.optim.AdamW(self.actor.parameters(), lr=self.cfg.actor_lr, betas=(0.9, 0.999))
        self.critic_opt = torch.optim.AdamW(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=self.cfg.critic_lr,
            betas=(0.9, 0.999),
        )
        self.alpha_opt = torch.optim.AdamW([self.log_alpha], lr=self.cfg.alpha_lr, betas=(0.9, 0.999))

        if self.cfg.target_entropy is None:
            self.target_entropy = -float(action_dim)  # SAC-style default
        else:
            self.target_entropy = float(self.cfg.target_entropy)

        if self.cfg.entropy_norm_denom is None:
            self.entropy_norm_denom = float(action_dim)
        else:
            self.entropy_norm_denom = float(self.cfg.entropy_norm_denom)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def _freeze_critics(self, freeze: bool) -> None:
        for p in self.critic1.parameters():
            p.requires_grad_(not freeze)
        for p in self.critic2.parameters():
            p.requires_grad_(not freeze)

    def _compute_v(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        v1 = self.critic1(obs)
        v2 = self.critic2(obs)
        v_min = torch.minimum(v1, v2)
        v_avg = 0.5 * (v1 + v2)
        return v1, v2, v_min, v_avg

    def update(self, env: Any) -> Dict[str, float]:
        """
        One SAPO iteration:
          1) Unroll differentiable horizon H from current env state
          2) Actor update by analytic gradients through env + (frozen) critic bootstrap citeturn2view0turn1view1
          3) Detach rollout tensors (stopgrad) then:
              - temperature update (unnormalized entropy) citeturn2view0
              - critic TD(lambda) updates (normalized entropy, min critic for targets) citeturn2view0turn5view0
        """
        H = self.cfg.horizon
        gamma = float(self.cfg.gamma)
        lam = float(self.cfg.td_lambda)

        # --------
        # (A) Differentiable rollout for ACTOR update
        # --------
        env_state = env.get_env_state()

        obs_raw0, _state_vec0, _cache0 = env.compute_obs_state(env_state, return_cache=True)
        obs0 = env.normalize_obs(obs_raw0)

        # lists for actor objective (graph-connected)
        rewards_g: List[torch.Tensor] = []
        logpi_g: List[torch.Tensor] = []
        entnorm_g: List[torch.Tensor] = []
        done_g: List[torch.Tensor] = []

        # detached buffer for critic/alpha updates
        obs_buf: List[torch.Tensor] = []
        next_obs_buf: List[torch.Tensor] = []
        reward_buf: List[torch.Tensor] = []
        done_buf: List[torch.Tensor] = []
        logpi_buf: List[torch.Tensor] = []
        entraw_buf: List[torch.Tensor] = []

        obs_t = obs0
        alive = torch.ones((self.num_envs, 1), device=self.device, dtype=self.dtype)

        for t in range(H):
            action_t, log_pi_t, ent_raw_t = self.actor(obs_t)  # differentiable
            ent_norm_t = ent_raw_t / self.entropy_norm_denom

            (env_state_next,
             obs_norm_next,
             _obs_raw_next,
             _state_vec_next,
             reward_t,
             done_t,
             _info) = env.functional_step(env_state, action_t, compute_info=False, normalize_obs=True)
            # Ensure shapes are (B,1) for scalars to avoid time/batch shape pitfalls
            if reward_t.dim() == 1:
                reward_t = reward_t.unsqueeze(-1)
            if done_t.dim() == 1:
                done_t = done_t.unsqueeze(-1)
            # PPO (ppo_test.py) と同様: done になった要素は新しい軌道で partial reset して継続する
            if bool(done_t.any().detach().cpu().item()):
                init_state = getattr(self, "init_state", None)
                if (init_state is not None) and hasattr(env, "functional_partial_reset"):
                    done_mask = done_t.detach().squeeze(-1)
                    (env_state_next,
                     obs_norm_next,
                     _obs_raw_next,
                     _state_vec_next) = env.functional_partial_reset(
                        env_state_next,
                        init_state,
                        done_mask,
                        is_perturbed=bool(getattr(self, "reset_is_perturbed", False)),
                        regenerate_traj=bool(getattr(self, "regenerate_traj_on_done", True)),
                        normalize_obs=True,
                        obs_clip=5.0,
                    )

            # actor objective terms (keep graph)
            rewards_g.append(reward_t)
            logpi_g.append(log_pi_t)
            entnorm_g.append(ent_norm_t)
            done_g.append(done_t)

            # detached terms for critic / alpha
            obs_buf.append(obs_t.detach())
            next_obs_buf.append(obs_norm_next.detach())
            reward_buf.append(reward_t.detach())
            done_buf.append(done_t.detach())
            logpi_buf.append(log_pi_t.detach())
            entraw_buf.append(ent_raw_t.detach())

            # update rollout state
            env_state = env_state_next
            obs_t = obs_norm_next
            alive = alive * (1.0 - 1.0*done_t.detach())  # stopgrad mask

        obs_H = obs_t  # (B, obs_dim)

        # -------- Actor loss (maximize soft return) --------
        # Freeze critic params but allow dV/d(obs_H) to flow to actor via obs_H.
        self._freeze_critics(True)
        with torch.set_grad_enabled(True):
            _, _, _vminH, vavgH = self._compute_v(obs_H)
            alpha = self.alpha
            alpha_det = alpha.detach()

            G = torch.zeros((self.num_envs, 1), device=self.device, dtype=self.dtype)
            alive_g = torch.ones_like(G)
            
            for t in range(H):
                disc = gamma ** t
                
                # 修正: 生存フラグを更新する前に報酬を加算
                # (現在のステップで死んだとしても、その瞬間の報酬/ペナルティは受け取るべき)
                current_reward = rewards_g[t] + alpha_det * entnorm_g[t]
                G = G + disc * alive_g * current_reward
                
                # 次のステップへの生存フラグ更新
                alive_g = alive_g * (1.0 - 1.0 * done_g[t].detach())

            # bootstrap (もし最後まで生きていたら)
            G = G + (gamma ** H) * alive_g * vavgH

            actor_loss = -G.mean()

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        if self.cfg.grad_clip is not None and self.cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.grad_clip)
        self.actor_opt.step()
        self._freeze_critics(False)

        # IMPORTANT: detach env graph (SAPO pseudocode "stopgrad") citeturn1view1
        # (We already stored detached tensors; just ensure no references kept)
        del rewards_g, logpi_g, entnorm_g, done_g

        # --------
        # (B) Temperature (alpha) update (unnormalized entropy proxy) citeturn2view0
        # --------
        logpi_flat = torch.cat(logpi_buf, dim=0)  # (H*B,1)
        # SAC-style dual loss: maximize entropy => minimize alpha * (log_pi + target_entropy)
        alpha_loss = -(self.log_alpha * (logpi_flat + self.target_entropy).detach()).mean()

        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()

        alpha_det = self.alpha.detach()

        # --------
        # (C) Critic update: TD(lambda) targets with soft returns using MIN critic for bootstrapping citeturn2view0turn5view0
        # --------
        def _stack_time_major(buf, name: str) -> torch.Tensor:
            """
            Stack a time sequence into (H,B,...) and fix common (B,H,...) swaps.
            Accepts either a Python list[Tensor] of length H or an already-stacked Tensor.
            """
            if isinstance(buf, torch.Tensor):
                x = buf
            else:
                x = torch.stack(buf, dim=0)
            # Ensure scalar sequences have shape (H,B,1)
            if x.dim() == 2:
                x = x.unsqueeze(-1)
            if x.shape[0] != H:
                raise RuntimeError(f"{name}: expected time dim H={H} at dim0, got shape {tuple(x.shape)}")
            if x.shape[1] != self.num_envs:
                raise RuntimeError(f"{name}: expected batch dim B={self.num_envs} at dim1, got shape {tuple(x.shape)}")
            return x

        obs = _stack_time_major(obs_buf, "obs")        # (H,B,obs_dim)
        next_obs = _stack_time_major(next_obs_buf, "next_obs")  # (H,B,obs_dim)
        rewards = _stack_time_major(reward_buf, "rewards")  # (H,B,1)
        dones = _stack_time_major(done_buf, "dones")      # (H,B,1)
        ent_raw = _stack_time_major(entraw_buf, "ent_raw")  # (H,B,1)
        ent_norm = ent_raw / self.entropy_norm_denom

        soft_rewards = rewards + alpha_det * ent_norm

        # bootstrap value at next state (for each step)
        with torch.no_grad():
            # V_min(next_obs_t)
            v1_next = self.critic1(next_obs.view(-1, self.obs_dim)).view(H, self.num_envs, 1)
            v2_next = self.critic2(next_obs.view(-1, self.obs_dim)).view(H, self.num_envs, 1)
            vmin_next = torch.minimum(v1_next, v2_next)

            # TD(lambda) return
            targets = torch.zeros_like(soft_rewards)
            ret = vmin_next[-1]  # start with bootstrap at last next_obs
            for t in reversed(range(H)):
                not_done = 1.0 - 1.0*dones[t]
                if t == H - 1:
                    ret = soft_rewards[t] + gamma * not_done * vmin_next[t]
                else:
                    # TD(lambda): mix 1-step bootstrapped value and longer return
                    ret = soft_rewards[t] + gamma * not_done * ((1.0 - 1.0*lam) * vmin_next[t] + lam * ret)
                targets[t] = ret

            if self.cfg.value_target_clip is not None:
                targets = targets.clamp(-float(self.cfg.value_target_clip), float(self.cfg.value_target_clip))

        # Flatten for minibatch SGD
        X = obs.view(-1, self.obs_dim)          # (H*B,obs_dim)
        Y = targets.view(-1, 1)                 # (H*B,1)

        N = X.size(0)
        mb = min(self.cfg.minibatch_size, N)

        critic_losses: List[float] = []
        for _ in range(int(self.cfg.critic_updates)):
            idx = torch.randint(0, N, (mb,), device=self.device)
            xb = X[idx]
            yb = Y[idx]

            v1 = self.critic1(xb)
            v2 = self.critic2(xb)
            loss = nn.functional.mse_loss(v1, yb) + nn.functional.mse_loss(v2, yb)

            self.critic_opt.zero_grad(set_to_none=True)
            loss.backward()
            if self.cfg.grad_clip is not None and self.cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(self.critic1.parameters()) + list(self.critic2.parameters()),
                    self.cfg.grad_clip,
                )
            self.critic_opt.step()
            critic_losses.append(float(loss.detach().cpu().item()))

        # advance the REAL env state (stateful) to match the last env_state in rollout
        # (so next update continues from where we left off)
        env.set_env_state(_detach_dataclass_like(env_state))

        out = {
            "actor_loss": float(actor_loss.detach().cpu().item()),
            "alpha": float(self.alpha.detach().cpu().item()),
            "alpha_loss": float(alpha_loss.detach().cpu().item()),
            "critic_loss": float(np.mean(critic_losses) if critic_losses else 0.0),
            "mean_reward": float(rewards.mean().cpu().item()),
            "mean_entropy": float(ent_raw.mean().cpu().item()),
        }
        return out