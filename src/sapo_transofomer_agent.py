
import math
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.distributions as D
from torch.nn.parameter import UninitializedParameter


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



# -----------------------------
# Transformer (causal + relative positional bias)
# -----------------------------

class _RelPosCausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention with learnable relative position bias.
    - Input: (B, L, D)
    - Output: (B, L, D)
    """
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float = 0.0):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.head_dim = int(d_model // n_heads)
        self.max_seq_len = int(max_seq_len)

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out = nn.Linear(d_model, d_model, bias=True)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        # Relative position bias table: (2*max_seq_len-1, n_heads)
        self.rel_bias = nn.Parameter(torch.zeros(2 * self.max_seq_len - 1, self.n_heads))

    def _relative_bias(self, L: int, device: torch.device) -> torch.Tensor:
        # positions: (L,)
        pos = torch.arange(L, device=device)
        # rel[i, j] = j - i  (query i attends to key j)
        rel = pos[None, :] - pos[:, None]  # (L, L)
        rel = torch.clamp(rel, -(self.max_seq_len - 1), self.max_seq_len - 1)
        idx = rel + (self.max_seq_len - 1)  # shift to [0, 2*max_seq_len-2]
        # (L, L, n_heads) -> (n_heads, L, L)
        b = self.rel_bias[idx]  # gather
        return b.permute(2, 0, 1).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        if L > self.max_seq_len:
            raise ValueError(f"Sequence length L={L} exceeds max_seq_len={self.max_seq_len}")

        qkv = self.qkv(x)  # (B, L, 3D)
        q, k, v = qkv.chunk(3, dim=-1)

        # (B, n_heads, L, head_dim)
        q = q.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        # scaled dot-product attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))  # (B, n_heads, L, L)

        # add relative position bias
        att = att + self._relative_bias(L, x.device).unsqueeze(0)

        # causal mask (no attention to future)
        causal = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1)
        att = att.masked_fill(causal.unsqueeze(0).unsqueeze(0), float("-inf"))

        w = torch.softmax(att, dim=-1)
        w = self.attn_drop(w)

        y = w @ v  # (B, n_heads, L, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, L, D)  # (B, L, D)
        y = self.resid_drop(self.out(y))
        return y


class _TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = _RelPosCausalSelfAttention(d_model, n_heads, max_seq_len, dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class CausalTransformerEncoder(nn.Module):
    """
    Causal Transformer encoder for fixed-window history.
    """
    def __init__(
        self,
        token_dim: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        max_seq_len: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.token_dim = int(token_dim)
        self.d_model = int(d_model)
        self.max_seq_len = int(max_seq_len)

        self.in_proj = nn.Linear(self.token_dim, self.d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [_TransformerBlock(self.d_model, n_heads, self.max_seq_len, dropout=dropout) for _ in range(int(n_layers))]
        )
        self.ln_f = nn.LayerNorm(self.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, token_dim)
        x = self.drop(self.in_proj(x))
        for blk in self.blocks:
            x = blk(x)
        return self.ln_f(x)

class SquashedGaussianActor(nn.Module):
    """
    History-dependent squashed Gaussian policy using:
      - causal Transformer over a fixed window of observation tokens
      - MLP head that combines the latest Transformer state and preview features

    Inputs:
      obs_seq: (B,L,token_dim) or (B,token_dim) (treated as L=1)
      preview: (B,P) optional (kappa_preview + v_preview flattened)
    Outputs:
      action (B,A), log_pi (B,1), entropy_proxy (B,1) where entropy_proxy ≈ -log_pi.
    """
    def __init__(
        self,
        obs_dim: int,  # token_dim
        action_dim: int,
        hidden_dim: int,
        action_min: Optional[torch.Tensor] = None,
        action_max: Optional[torch.Tensor] = None,
        log_std_min: float = -10.0,
        log_std_max: float = 2.0,
        *,
        d_model: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        max_seq_len: int = 20,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.token_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

        # action scaling
        if action_min is None or action_max is None:
            # default to [-1,1]
            action_min = torch.full((action_dim,), -1.0)
            action_max = torch.full((action_dim,), 1.0)
        self.register_buffer("action_min", action_min.clone().detach())
        self.register_buffer("action_max", action_max.clone().detach())
        self.register_buffer("action_scale", (self.action_max - self.action_min) / 2.0)
        self.register_buffer("action_bias", (self.action_max + self.action_min) / 2.0)

        # transformer encoder (causal + relative positional bias)
        self.encoder = CausalTransformerEncoder(
            token_dim=self.token_dim,
            d_model=int(d_model),
            n_layers=int(n_layers),
            n_heads=int(n_heads),
            max_seq_len=int(max_seq_len),
            dropout=float(dropout),
        )

        # post-encoder MLP (lazy to allow preview dim to be decided by caller)
        self.post_ln = nn.LayerNorm(int(d_model))
        self.post_fc1 = nn.LazyLinear(hidden_dim)
        self.post_act = nn.SiLU()
        self.post_fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def _coerce_inputs(
        self,
        obs: torch.Tensor,
        preview: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          obs_seq: (B,L,token_dim)
          preview: (B,P) possibly empty
        """
        if obs.dim() == 2:
            # allow obs to be either token-only or token+preview (single-step)
            if obs.size(1) < self.token_dim:
                raise ValueError(f"obs has dim {obs.size(1)} < token_dim {self.token_dim}")
            if obs.size(1) > self.token_dim and preview is None:
                preview = obs[:, self.token_dim:]
                obs = obs[:, :self.token_dim]
            obs_seq = obs.unsqueeze(1)  # (B,1,D)
        elif obs.dim() == 3:
            if obs.size(-1) != self.token_dim:
                raise ValueError(f"obs_seq last dim must be token_dim={self.token_dim}, got {obs.size(-1)}")
            obs_seq = obs
        else:
            raise ValueError(f"obs must be (B,D) or (B,L,D), got {tuple(obs.shape)}")

        if preview is None:
            preview = obs_seq.new_zeros((obs_seq.size(0), 0))
        return obs_seq, preview

    def forward(self, obs: torch.Tensor, preview: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        obs_seq, preview = self._coerce_inputs(obs, preview)

        h_seq = self.encoder(obs_seq)           # (B,L,d_model)
        h_last = h_seq[:, -1, :]                # (B,d_model)
        h_last = self.post_ln(h_last)

        h = torch.cat([h_last, preview], dim=-1)
        h = self.post_act(self.post_fc1(h))
        h = self.post_act(self.post_fc2(h))

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
    def act(self, obs: torch.Tensor, preview: Optional[torch.Tensor] = None, deterministic: bool = False) -> torch.Tensor:
        obs_seq, preview = self._coerce_inputs(obs, preview)

        h_seq = self.encoder(obs_seq)
        h_last = self.post_ln(h_seq[:, -1, :])

        h = torch.cat([h_last, preview], dim=-1)
        h = self.post_act(self.post_fc1(h))
        h = self.post_act(self.post_fc2(h))

        mu = self.mu_head(h)
        if deterministic:
            z = mu
        else:
            log_std = self.log_std_head(h).clamp(self.log_std_min, self.log_std_max)
            std = torch.exp(log_std)
            z = (D.Normal(mu, std)).sample()
        return _tanh_squash(z, self.action_scale, self.action_bias)


class CriticV(nn.Module):
    """
    Soft value critic V(s) using the same encoder structure as the actor:
      - causal Transformer over history tokens
      - MLP head that combines last token state and preview features
    """
    def __init__(
        self,
        obs_dim: int,  # token_dim
        hidden_dim: int,
        *,
        d_model: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        max_seq_len: int = 20,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.token_dim = int(obs_dim)

        self.encoder = CausalTransformerEncoder(
            token_dim=self.token_dim,
            d_model=int(d_model),
            n_layers=int(n_layers),
            n_heads=int(n_heads),
            max_seq_len=int(max_seq_len),
            dropout=float(dropout),
        )
        self.post_ln = nn.LayerNorm(int(d_model))
        self.post_fc1 = nn.LazyLinear(hidden_dim)
        self.post_act = nn.SiLU()
        self.post_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.v_head = nn.Linear(hidden_dim, 1)

    def _coerce_inputs(
        self,
        obs: torch.Tensor,
        preview: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if obs.dim() == 2:
            if obs.size(1) < self.token_dim:
                raise ValueError(f"obs has dim {obs.size(1)} < token_dim {self.token_dim}")
            if obs.size(1) > self.token_dim and preview is None:
                preview = obs[:, self.token_dim:]
                obs = obs[:, :self.token_dim]
            obs_seq = obs.unsqueeze(1)
        elif obs.dim() == 3:
            if obs.size(-1) != self.token_dim:
                raise ValueError(f"obs_seq last dim must be token_dim={self.token_dim}, got {obs.size(-1)}")
            obs_seq = obs
        else:
            raise ValueError(f"obs must be (B,D) or (B,L,D), got {tuple(obs.shape)}")

        if preview is None:
            preview = obs_seq.new_zeros((obs_seq.size(0), 0))
        return obs_seq, preview

    def forward(self, obs: torch.Tensor, preview: Optional[torch.Tensor] = None) -> torch.Tensor:
        obs_seq, preview = self._coerce_inputs(obs, preview)
        h_seq = self.encoder(obs_seq)
        h_last = self.post_ln(h_seq[:, -1, :])
        h = torch.cat([h_last, preview], dim=-1)
        h = self.post_act(self.post_fc1(h))
        h = self.post_act(self.post_fc2(h))
        return self.v_head(h)


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

    # history / transformer
    history_len: int = 20
    transformer_d_model: int = 128
    transformer_n_layers: int = 2
    transformer_n_heads: int = 4
    transformer_dropout: float = 0.0
    history_detach: bool = True


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
            d_model=self.cfg.transformer_d_model,
            n_layers=self.cfg.transformer_n_layers,
            n_heads=self.cfg.transformer_n_heads,
            max_seq_len=self.cfg.history_len,
            dropout=self.cfg.transformer_dropout,
        ).to(device=device, dtype=dtype)

        # double critic (value ensemble)
        self.critic1 = CriticV(
            obs_dim,
            self.cfg.hidden_dim,
            d_model=self.cfg.transformer_d_model,
            n_layers=self.cfg.transformer_n_layers,
            n_heads=self.cfg.transformer_n_heads,
            max_seq_len=self.cfg.history_len,
            dropout=self.cfg.transformer_dropout,
        ).to(device=device, dtype=dtype)
        self.critic2 = CriticV(
            obs_dim,
            self.cfg.hidden_dim,
            d_model=self.cfg.transformer_d_model,
            n_layers=self.cfg.transformer_n_layers,
            n_heads=self.cfg.transformer_n_heads,
            max_seq_len=self.cfg.history_len,
            dropout=self.cfg.transformer_dropout,
        ).to(device=device, dtype=dtype)

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

    
    def _ensure_lazy_initialized(self, hist_seq: torch.Tensor, preview: Optional[torch.Tensor]) -> None:
        """Initialize LazyLinear parameters (if any) by running a dummy forward pass."""
        needs_init = False
        for mod in (self.actor, self.critic1, self.critic2):
            for p in mod.parameters():
                if isinstance(p, UninitializedParameter):
                    needs_init = True
                    break
            if needs_init:
                break
        if not needs_init:
            return

        if preview is None:
            preview = hist_seq.new_zeros((hist_seq.size(0), 0))

        with torch.no_grad():
            _ = self.actor(hist_seq, preview)
            _ = self.critic1(hist_seq, preview)
            _ = self.critic2(hist_seq, preview)

    def _freeze_critics(self, freeze: bool) -> None:
            for p in self.critic1.parameters():
                if isinstance(p, UninitializedParameter):
                    continue
                p.requires_grad_(not freeze)
            for p in self.critic2.parameters():
                if isinstance(p, UninitializedParameter):
                    continue
                p.requires_grad_(not freeze)

    def _compute_v(self, obs: torch.Tensor, preview: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        v1 = self.critic1(obs, preview)
        v2 = self.critic2(obs, preview)
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

        # preview features (kappa_preview, v_preview) are kept separate from obs tokens
        def _get_preview(e_state) -> torch.Tensor:
            if hasattr(env, "get_kappa_preview") and hasattr(env, "get_v_preview"):
                k = env.get_kappa_preview(e_state)
                vpr = env.get_v_preview(e_state)
                return torch.cat([k, vpr], dim=1)
            return obs0.new_zeros((obs0.size(0), 0))

        # fixed-window history buffer for Transformer
        L_hist = int(self.cfg.history_len)
        hist_seq = torch.zeros((self.num_envs, L_hist, self.obs_dim), device=self.device, dtype=self.dtype)
        hist_seq[:, -1, :] = obs0
        preview_t = _get_preview(env_state)
        self._ensure_lazy_initialized(hist_seq, preview_t)

        # lists for actor objective (graph-connected)
        rewards_g: List[torch.Tensor] = []
        logpi_g: List[torch.Tensor] = []
        entnorm_g: List[torch.Tensor] = []
        done_g: List[torch.Tensor] = []

        # detached buffer for critic/alpha
        hist_buf: List[torch.Tensor] = []
        next_hist_buf: List[torch.Tensor] = []
        preview_buf: List[torch.Tensor] = []
        next_preview_buf: List[torch.Tensor] = []
        reward_buf: List[torch.Tensor] = []
        done_buf: List[torch.Tensor] = []
        logpi_buf: List[torch.Tensor] = []
        entraw_buf: List[torch.Tensor] = []

        obs_t = obs0
        alive = torch.ones((self.num_envs, 1), device=self.device, dtype=self.dtype)

        for t in range(H):
            action_t, log_pi_t, ent_raw_t = self.actor(hist_seq, preview_t)  # differentiable
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
                    try:
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
                    except TypeError:
                        (env_state_next,
                         obs_norm_next,
                         _obs_raw_next,
                         _state_vec_next) = env.functional_partial_reset(
                            env_state_next,
                            init_state,
                            done_mask,
                            is_perturbed=bool(getattr(self, "reset_is_perturbed", False)),
                            normalize_obs=True,
                            obs_clip=5.0,
                        )

            # stopgrad mask (alive is for actor-loss bookkeeping; not used for env reset)
            done_t_det = done_t.detach()
            reward_t = reward_t * alive
            alive = alive * (1.0 - 1.0*done_t_det)

            # actor objective (graph-connected)
            rewards_g.append(reward_t)
            logpi_g.append(log_pi_t)
            entnorm_g.append(ent_norm_t)
            done_g.append(done_t)

            # update history for next step (stop-grad on the "old" part like TXL-style memory)
            hist_old = hist_seq[:, 1:, :]
            if bool(getattr(self.cfg, "history_detach", True)):
                hist_old = hist_old.detach()
            hist_seq_next = torch.cat([hist_old, obs_norm_next.unsqueeze(1)], dim=1)

            # if we did a partial reset, clear history for those envs
            if bool(done_t.any().detach().cpu().item()):
                done_mask = done_t.detach().squeeze(-1)
                reset_seq = torch.zeros_like(hist_seq_next)
                reset_seq[:, -1, :] = obs_norm_next
                hist_seq_next = torch.where(done_mask.view(-1, 1, 1), reset_seq, hist_seq_next)

            preview_next = _get_preview(env_state_next)

            # detached terms for critic / alpha
            hist_buf.append(hist_seq.detach())
            next_hist_buf.append(hist_seq_next.detach())
            preview_buf.append(preview_t.detach())
            next_preview_buf.append(preview_next.detach())
            reward_buf.append(reward_t.detach())
            done_buf.append(done_t.detach())
            logpi_buf.append(log_pi_t.detach())
            entraw_buf.append(ent_raw_t.detach())

            # update rollout state
            env_state = env_state_next
            obs_t = obs_norm_next
            hist_seq = hist_seq_next
            preview_t = preview_next

        obs_H = hist_seq  # (B, L_hist, obs_dim)
        preview_H = preview_t  # (B, P+Q)

        # -------- Actor loss (maximize soft return) --------
        # Freeze critic params but allow dV/d(obs_H) to flow to actor via obs_H.
        self._freeze_critics(True)
        with torch.set_grad_enabled(True):
            _, _, _vminH, vavgH = self._compute_v(obs_H, preview_H)
            alpha = self.alpha
            alpha_det = alpha.detach()

            G = torch.zeros((self.num_envs, 1), device=self.device, dtype=self.dtype)
            alive_g = torch.ones_like(G)

            for t in range(H):
                disc = gamma ** t

                # 修正: 生存フラグを更新する前に報酬を加算
                not_done = 1.0 - 1.0*done_g[t]
                G = G + disc * alive_g * (rewards_g[t] + alpha_det * entnorm_g[t])
                alive_g = alive_g * not_done

            actor_obj = (G + (gamma ** H) * alive_g * vavgH).mean()
            actor_loss = -actor_obj

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        if self.cfg.grad_clip is not None and self.cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.grad_clip)
        self.actor_opt.step()
        self._freeze_critics(False)

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

        obs = _stack_time_major(hist_buf, "hist")        # (H,B,L,obs_dim)
        next_obs = _stack_time_major(next_hist_buf, "next_hist")  # (H,B,L,obs_dim)
        preview = _stack_time_major(preview_buf, "preview")      # (H,B,P)
        next_preview = _stack_time_major(next_preview_buf, "next_preview")  # (H,B,P)
        rewards = _stack_time_major(reward_buf, "rewards")  # (H,B,1)
        dones = _stack_time_major(done_buf, "dones")      # (H,B,1)
        ent_raw = _stack_time_major(entraw_buf, "ent_raw")  # (H,B,1)
        ent_norm = ent_raw / self.entropy_norm_denom

        soft_rewards = rewards + alpha_det * ent_norm

        # bootstrap value at next state (for each step)
        with torch.no_grad():
            Lh = next_obs.size(2)
            v1_next = self.critic1(
                next_obs.view(-1, Lh, self.obs_dim),
                next_preview.view(-1, next_preview.size(-1)),
            ).view(H, self.num_envs, 1)
            v2_next = self.critic2(
                next_obs.view(-1, Lh, self.obs_dim),
                next_preview.view(-1, next_preview.size(-1)),
            ).view(H, self.num_envs, 1)
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
        Lh = obs.size(2)
        X_seq = obs.view(-1, Lh, self.obs_dim)          # (H*B,L,obs_dim)
        X_prev = preview.view(-1, preview.size(-1))     # (H*B,P)
        Y = targets.view(-1, 1)                         # (H*B,1)

        N = X_seq.size(0)
        mb = min(self.cfg.minibatch_size, N)

        critic_losses: List[float] = []
        for _ in range(int(self.cfg.critic_updates)):
            idx = torch.randint(0, N, (mb,), device=self.device)
            xb_seq = X_seq[idx]
            xb_prev = X_prev[idx]
            yb = Y[idx]

            v1 = self.critic1(xb_seq, xb_prev)
            v2 = self.critic2(xb_seq, xb_prev)
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
