import torch
import torch.nn as nn
import torch.distributions as D


class RolloutBufferBatched:
    """
    形状 (T, B, ...) でロールアウトを保持するバッチ版バッファ。

    num_steps: T （1エピソードあたりのステップ数。全環境共通）
    num_envs: B （並列環境数）
    """

    def __init__(
        self,
        num_steps: int,
        num_envs: int,
        obs_dim: int,
        action_dim: int,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        self.dtype = dtype

        T, B = num_steps, num_envs

        self.obs = torch.zeros(T, B, obs_dim, device=device, dtype=dtype)
        self.actions = torch.zeros(T, B, action_dim, device=device, dtype=dtype)
        self.rewards = torch.zeros(T, B, 1, device=device, dtype=dtype)
        self.dones = torch.zeros(T, B, 1, device=device, dtype=dtype)
        self.values = torch.zeros(T, B, 1, device=device, dtype=dtype)
        self.log_probs = torch.zeros(T, B, 1, device=device, dtype=dtype)

        self.advantages = torch.zeros(T, B, 1, device=device, dtype=dtype)
        self.returns = torch.zeros(T, B, 1, device=device, dtype=dtype)

        self.step = 0  # 現在何ステップ目まで埋まっているか

    def reset(self):
        self.step = 0
        # 中身は上書きされるので必須ではないが、気になるならここで zero_() してもよい

    def add(
        self,
        obs_batch: torch.Tensor,      # (B, obs_dim)
        actions_batch: torch.Tensor,  # (B, action_dim)
        rewards_batch: torch.Tensor,  # (B, 1)
        dones_batch: torch.Tensor,    # (B, 1) float(0 or 1)
        values_batch: torch.Tensor,   # (B, 1)
        log_probs_batch: torch.Tensor # (B, 1)
    ):
        """
        1ステップ分（B 環境分）のデータを T の現在位置に追加
        """
        t = self.step
        assert t < self.num_steps, f"Rollout buffer overflow: step={t}, num_steps={self.num_steps}"

        self.obs[t].copy_(obs_batch)
        self.actions[t].copy_(actions_batch)
        self.rewards[t].copy_(rewards_batch)
        self.dones[t].copy_(dones_batch)
        self.values[t].copy_(values_batch)
        self.log_probs[t].copy_(log_probs_batch)

        self.step += 1

    def is_full(self) -> bool:
        return self.step >= self.num_steps

    def compute_gae(
        self,
        last_values: torch.Tensor,  # (B, 1)
        last_dones: torch.Tensor,   # (B, 1) float(0 or 1)
        gamma: float,
        lam: float,
    ):
        """
        GAE をバッチで計算。
        形状は (T, B, 1) のまま保持。
        """
        T, B = self.num_steps, self.num_envs
        assert self.step == T, f"buffer not full: step={self.step}, num_steps={T}"

        advantages = torch.zeros(T, B, 1, device=self.device, dtype=self.dtype)
        returns = torch.zeros(T, B, 1, device=self.device, dtype=self.dtype)

        # SB3 と同じロジック:
        # lastgaelam = 0 から始めて、後ろ向きにスイープ
        gae = torch.zeros(B, 1, device=self.device, dtype=self.dtype)

        for t in reversed(range(T)):
            if t == T - 1:
                next_nonterminal = 1.0 - last_dones  # (B,1)
                next_values = last_values            # (B,1)
            else:
                # next_nonterminal = 1.0 - self.dones[t + 1]  # (B,1)
                next_nonterminal = 1.0 - self.dones[t]  # (B,1)
                next_values = self.values[t + 1]             # (B,1)

            # delta: TD誤差
            delta = self.rewards[t] + gamma * next_values * next_nonterminal - self.values[t]
            gae = delta + gamma * lam * next_nonterminal * gae
            advantages[t] = gae
            returns[t] = advantages[t] + self.values[t]

        self.advantages.copy_(advantages)
        self.returns.copy_(returns)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int = 2, hidden_dim: int = 128):
        super().__init__()

        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        # policy head
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        # value head
        self.v_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor):
        """
        obs: (N, obs_dim)
        戻り:
            dist : Normal(μ, σ) with shape (..., action_dim)
            value: (N, 1)
        """
        h = self.body(obs)
        mu = self.mu_head(h)             # (N, action_dim)
        std = torch.exp(self.log_std)    # (action_dim,)
        dist = D.Normal(mu, std)
        value = self.v_head(h)           # (N, 1)
        return dist, value

    def act(self, obs: torch.Tensor):
        """
        obs: (N, obs_dim)
        戻り:
            actions : (N, action_dim)
            log_prob: (N, 1)
            value   : (N, 1)
        """
        dist, value = self.forward(obs)
        action = dist.sample()                 # (N, action_dim)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)  # (N,1)
        return action, log_prob, value


class PPOAgentBatched:
    """
    BatchedPathTrackingEnvFrenet など B 個の並列環境向け PPO エージェント。

    - RolloutBufferBatched で (T, B, ...) 形式のロールアウトを収集
    - update() で PPO 更新（全サンプルをミニバッチに分けて SGD）
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int = 2,
        num_envs: int = 4,
        rollout_steps: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_eps: float = 0.2,
        epochs: int = 10,
        batch_size: int = 64,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.device = device
        self.dtype = dtype

        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm

        self.num_envs = num_envs
        self.rollout_steps = rollout_steps
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.net = ActorCritic(obs_dim, action_dim).to(device)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=lr)

        self.buffer = RolloutBufferBatched(
            num_steps=rollout_steps,
            num_envs=num_envs,
            obs_dim=obs_dim,
            action_dim=action_dim,
            device=device,
            dtype=dtype,
        )

    @torch.no_grad()
    def act_batch(self, obs_batch: torch.Tensor):
        """
        obs_batch: (B, obs_dim) の Tensor （環境からそのまま）
        戻り:
            actions  : (B, action_dim)
            log_probs: (B, 1)
            values   : (B, 1)
        """
        assert obs_batch.dim() == 2 and obs_batch.size(1) == self.obs_dim
        obs_t = obs_batch.to(self.device, self.dtype)
        actions, log_probs, values = self.net.act(obs_t)
        return actions, log_probs, values

    def store_step(
        self,
        obs_batch: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
        log_probs: torch.Tensor,
    ):
        """
        1環境ステップ分（B環境まとめて）をバッファに追加。
        引数はすべて torch.Tensor で渡す。
        shapes:
            obs_batch: (B, obs_dim)
            actions  : (B, action_dim)
            rewards  : (B,) or (B,1)
            dones    : (B,) or (B,1)  bool or float(0/1)
            values   : (B,1)
            log_probs: (B,1)
        """
        B = self.num_envs
        assert obs_batch.shape == (B, self.obs_dim)
        assert actions.shape == (B, self.action_dim)
        assert values.shape == (B, 1)
        assert log_probs.shape == (B, 1)

        # rewards, dones を (B,1) float に整形
        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(1)
        if dones.dim() == 1:
            dones = dones.unsqueeze(1)
        rewards = rewards.to(self.device, self.dtype)
        dones = dones.to(self.device, self.dtype)

        self.buffer.add(
            obs_batch.to(self.device, self.dtype),
            actions,
            rewards,
            dones,
            values,
            log_probs,
        )

    def update(self, last_obs: torch.Tensor, last_done: torch.Tensor):
        """
        RolloutBufferBatched に rollouts が T ステップ分溜まったあとに呼ぶ。

        last_obs : (B, obs_dim) 最後の状態
        last_done: (B,)  bool  最後の状態が terminal か
        """
        B = self.num_envs
        T = self.rollout_steps

        assert self.buffer.step == T, f"buffer not full: step={self.buffer.step}, T={T}"
        assert last_obs.shape == (B, self.obs_dim)
        assert last_done.shape[0] == B

        # GAE 計算用の last_value
        last_obs_t = last_obs.to(self.device, self.dtype)
        last_done_t = last_done.view(B, 1).to(self.device, self.dtype)

        with torch.no_grad():
            _, last_values = self.net.forward(last_obs_t)  # (B,1)

        self.buffer.compute_gae(
            last_values=last_values,
            last_dones=last_done_t,
            gamma=self.gamma,
            lam=self.lam,
        )

        # ===== PPO 更新 =====
        obs = self.buffer.obs          # (T,B,obs_dim)
        actions = self.buffer.actions  # (T,B,action_dim)
        old_log_probs = self.buffer.log_probs  # (T,B,1)
        returns = self.buffer.returns         # (T,B,1)
        advantages = self.buffer.advantages   # (T,B,1)

        # flatten: (T*B, ...)
        N = T * B
        obs = obs.view(N, self.obs_dim)
        actions = actions.view(N, self.action_dim)
        old_log_probs = old_log_probs.view(N, 1)
        returns = returns.view(N, 1)
        advantages = advantages.view(N, 1)

        # advantages を正規化
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.epochs):
            indices = torch.randperm(N, device=self.device)
            for start in range(0, N, self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]

                batch_obs = obs[batch_idx]              # (Mb, obs_dim)
                batch_actions = actions[batch_idx]      # (Mb, action_dim)
                batch_old_log_probs = old_log_probs[batch_idx]  # (Mb,1)
                batch_returns = returns[batch_idx]      # (Mb,1)
                batch_advantages = advantages[batch_idx]  # (Mb,1)

                dist, values = self.net.forward(batch_obs)
                new_log_probs = dist.log_prob(batch_actions).sum(-1, keepdim=True)  # (Mb,1)
                entropy = dist.entropy().sum(-1, keepdim=True)                      # (Mb,1)

                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.clip_eps,
                    1.0 + self.clip_eps,
                ) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.functional.mse_loss(values, batch_returns)
                entropy_loss = -entropy.mean()

                loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optim.step()

        # バッファをクリア（step だけリセット）
        self.buffer.reset()
