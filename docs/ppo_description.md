
# バッチ版 PPO 実装の数式解説メモ（$...$ 形式）

このドキュメントでは、実装済みの **バッチ版 PPO (`PPOAgentBatched`)** の処理フローを、  
数式とコードを対応づけながら説明します。数式は `$...$` / `$$...$$` 形式で書いています。

---

## 1. 問題設定と記号

- 時間ステップ: $t = 0, 1, \dots, T-1$
- 並列環境数: $B$
- 観測: $s_t^b \in \mathbb{R}^{d_{\text{obs}}}$ （環境 $b$ の時刻 $t$ の状態）
- 行動: $a_t^b \in \mathbb{R}^{d_{\text{act}}}$
- 報酬: $r_t^b \in \mathbb{R}$
- 終了フラグ: $d_t^b \in \{0, 1\}$ (`done`)
- 方策ネットワークのパラメータ: $\theta$
- 価値（バリュー）ネットワークのパラメータ: $\phi$  
  （実装では Actor-Critic の 1 モデル `ActorCritic` で $\theta$ と $\phi$ を共有）

環境から見た形状：

- `obs` : `(T, B, obs_dim)`
- `actions` : `(T, B, action_dim)`
- `rewards`, `dones`, `values`, `log_probs` : `(T, B, 1)`

コード上では、`RolloutBufferBatched` でこれをバッファリングしています。

```python
class RolloutBufferBatched:
    def __init__(
        self,
        num_steps: int,
        num_envs: int,
        obs_dim: int,
        action_dim: int,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.obs = torch.zeros(T, B, obs_dim, ...)
        self.actions = torch.zeros(T, B, action_dim, ...)
        self.rewards = torch.zeros(T, B, 1, ...)
        self.dones = torch.zeros(T, B, 1, ...)
        self.values = torch.zeros(T, B, 1, ...)
        self.log_probs = torch.zeros(T, B, 1, ...)
```

---

## 2. 方策（Actor）と価値（Critic）

### 2.1 正規分布ポリシー

連続行動の PPO では、方策を **多変量正規分布**

$$
\pi_\theta(a_t \mid s_t) = \mathcal{N}\big(\mu_\theta(s_t), \Sigma\big)
$$

としてモデル化しています。

実装では、

- ネットワーク本体 `body` で特徴量 $h_t = f_\theta(s_t)$ を計算
- `mu_head` で平均ベクトル $\mu_\theta(s_t)$ を出力
- 共分散行列 $\Sigma$ は対角行列とし、対角成分 $\sigma^2$ を `log_std` という学習パラメータから生成

```python
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim=2, hidden_dim=128):
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs):
        h = self.body(obs)
        mu = self.mu_head(h)              # μθ(s)
        std = torch.exp(self.log_std)     # σ（各次元一定）
        dist = D.Normal(mu, std)          # πθ(a|s)
        value = self.v_head(h)
        return dist, value
```

### 2.2 価値関数

価値関数 $V_\phi(s_t)$ は、同じ特徴量 $h_t$ から線形ヘッドで出力：

$$
V_\phi(s_t) = \text{v\_head}(h_t)
$$

コード：

```python
self.v_head = nn.Linear(hidden_dim, 1)

def forward(self, obs):
    h = self.body(obs)
    ...
    value = self.v_head(h)   # Vφ(s)
    return dist, value
```

### 2.3 行動サンプリングとログ確率

行動は $\pi_\theta(a_t \mid s_t)$ からサンプリングします。

$$
a_t \sim \pi_\theta(\cdot \mid s_t)
$$

コード：

```python
def act(self, obs):
    dist, value = self.forward(obs)
    action = dist.sample()  # a_t
    log_prob = dist.log_prob(action).sum(-1, keepdim=True)  # log πθ(a_t|s_t)
    return action, log_prob, value
```

`PPOAgentBatched` 側では、バッチ版 `act_batch` を用意して、  
環境からの観測 `(B, obs_dim)` 全体に対して一度に計算しています。

```python
@torch.no_grad()
def act_batch(self, obs_batch: torch.Tensor):
    obs_t = obs_batch.to(self.device, self.dtype)
    actions, log_probs, values = self.net.act(obs_t)
    return actions, log_probs, values
```

---

## 3. ロールアウト収集

1 回の PPO 更新あたりに、

- 時間方向に $T$ ステップ
- 環境 $B$ 個

分のロールアウトを集めます。

### 3.1 1ステップの追加

時刻 $t$ での観測 $s_t^b$、行動 $a_t^b$、報酬 $r_t^b$、`done` $d_t^b$、
価値 $V_\phi(s_t^b)$、ログ確率 $\log \pi_\theta(a_t^b \mid s_t^b)$ を  
まとめてバッファに追加します。

```python
def store_step(
    self,
    obs_batch, actions, rewards, dones, values, log_probs
):
    # obs_batch: (B, obs_dim)
    # actions  : (B, action_dim)
    # rewards  : (B,) or (B,1)
    # dones    : (B,) or (B,1)
    ...
    self.buffer.add(
        obs_batch.to(self.device, self.dtype),
        actions,
        rewards,   # (B,1)
        dones,     # (B,1)
        values,
        log_probs,
    )
```

`RolloutBufferBatched.add` は、それを内部の `(T,B,...)` 配列の `t` 行目に格納します：

```python
def add(self, obs_batch, actions_batch, rewards_batch, dones_batch, values_batch, log_probs_batch):
    t = self.step
    self.obs[t].copy_(obs_batch)
    self.actions[t].copy_(actions_batch)
    self.rewards[t].copy_(rewards_batch)
    self.dones[t].copy_(dones_batch)
    self.values[t].copy_(values_batch)
    self.log_probs[t].copy_(log_probs_batch)
    self.step += 1
```

---

## 4. GAE (Generalized Advantage Estimation)

PPO でよく使われる GAE は、  
「利得（advantage）をバイアスと分散のバランスを取りながら平滑化」する手法です。

### 4.1 TD 誤差 $\delta_t$

各時刻の TD 誤差を

$$
\delta_t^b
= r_t^b + \gamma V_\phi(s_{t+1}^b) \cdot (1 - d_{t+1}^b) - V_\phi(s_t^b)
$$

と定義します。  
ここで $d_t^b = 1$ ならエピソード終端なので、その先の価値は 0 にします。

実装では、最後の状態だけ `last_values` と `last_dones` を外部から渡し、
後ろ向きループの中で `next_values` / `next_nonterminal` を切り替えています。

```python
for t in reversed(range(T)):
    if t == T - 1:
        next_nonterminal = 1.0 - last_dones
        next_values = last_values
    else:
        next_nonterminal = 1.0 - self.dones[t + 1]
        next_values = self.values[t + 1]

    delta = self.rewards[t] + gamma * next_values * next_nonterminal - self.values[t]
    gae = delta + gamma * lam * next_nonterminal * gae
    advantages[t] = gae
    returns[t] = advantages[t] + self.values[t]
```

ここで `next_nonterminal = 1 - done` が $(1 - d_{t+1})$ に対応します。  
テンソルの形はすべて `(B,1)` です。

### 4.2 GAE の再帰

GAE の定義は

$$
A_t^b
= \delta_t^b + \gamma \lambda (1 - d_{t+1}^b) A_{t+1}^b
$$

です。  
上のループの中で、`gae` 変数に対して

$$
\text{gae} \leftarrow \delta_t^b + \gamma \lambda (1 - d_{t+1}^b) \cdot \text{gae}
$$

と更新しているのが GAE に対応しています。

最終的に `buffer.advantages` と `buffer.returns` に格納されます。

---

## 5. PPO の目的関数

PPO の基本的な目的関数は、
「古い方策 $\pi_{\theta_{\text{old}}}$ に対してクリップされた比率を使う」ものです。

### 5.1 方策比率 $r_t(\theta)$

$$
r_t^b(\theta)
= \frac{\pi_\theta(a_t^b \mid s_t^b)}{\pi_{\theta_{\text{old}}}(a_t^b \mid s_t^b)}
= \exp\big( \log \pi_\theta(a_t^b \mid s_t^b) - \log \pi_{\theta_{\text{old}}}(a_t^b \mid s_t^b) \big)
$$

実装では、`old_log_probs` をバッファに保存し、
新しい方策で計算した `new_log_probs` との差を exponentiate しています。

```python
dist, values = self.net.forward(batch_obs)
new_log_probs = dist.log_prob(batch_actions).sum(-1, keepdim=True)

ratio = torch.exp(new_log_probs - batch_old_log_probs)
```

### 5.2 クリップ付き目的関数

PPO の surrogate objective は

$$
L^{\text{CLIP}}(\theta)
= \mathbb{E}_{t,b}\Big[
    \min\big(
        r_t^b(\theta) A_t^b,\;\;
        \text{clip}(r_t^b(\theta), 1-\epsilon, 1+\epsilon) A_t^b
    \big)
\Big]
$$

です。ここで $\epsilon = \text{clip\_eps}$ です。

コード：

```python
surr1 = ratio * batch_advantages
surr2 = torch.clamp(
    ratio,
    1.0 - self.clip_eps,
    1.0 + self.clip_eps,
) * batch_advantages

policy_loss = -torch.min(surr1, surr2).mean()
```

- ここで `batch_advantages` は GAE で計算した $A_t^b$ を正規化したものです。
- `policy_loss` は **最大化したい** $L^{\text{CLIP}}$ に符号を変えて **最小化**する形にしています。

### 5.3 価値関数の損失

価値関数の損失は、シンプルな MSE：

$$
L^{\text{VF}}(\phi) = \mathbb{E}_{t,b}\big[(V_\phi(s_t^b) - R_t^b)^2\big]
$$

ここで $R_t^b$ は上で計算した `returns` に対応します。

```python
value_loss = nn.functional.mse_loss(values, batch_returns)
```

### 5.4 エントロピー正則化

方策の「広がり」を保つために、エントロピー

$$
\mathcal{H}[\pi_\theta(\cdot \mid s_t^b)]
$$

を最大化する項を追加します。

実装では、`dist.entropy()` を使って

```python
entropy = dist.entropy().sum(-1, keepdim=True)
entropy_loss = -entropy.mean()
```

とし、全体の loss に `ent_coef * entropy_loss` を足しています。  
（`entropy_loss` は **負号付き**なので、全体 loss の最小化 ≒ エントロピー最大化になります）

### 5.5 全体の損失関数

最終的な PPO の損失は

$$
L(\theta, \phi)
= L^{\text{CLIP}}(\theta)
+ c_1 L^{\text{VF}}(\phi)
+ c_2 \mathbb{E}[-\mathcal{H}(\pi_\theta)]
$$

を **最小化**します。コードでは：

```python
loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

self.optim.zero_grad()
loss.backward()
nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
self.optim.step()
```

- `vf_coef = c_1`
- `ent_coef = c_2`
- `max_grad_norm` で勾配クリッピング

---

## 6. バッチ PPO 更新フローまとめ

1. **ロールアウト収集**
   - for t in 0..T-1:
     1. 現在の観測 `obs_t`（形 `(B, obs_dim)`）をエージェントに渡す
     2. `actions, log_probs, values = agent.act_batch(obs_t)`
     3. 環境 `env.step(actions)` で `(next_obs, reward, done, ...)` を得る
     4. `agent.store_step(obs_t, actions, reward, done, values, log_probs)`
     5. `obs_t = next_obs`

2. **GAE とリターンの計算**
   - 最後の状態 `last_obs` で `last_values` を計算
   - `last_done` と合わせて `buffer.compute_gae(last_values, last_done, gamma, lam)`

3. **PPO 更新（複数エポック）**
   - バッファの `(T,B,...)` を flatten して `(N, ...)` にする（$N = T \times B$）
   - `advantages` を正規化
   - ランダムにシャッフルしてミニバッチごとに：
     - 新しい方策 $\pi_\theta$ で `new_log_probs`, `values`, `entropy` を計算
     - `policy_loss`, `value_loss`, `entropy_loss` を計算
     - 合成 loss を backprop + optimizer step

この一連の流れが、`PPOAgentBatched.update()` の中に対応しています。

```python
def update(self, last_obs: torch.Tensor, last_done: torch.Tensor):
    # 1) last_values 計算 → GAE / returns 計算
    ...
    self.buffer.compute_gae(...)

    # 2) flatten して advantages 正規化
    obs = self.buffer.obs.view(N, self.obs_dim)
    ...
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # 3) エポック×ミニバッチで PPO 更新
    for _ in range(self.epochs):
        indices = torch.randperm(N, device=self.device)
        for start in range(0, N, self.batch_size):
            ...
            dist, values = self.net.forward(batch_obs)
            new_log_probs = dist.log_prob(batch_actions).sum(-1, keepdim=True)
            entropy = dist.entropy().sum(-1, keepdim=True)

            ratio = torch.exp(new_log_probs - batch_old_log_probs)
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * batch_advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = nn.functional.mse_loss(values, batch_returns)
            entropy_loss = -entropy.mean()

            loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

            self.optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
            self.optim.step()
```

---

## 7. まとめ

- **`RolloutBufferBatched`** は `(T,B,...)` 形式で時系列と並列環境を同時に扱うバッファ
- **`ActorCritic`** は
  - 正規分布方策 $\pi_\theta(a\mid s) = \mathcal{N}(\mu_\theta(s), \Sigma)$
  - 価値関数 $V_\phi(s)$
  を同時に出力。
- **`PPOAgentBatched`** は
  - `act_batch` で B 環境分の行動・ログ確率・価値を一度に計算
  - GAE で $A_t$ と $R_t$ を求め
  - クリップ PPO 目的 $L^{\text{CLIP}}$ + 価値損失 + エントロピーで学習

という構造になっています。
