"""PPO (Schulman et al. 2017) on Acrobot-v1 (discrete-action, categorical policy).

Uses canonical `gym.make("Acrobot-v1")` for env physics (single-step
RK4 with dt=0.2). Reset state pinned to (0, 0, 0, 0) with float64 to
mirror `Gym.ClassicControl.Acrobot.reset = MkA 0 0 0 0` — canonical
Gymnasium randomizes each state component ~ U(-0.1, 0.1); both Idris
and torch_ref pin to deterministic hanging-down start.

Acrobot is the canonical "PPO clipped-surrogate demonstrates"
benchmark — discrete actions (3: -1/0/+1 torque), sparse reward
(-1/step, 0 at goal), longer horizon, 500-step TimeLimit.
"""

from __future__ import annotations

import random
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_ref.training.runner import format_elapsed, mem_suffix

MAX_STEPS = 500  # gymnasium Acrobot-v1 default TimeLimit


def make_acrobot_env(seed: int) -> gym.Env:
    """Create a seeded Acrobot-v1 env. Use `reset_to_zero` to pin the
    initial state to (0,0,0,0) after each `env.reset()`."""
    env = gym.make("Acrobot-v1")
    env.reset(seed=seed)
    return env


def reset_to_zero(env: gym.Env) -> np.ndarray:
    """Pin env state to (0, 0, 0, 0) (hanging-down) and return obs (float64).

    Canonical Acrobot-v1 randomizes each component ~ U(-0.1, 0.1).
    Pinning to zero matches idris-gym `MkA 0 0 0 0`.
    """
    env.unwrapped.state = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)  # pyright: ignore[reportAttributeAccessIssue]
    # Obs is [cos(th1), sin(th1), cos(th2), sin(th2), dth1, dth2].
    return np.array([1.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def obs_tensor(obs: np.ndarray) -> Tensor:
    return torch.tensor(obs, dtype=torch.float64)


class Actor(nn.Module):
    """Categorical policy over 3 actions (-1, 0, +1 torque)."""

    def __init__(self, obs_dim: int = 6, hidden: int = 64, num_actions: int = 3) -> None:
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden, dtype=torch.float64)
        self.fc2 = nn.Linear(hidden, hidden, dtype=torch.float64)
        self.head = nn.Linear(hidden, num_actions, dtype=torch.float64)

    def forward(self, x: Tensor) -> Tensor:
        h = torch.tanh(self.fc2(torch.tanh(self.fc1(x))))
        return self.head(h)


class Critic(nn.Module):
    def __init__(self, obs_dim: int = 6, hidden: int = 64) -> None:
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden, dtype=torch.float64)
        self.fc2 = nn.Linear(hidden, hidden, dtype=torch.float64)
        self.head = nn.Linear(hidden, 1, dtype=torch.float64)

    def forward(self, x: Tensor) -> Tensor:
        h = torch.tanh(self.fc2(torch.tanh(self.fc1(x))))
        return self.head(h).squeeze(-1)


def sample_action(actor: Actor, obs: Tensor, rng: random.Random) -> tuple[int, float]:
    """Sample a discrete action from the categorical policy. Returns (action, log π(a|s))."""
    with torch.no_grad():
        logits = actor(obs)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        u = rng.random()
        cum = 0.0
        a = 0
        for i, p in enumerate(probs.tolist()):
            cum += p
            if u <= cum:
                a = i
                break
        else:
            a = probs.shape[0] - 1
        lp = float(log_probs[a].item())
    return a, lp


def gae(
    rewards: list[float], values: list[float], dones: list[bool], bootstrap: float,
    gamma: float, lam: float,
) -> tuple[list[float], list[float]]:
    advantages: list[float] = []
    a_next = 0.0
    v_next = bootstrap
    for t in reversed(range(len(rewards))):
        mask = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * v_next * mask - values[t]
        a = delta + gamma * lam * mask * a_next
        advantages.append(a)
        a_next = a
        v_next = values[t]
    advantages.reverse()
    returns = [adv + v for adv, v in zip(advantages, values, strict=True)]
    return advantages, returns


def collect_rollout(
    actor: Actor, critic: Critic, env: gym.Env, obs_np: np.ndarray,
    n_steps: int, max_ep_len: int, rng: random.Random,
) -> tuple[
    list[Tensor], list[int], list[float], list[float], list[float], list[bool],
    np.ndarray, list[float],
]:
    obs_list: list[Tensor] = []
    act_list: list[int] = []
    lp_list: list[float] = []
    rew_list: list[float] = []
    val_list: list[float] = []
    done_list: list[bool] = []
    ep_returns: list[float] = []
    ep_sum = 0.0
    ep_len = 0
    for _ in range(n_steps):
        obs = obs_tensor(obs_np)
        action, lp = sample_action(actor, obs, rng)
        with torch.no_grad():
            v = float(critic(obs).item())
        next_obs_np, reward, term, trunc, _ = env.step(action)
        next_obs_np = next_obs_np.astype(np.float64)
        ep_sum += float(reward)
        ep_len += 1
        truncate = ep_len >= max_ep_len
        done = bool(term or trunc or truncate)
        obs_list.append(obs)
        act_list.append(action)
        lp_list.append(lp)
        rew_list.append(float(reward))
        val_list.append(v)
        done_list.append(done)
        if done:
            ep_returns.append(ep_sum)
            ep_sum = 0.0
            ep_len = 0
            env.reset()
            obs_np = reset_to_zero(env)
        else:
            obs_np = next_obs_np
    return obs_list, act_list, lp_list, rew_list, val_list, done_list, obs_np, ep_returns


def ppo_update(
    actor: Actor, critic: Critic, actor_opt: torch.optim.Optimizer,
    critic_opt: torch.optim.Optimizer, obs: Tensor, actions: Tensor,
    old_log_probs: Tensor, advantages: Tensor, returns: Tensor, clip_eps: float,
    entropy_coef: float, k_epochs: int, batch_size: int, rng: random.Random,
) -> None:
    n = obs.shape[0]
    indices = list(range(n))
    for _ in range(k_epochs):
        rng.shuffle(indices)
        for start in range(0, n, batch_size):
            batch = indices[start : start + batch_size]
            o_b = obs[batch]
            a_b = actions[batch]
            lp_old = old_log_probs[batch]
            adv_b = advantages[batch]
            ret_b = returns[batch]
            logits = actor(o_b)
            log_probs = F.log_softmax(logits, dim=-1)
            lp_new = log_probs.gather(1, a_b.unsqueeze(1)).squeeze(1)
            ratio = torch.exp(lp_new - lp_old)
            s1 = ratio * adv_b
            s2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_b
            policy_loss = -torch.min(s1, s2).mean()
            entropy = -(torch.exp(log_probs) * log_probs).sum(dim=-1).mean()
            actor_loss = policy_loss - entropy_coef * entropy
            actor_opt.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
            actor_opt.step()
            v_b = critic(o_b)
            critic_loss = F.mse_loss(v_b, ret_b)
            critic_opt.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
            critic_opt.step()


def train_ppo(
    total_rollouts: int = 100, rollout_steps: int = 1024, lr: float = 3e-4,
    gamma: float = 0.99, lam: float = 0.95, clip_eps: float = 0.2,
    entropy_coef: float = 0.01, k_epochs: int = 10, batch_size: int = 64,
    max_ep_len: int = MAX_STEPS, seed: int = 42, log_every: int = 10,
) -> tuple[Actor, list[float]]:
    torch.manual_seed(seed)
    rng = random.Random(seed)
    actor = Actor()
    critic = Critic()
    actor_opt = torch.optim.Adam(actor.parameters(), lr=lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=lr)
    env = make_acrobot_env(seed)
    obs_np = reset_to_zero(env)
    history: list[float] = []
    t_start = time.monotonic()
    for r in range(total_rollouts):
        obs_l, act_l, lp_l, rew_l, val_l, done_l, obs_np, ep_rets = collect_rollout(
            actor, critic, env, obs_np, rollout_steps, max_ep_len, rng,
        )
        with torch.no_grad():
            bootstrap = (
                0.0 if (done_l and done_l[-1])
                else float(critic(obs_tensor(obs_np)).item())
            )
        advs, rets = gae(rew_l, val_l, done_l, bootstrap, gamma, lam)
        obs_t = torch.stack(obs_l)
        act_t = torch.tensor(act_l, dtype=torch.long)
        lp_t = torch.tensor(lp_l, dtype=torch.float64)
        adv_t = torch.tensor(advs, dtype=torch.float64)
        ret_t = torch.tensor(rets, dtype=torch.float64)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        ppo_update(
            actor, critic, actor_opt, critic_opt, obs_t, act_t, lp_t, adv_t, ret_t,
            clip_eps, entropy_coef, k_epochs, batch_size, rng,
        )
        history.extend(ep_rets)
        if (r + 1) % log_every == 0:
            recent = history[-50:] or [0.0]
            last_ep = history[-1] if history else 0.0
            print(
                f"  {format_elapsed(t_start)} {r + 1}\tloss={-last_ep:.6f}"
                f"{mem_suffix()}\treturn={last_ep:.1f}"
                f"\trecent_50={sum(recent)/len(recent):.1f}"
            )
    return actor, history


def evaluate(actor: Actor, n_episodes: int = 20, max_ep_len: int = MAX_STEPS) -> float:
    env = make_acrobot_env(seed=0)
    total = 0.0
    for _ in range(n_episodes):
        env.reset()
        obs_np = reset_to_zero(env)
        ep_return = 0.0
        for _ in range(max_ep_len):
            obs = obs_tensor(obs_np)
            with torch.no_grad():
                logits = actor(obs)
            action = int(torch.argmax(logits).item())
            next_obs_np, reward, term, trunc, _ = env.step(action)
            ep_return += float(reward)
            if term or trunc:
                break
            obs_np = next_obs_np.astype(np.float64)
        total += ep_return
    return total / n_episodes


if __name__ == "__main__":
    print("=== PPO on Acrobot ===")
    actor, history = train_ppo()
    avg = evaluate(actor)
    print(f"\nEval (20 episodes, greedy): avg_return={avg:.1f}")
    print(f"RESULT\tavg_return={avg:.1f}\trollouts={100}\tseed=42")
