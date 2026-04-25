"""PPO (Schulman et al. 2017) on Pendulum-v1.

Gaussian policy with clipped surrogate objective, GAE advantages,
multi-epoch mini-batch updates. Self-contained Pendulum physics
matching `Gym.ClassicControl.Pendulum`.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Pendulum environment (matches Idris Gym.ClassicControl.Pendulum)
# ---------------------------------------------------------------------------

GRAVITY = 10.0
MASS_POLE = 1.0
POLE_LEN = 1.0
MAX_TORQUE = 2.0
MAX_SPEED = 8.0
DT = 0.05
MAX_STEPS = 200


@dataclass
class PendulumState:
    theta: float = math.pi  # hanging down
    theta_dot: float = 0.0


def angle_normalize(x: float) -> float:
    two_pi = 2.0 * math.pi
    return x - two_pi * math.floor((x + math.pi) / two_pi)


def pendulum_step(s: PendulumState, action: float) -> tuple[float, PendulumState, bool]:
    torque = max(-MAX_TORQUE, min(MAX_TORQUE, action))
    th_norm = angle_normalize(s.theta)
    reward = -(th_norm**2 + 0.1 * s.theta_dot**2 + 0.001 * torque**2)
    dth1 = s.theta_dot + (
        3.0 * GRAVITY / (2.0 * POLE_LEN) * math.sin(s.theta)
        + 3.0 / (MASS_POLE * POLE_LEN**2) * torque
    ) * DT
    dth2 = max(-MAX_SPEED, min(MAX_SPEED, dth1))
    th1 = s.theta + dth2 * DT
    return reward, PendulumState(th1, dth2), False


def observe(s: PendulumState) -> Tensor:
    return torch.tensor(
        [math.cos(s.theta), math.sin(s.theta), s.theta_dot], dtype=torch.float64
    )


# ---------------------------------------------------------------------------
# Actor-critic (separate nets, Gaussian policy)
# ---------------------------------------------------------------------------


class Actor(nn.Module):
    """Gaussian policy with state-independent log_std (common PPO-for-Pendulum choice)."""

    def __init__(self, obs_dim: int = 3, hidden: int = 64) -> None:
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden, dtype=torch.float64)
        self.fc2 = nn.Linear(hidden, hidden, dtype=torch.float64)
        self.mean_head = nn.Linear(hidden, 1, dtype=torch.float64)
        # State-independent log_std, initialized so initial std ≈ 1
        self.log_std = nn.Parameter(torch.zeros(1, dtype=torch.float64))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = torch.tanh(self.fc2(torch.tanh(self.fc1(x))))
        mean = self.mean_head(h).squeeze(-1)
        log_std = self.log_std.squeeze(0) + torch.zeros_like(mean)
        return mean, log_std


class Critic(nn.Module):
    def __init__(self, obs_dim: int = 3, hidden: int = 64) -> None:
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden, dtype=torch.float64)
        self.fc2 = nn.Linear(hidden, hidden, dtype=torch.float64)
        self.head = nn.Linear(hidden, 1, dtype=torch.float64)

    def forward(self, x: Tensor) -> Tensor:
        h = torch.tanh(self.fc2(torch.tanh(self.fc1(x))))
        return self.head(h).squeeze(-1)


def gaussian_log_prob(mean: Tensor, log_std: Tensor, action: Tensor) -> Tensor:
    std = torch.exp(log_std)
    return -0.5 * ((action - mean) / std) ** 2 - log_std - 0.5 * math.log(2 * math.pi)


def sample_action(actor: Actor, obs: Tensor, rng: random.Random) -> tuple[float, float]:
    with torch.no_grad():
        mean, log_std = actor(obs)
        std = torch.exp(log_std)
        eps = rng.gauss(0.0, 1.0)
        action = float(mean.item()) + float(std.item()) * eps
        lp = float(
            gaussian_log_prob(
                mean, log_std, torch.tensor(action, dtype=torch.float64)
            ).item()
        )
    return action, lp


# ---------------------------------------------------------------------------
# GAE
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# PPO training
# ---------------------------------------------------------------------------


def collect_rollout(
    actor: Actor, critic: Critic, state: PendulumState, n_steps: int, rng: random.Random,
) -> tuple[
    list[Tensor], list[float], list[float], list[float], list[float], list[bool],
    PendulumState, list[float],
]:
    obs_list: list[Tensor] = []
    act_list: list[float] = []
    lp_list: list[float] = []
    rew_list: list[float] = []
    val_list: list[float] = []
    done_list: list[bool] = []
    ep_returns: list[float] = []
    ep_sum = 0.0
    ep_len = 0
    for _ in range(n_steps):
        obs = observe(state)
        action, lp = sample_action(actor, obs, rng)
        with torch.no_grad():
            v = float(critic(obs).item())
        reward, state, _ = pendulum_step(state, action)
        ep_sum += reward
        ep_len += 1
        # Use fixed-length episodes (200 steps) since Pendulum has no natural termination
        done = ep_len >= MAX_STEPS
        obs_list.append(obs)
        act_list.append(action)
        lp_list.append(lp)
        rew_list.append(reward)
        val_list.append(v)
        done_list.append(done)
        if done:
            ep_returns.append(ep_sum)
            ep_sum = 0.0
            ep_len = 0
            state = PendulumState()
    return obs_list, act_list, lp_list, rew_list, val_list, done_list, state, ep_returns


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
            mean, log_std = actor(o_b)
            lp_new = gaussian_log_prob(mean, log_std, a_b)
            ratio = torch.exp(lp_new - lp_old)
            s1 = ratio * adv_b
            s2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_b
            policy_loss = -torch.min(s1, s2).mean()
            entropy = (0.5 * math.log(2 * math.pi * math.e) + log_std).mean()
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
    total_rollouts: int = 200, rollout_steps: int = 2048, lr: float = 3e-4,
    gamma: float = 0.99, lam: float = 0.95, clip_eps: float = 0.2,
    entropy_coef: float = 0.0, k_epochs: int = 10, batch_size: int = 64,
    seed: int = 42, log_every: int = 10,
) -> tuple[Actor, list[float]]:
    torch.manual_seed(seed)
    rng = random.Random(seed)
    actor = Actor()
    critic = Critic()
    actor_opt = torch.optim.Adam(actor.parameters(), lr=lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=lr)
    state = PendulumState()
    history: list[float] = []
    for r in range(total_rollouts):
        obs_l, act_l, lp_l, rew_l, val_l, done_l, state, ep_rets = collect_rollout(
            actor, critic, state, rollout_steps, rng
        )
        with torch.no_grad():
            bootstrap = float(critic(observe(state)).item())
        advs, rets = gae(rew_l, val_l, done_l, bootstrap, gamma, lam)
        obs_t = torch.stack(obs_l)
        act_t = torch.tensor(act_l, dtype=torch.float64)
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
            print(
                f"  rollout {r + 1:4d}  eps={len(ep_rets):2d}  "
                f"recent_50_return={sum(recent)/len(recent):.1f}"
            )
    return actor, history


def evaluate(actor: Actor, n_episodes: int = 20) -> float:
    total = 0.0
    for _ in range(n_episodes):
        state = PendulumState()
        ep_return = 0.0
        for _ in range(MAX_STEPS):
            obs = observe(state)
            with torch.no_grad():
                mean, _ = actor(obs)
            action = float(mean.item())
            reward, state, _ = pendulum_step(state, action)
            ep_return += reward
        total += ep_return
    return total / n_episodes


if __name__ == "__main__":
    print("=== PPO on Pendulum ===")
    actor, history = train_ppo()
    avg = evaluate(actor)
    print(f"\nEval (20 episodes, greedy): avg_return={avg:.1f}")
    print(f"RESULT\tavg_return={avg:.1f}\trollouts={200}\tseed=42")
