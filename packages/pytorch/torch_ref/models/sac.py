"""SAC (Haarnoja et al. 2018) on Pendulum-v1.

Stochastic tanh-squashed Gaussian actor, twin Q-networks, Polyak-averaged
target Q-networks, fixed entropy temperature α. Uses canonical
`gym.make("Pendulum-v1")` for env physics — reset state is pinned to
(theta=π, theta_dot=0) to mirror idris-gym's deterministic `MkP Pi 0.0`
(see `docs/develop/reference-alignment.md`).

Aligned with `Example.Sac` (Idris): separate actor + Q1 + Q2 networks
registered under distinct paramId scope prefixes on the Idris side, and
three separate Adam optimizers (one per network). Polyak soft target
update τ=0.005 applied every step, matching the Idris `polyakBlend`
wrapper that calls the C-backend `polyak_blend` FFI.
"""

from __future__ import annotations

import copy
import math
import random
import time
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_ref.training.runner import format_elapsed, mem_suffix

MAX_ACTION = 2.0  # Pendulum torque range
MAX_STEPS = 200  # gymnasium Pendulum-v1 default TimeLimit


def _reset_to_pi(env: gym.Env) -> np.ndarray:
    """Pin env state to (theta=π, theta_dot=0) and return the obs.

    Canonical Pendulum-v1 randomizes the init within theta ∈ [-π, π],
    theta_dot ∈ [-1, 1]; idris-gym uses deterministic worst-case
    `MkP Pi 0.0`. The torch reference pins to match — eliminating
    state-distribution differences from convergence comparisons.
    """
    env.unwrapped.state = np.array([math.pi, 0.0], dtype=np.float64)  # pyright: ignore[reportAttributeAccessIssue]
    th, dth = env.unwrapped.state  # pyright: ignore[reportAttributeAccessIssue]
    return np.array([math.cos(th), math.sin(th), dth], dtype=np.float64)


def _obs_tensor(obs: np.ndarray) -> Tensor:
    return torch.tensor(obs, dtype=torch.float64)


# ---------------------------------------------------------------------------
# Actor: tanh-squashed Gaussian policy
# ---------------------------------------------------------------------------


class Actor(nn.Module):
    def __init__(self, obs_dim: int = 3, hidden: int = 64) -> None:
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden, dtype=torch.float64)
        self.fc2 = nn.Linear(hidden, hidden, dtype=torch.float64)
        self.mean_head = nn.Linear(hidden, 1, dtype=torch.float64)
        self.log_std = nn.Parameter(torch.zeros(1, dtype=torch.float64))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = F.relu(self.fc2(F.relu(self.fc1(x))))
        mean = self.mean_head(h).squeeze(-1)
        log_std = torch.clamp(
            self.log_std.squeeze(0) + torch.zeros_like(mean), min=-5.0, max=2.0
        )
        return mean, log_std

    def sample(self, x: Tensor, rng: random.Random | None = None) -> tuple[Tensor, Tensor]:
        """Reparameterized sample: a = tanh(mean + std * eps) * MAX_ACTION.

        Returns (action, log_prob), both with gradient flow through the actor
        when x requires grad (via reparameterization trick).
        """
        mean, log_std = self(x)
        std = torch.exp(log_std)
        if rng is None:
            eps = torch.randn_like(mean)
        else:
            eps = torch.tensor(rng.gauss(0.0, 1.0), dtype=torch.float64)
        u = mean + std * eps  # pre-tanh
        a_squashed = torch.tanh(u)
        action = a_squashed * MAX_ACTION
        # Gaussian log-prob of u, corrected for tanh squash + action scaling:
        #   log_prob = gaussian_log_prob(u) - log(1 - tanh(u)^2 + ε) - log(MAX_ACTION)
        log_prob_u = -0.5 * ((u - mean) / std) ** 2 - log_std - 0.5 * math.log(2 * math.pi)
        log_prob = log_prob_u - torch.log(1.0 - a_squashed**2 + 1e-6) - math.log(MAX_ACTION)
        return action, log_prob


# ---------------------------------------------------------------------------
# Q-networks: (obs, action) → scalar
# ---------------------------------------------------------------------------


class QNet(nn.Module):
    def __init__(self, obs_dim: int = 3, act_dim: int = 1, hidden: int = 64) -> None:
        super().__init__()
        self.fc1 = nn.Linear(obs_dim + act_dim, hidden, dtype=torch.float64)
        self.fc2 = nn.Linear(hidden, hidden, dtype=torch.float64)
        self.head = nn.Linear(hidden, 1, dtype=torch.float64)

    def forward(self, obs: Tensor, action: Tensor) -> Tensor:
        a = action.unsqueeze(-1) if action.dim() == obs.dim() - 1 else action
        x = torch.cat([obs, a], dim=-1)
        h = F.relu(self.fc2(F.relu(self.fc1(x))))
        return self.head(h).squeeze(-1)


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buf: deque[tuple[list[float], float, float, list[float], bool]] = deque(
            maxlen=capacity
        )

    def push(self, obs: list[float], a: float, r: float, next_obs: list[float], done: bool) -> None:
        self.buf.append((obs, a, r, next_obs, done))

    def sample(self, n: int, rng: random.Random) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        batch = rng.sample(self.buf, n)
        obs = torch.tensor([b[0] for b in batch], dtype=torch.float64)
        actions = torch.tensor([b[1] for b in batch], dtype=torch.float64)
        rewards = torch.tensor([b[2] for b in batch], dtype=torch.float64)
        next_obs = torch.tensor([b[3] for b in batch], dtype=torch.float64)
        dones = torch.tensor([float(b[4]) for b in batch], dtype=torch.float64)
        return obs, actions, rewards, next_obs, dones

    def __len__(self) -> int:
        return len(self.buf)


# ---------------------------------------------------------------------------
# SAC update
# ---------------------------------------------------------------------------


def sac_update(
    actor: Actor, q1: QNet, q2: QNet, q1_target: QNet, q2_target: QNet,
    actor_opt: torch.optim.Optimizer, q1_opt: torch.optim.Optimizer,
    q2_opt: torch.optim.Optimizer, buffer: ReplayBuffer, batch_size: int,
    gamma: float, alpha: float, rng: random.Random,
) -> float:
    obs, actions, rewards, next_obs, dones = buffer.sample(batch_size, rng)
    with torch.no_grad():
        next_action, next_logp = actor.sample(next_obs)
        target_q = torch.min(q1_target(next_obs, next_action), q2_target(next_obs, next_action))
        target = rewards + gamma * (1.0 - dones) * (target_q - alpha * next_logp)

    # Q losses (Bellman MSE)
    q1_loss = F.mse_loss(q1(obs, actions), target)
    q2_loss = F.mse_loss(q2(obs, actions), target)
    q1_opt.zero_grad()
    q1_loss.backward()
    q1_opt.step()
    q2_opt.zero_grad()
    q2_loss.backward()
    q2_opt.step()

    # Actor loss: E[α * log π(a|s) - min(Q1(s,a), Q2(s,a))]
    sampled_action, logp = actor.sample(obs)
    q_min = torch.min(q1(obs, sampled_action), q2(obs, sampled_action))
    actor_loss = (alpha * logp - q_min).mean()
    actor_opt.zero_grad()
    actor_loss.backward()
    actor_opt.step()

    return float(actor_loss.item())


def polyak_update(target: nn.Module, online: nn.Module, tau: float) -> None:
    """target ← (1-τ)·target + τ·online, in-place. Matches the Idris
    backend's `polyak_blend` FFI semantics."""
    with torch.no_grad():
        for t, o in zip(target.parameters(), online.parameters(), strict=True):
            t.mul_(1.0 - tau).add_(o, alpha=tau)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_sac(
    total_steps: int = 30000, buffer_capacity: int = 100000, batch_size: int = 64,
    lr: float = 3e-4, gamma: float = 0.99, alpha: float = 0.2,
    warmup_steps: int = 1000, tau: float = 0.005,
    seed: int = 42, log_every: int = 2000,
) -> tuple[Actor, list[float]]:
    """Polyak soft update τ=0.005 every step, matching the Idris port which
    calls `polyakBlend` after each gradient step."""
    torch.manual_seed(seed)
    rng = random.Random(seed)
    actor = Actor()
    q1 = QNet()
    q2 = QNet()
    q1_target = copy.deepcopy(q1)
    q2_target = copy.deepcopy(q2)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=lr)
    q1_opt = torch.optim.Adam(q1.parameters(), lr=lr)
    q2_opt = torch.optim.Adam(q2.parameters(), lr=lr)
    buffer = ReplayBuffer(buffer_capacity)

    env = gym.make("Pendulum-v1")
    env.reset(seed=seed)
    obs_np = _reset_to_pi(env)
    history: list[float] = []
    ep_return = 0.0
    t_start = time.monotonic()
    for step in range(total_steps):
        obs = _obs_tensor(obs_np)
        if step < warmup_steps:
            action = rng.uniform(-MAX_ACTION, MAX_ACTION)
        else:
            with torch.no_grad():
                a_t, _ = actor.sample(obs)
                action = float(a_t.item())
        next_obs_np, reward, term, trunc, _ = env.step(np.array([action], dtype=np.float32))
        next_obs_np = next_obs_np.astype(np.float64)
        reward_f = float(reward)
        ep_return += reward_f
        done = bool(term or trunc)
        buffer.push(obs_np.tolist(), action, reward_f, next_obs_np.tolist(), done)
        obs_np = next_obs_np
        if done:
            history.append(ep_return)
            ep_return = 0.0
            env.reset()
            obs_np = _reset_to_pi(env)
        if len(buffer) >= max(batch_size, warmup_steps):
            sac_update(
                actor, q1, q2, q1_target, q2_target,
                actor_opt, q1_opt, q2_opt, buffer, batch_size, gamma, alpha, rng,
            )
            # Polyak soft update every step (matches Idris).
            polyak_update(q1_target, q1, tau)
            polyak_update(q2_target, q2, tau)
        if (step + 1) % log_every == 0:
            recent = history[-20:] or [0.0]
            last_ep = history[-1] if history else 0.0
            print(
                f"  {format_elapsed(t_start)} {step + 1}\tloss={-last_ep:.6f}"
                f"{mem_suffix()}\treturn={last_ep:.1f}"
                f"\trecent_20={sum(recent)/len(recent):.1f}"
            )
    return actor, history


def evaluate(actor: Actor, n_episodes: int = 20) -> float:
    env = gym.make("Pendulum-v1")
    env.reset(seed=0)
    total = 0.0
    for _ in range(n_episodes):
        env.reset()
        obs_np = _reset_to_pi(env)
        ep_return = 0.0
        for _ in range(MAX_STEPS):
            obs = _obs_tensor(obs_np)
            with torch.no_grad():
                mean, _ = actor(obs)
            action = float(torch.tanh(mean).item()) * MAX_ACTION
            obs_np, reward, term, trunc, _ = env.step(np.array([action], dtype=np.float32))
            obs_np = obs_np.astype(np.float64)
            ep_return += float(reward)
            if term or trunc:
                break
        total += ep_return
    return total / n_episodes


if __name__ == "__main__":
    print("=== SAC on Pendulum ===")
    actor, history = train_sac()
    avg = evaluate(actor)
    print(f"\nEval (20 episodes, greedy): avg_return={avg:.1f}")
    print(f"RESULT\tavg_return={avg:.1f}\tsteps={30000}\tseed=42")
