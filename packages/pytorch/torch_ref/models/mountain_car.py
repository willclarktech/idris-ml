"""DQN on MountainCar-v0 with reward shaping.

Self-contained MountainCar physics matching Gymnasium's constants and the
Idris idris-gym implementation. Sparse reward (-1/step until goal) is
augmented with velocity-magnitude shaping `r' = r + shaping * |v'|` to
provide a dense intermediate signal — the agent learns to build kinetic
energy as the proven precursor to reaching the goal.

Aligned with `Example.MountainCar` (Idris): same architecture (2 -> 64 ->
64 -> 3), same defaults (lr=1e-3, gamma=0.99, batch=64, buffer=50K,
target_sync=200, eps 1.0->0.05 over 50K steps, shaping=10.0), same eval
protocol (30 greedy episodes).
"""

from __future__ import annotations

import copy
import math
import random
from collections import deque
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# MountainCar environment (Gymnasium-compatible constants)
# ---------------------------------------------------------------------------

MIN_POSITION = -1.2
MAX_POSITION = 0.6
MAX_SPEED = 0.07
GOAL_POSITION = 0.5
MC_FORCE = 0.001
GRAVITY = 0.0025
MAX_STEPS = 200


@dataclass
class MCState:
    pos: float = -0.5
    vel: float = 0.0


def mc_step(state: MCState, action: int) -> tuple[float, MCState, bool]:
    """One MountainCar physics step. Action 0=push left, 1=no push, 2=push right.

    Returns (reward, next_state, done) — done is True when the goal is reached.
    """
    a = float(action) - 1.0  # -1, 0, +1
    vel = state.vel + a * MC_FORCE - math.cos(3.0 * state.pos) * GRAVITY
    vel = max(-MAX_SPEED, min(MAX_SPEED, vel))
    pos = state.pos + vel
    pos = max(MIN_POSITION, min(MAX_POSITION, pos))
    if pos == MIN_POSITION and vel < 0.0:
        vel = 0.0
    next_state = MCState(pos=pos, vel=vel)
    done = pos >= GOAL_POSITION
    return -1.0, next_state, done


def observe(state: MCState) -> Tensor:
    return torch.tensor([state.pos, state.vel], dtype=torch.float64)


# ---------------------------------------------------------------------------
# Q-network
# ---------------------------------------------------------------------------


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int = 2, num_actions: int = 3, hidden: int = 64) -> None:
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden, dtype=torch.float64)
        self.fc2 = nn.Linear(hidden, hidden, dtype=torch.float64)
        self.fc3 = nn.Linear(hidden, num_actions, dtype=torch.float64)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buf: deque[tuple[list[float], int, float, list[float], bool]] = deque(
            maxlen=capacity
        )

    def push(
        self,
        obs: list[float],
        action: int,
        reward: float,
        next_obs: list[float],
        done: bool,
    ) -> None:
        self.buf.append((obs, action, reward, next_obs, done))

    def sample(
        self, n: int, rng: random.Random
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        batch = rng.sample(self.buf, n)
        obs = torch.tensor([b[0] for b in batch], dtype=torch.float64)
        actions = torch.tensor([b[1] for b in batch], dtype=torch.long)
        rewards = torch.tensor([b[2] for b in batch], dtype=torch.float64)
        next_obs = torch.tensor([b[3] for b in batch], dtype=torch.float64)
        dones = torch.tensor([float(b[4]) for b in batch], dtype=torch.float64)
        return obs, actions, rewards, next_obs, dones

    def __len__(self) -> int:
        return len(self.buf)


# ---------------------------------------------------------------------------
# DQN
# ---------------------------------------------------------------------------


def eps_greedy_action(q: QNetwork, obs: Tensor, epsilon: float, rng: random.Random) -> int:
    if rng.random() < epsilon:
        return rng.randrange(3)
    with torch.no_grad():
        return int(torch.argmax(q(obs)).item())


def linear_epsilon(
    step: int, start: float = 1.0, end: float = 0.05, decay_steps: int = 50000
) -> float:
    frac = min(step / decay_steps, 1.0)
    return start + frac * (end - start)


def dqn_update(
    q: QNetwork,
    target: QNetwork,
    optimizer: torch.optim.Optimizer,
    buffer: ReplayBuffer,
    batch_size: int,
    gamma: float,
    rng: random.Random,
) -> float:
    """One gradient step. Returns the loss value."""
    obs, actions, rewards, next_obs, dones = buffer.sample(batch_size, rng)
    q_vals = q(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        target_max = target(next_obs).max(dim=1).values
        target_vals = rewards + gamma * target_max * (1.0 - dones)
    loss = F.mse_loss(q_vals, target_vals)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(q.parameters(), 10.0)
    optimizer.step()
    return float(loss.item())


def dqn_episode(
    q: QNetwork,
    target: QNetwork,
    optimizer: torch.optim.Optimizer,
    buffer: ReplayBuffer,
    step_count: int,
    batch_size: int,
    gamma: float,
    target_sync_every: int,
    eps_start: float,
    eps_end: float,
    eps_decay: int,
    shaping: float,
    rng: random.Random,
) -> tuple[int, float]:
    """One episode. Returns (new_step_count, raw_episodic_return).

    Reward used for buffer / training is the *shaped* reward (raw + shaping*|v'|).
    The returned ep_return is the *raw* reward sum so eval metrics align with
    standard MountainCar reporting (-200 floor, ~-110 for reliable solver).
    """
    state = MCState()
    ep_return = 0.0
    for _ in range(MAX_STEPS):
        obs = observe(state)
        epsilon = linear_epsilon(step_count, eps_start, eps_end, eps_decay)
        action = eps_greedy_action(q, obs, epsilon, rng)
        raw_reward, next_state, done = mc_step(state, action)
        ep_return += raw_reward
        shaped_reward = raw_reward + shaping * abs(next_state.vel)
        buffer.push(
            obs.tolist(),
            action,
            shaped_reward,
            observe(next_state).tolist(),
            done,
        )
        state = next_state
        step_count += 1

        if len(buffer) >= batch_size:
            dqn_update(q, target, optimizer, buffer, batch_size, gamma, rng)

        if step_count % target_sync_every == 0:
            target.load_state_dict(q.state_dict())

        if done:
            break
    return step_count, ep_return


def train_dqn(
    episodes: int = 1000,
    lr: float = 1e-3,
    gamma: float = 0.99,
    batch_size: int = 64,
    buffer_capacity: int = 50000,
    target_sync_every: int = 200,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    eps_decay: int = 50000,
    shaping: float = 10.0,
    seed: int = 42,
    log_every: int = 50,
) -> tuple[QNetwork, list[float]]:
    torch.manual_seed(seed)
    rng = random.Random(seed)
    q = QNetwork()
    target = copy.deepcopy(q)
    optimizer = torch.optim.Adam(q.parameters(), lr=lr)
    buffer = ReplayBuffer(buffer_capacity)
    history: list[float] = []
    step_count = 0
    for ep in range(episodes):
        step_count, ep_return = dqn_episode(
            q, target, optimizer, buffer, step_count,
            batch_size, gamma, target_sync_every,
            eps_start, eps_end, eps_decay, shaping, rng,
        )
        history.append(ep_return)
        if (ep + 1) % log_every == 0:
            recent = sum(history[-50:]) / min(len(history), 50)
            print(
                f"  episode {ep + 1:4d}  return={ep_return:.0f}  "
                f"recent_50={recent:.1f}  steps={step_count}"
            )
    return q, history


def evaluate(q: QNetwork, n_episodes: int = 30) -> float:
    total = 0.0
    for _ in range(n_episodes):
        state = MCState()
        ep_return = 0.0
        for _ in range(MAX_STEPS):
            obs = observe(state)
            with torch.no_grad():
                action = int(torch.argmax(q(obs)).item())
            reward, state, done = mc_step(state, action)
            ep_return += reward
            if done:
                break
        total += ep_return
    return total / n_episodes


if __name__ == "__main__":
    print("=== DQN on MountainCar ===")
    q, history = train_dqn()
    avg = evaluate(q)
    print(f"\nEval (30 episodes, greedy): avg_return={avg:.1f}")
    print(f"RESULT\tavg_return={avg:.1f}\tepisodes={len(history)}\tseed=42")
