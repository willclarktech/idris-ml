"""SAC (Haarnoja et al. 2018) on MountainCarContinuous-v0.

Continuous-action sibling to the discrete `mountain_car.py`. SAC pairs
naturally with the dense |v|-magnitude reward shaping that worked for
DQN on discrete MountainCar — without it, random Gaussian exploration
almost never reaches the goal in 999 steps.

Aligned with `Example.MountainCarCont` (Idris): same architecture
(actor 2→64→64→1, twin Q 3→64→64→1, fixed alpha=0.2, polyak τ=0.005),
same shaping (`r_shaped = r_raw + 10·|v_next|`), same eval protocol
(20 greedy episodes, raw return).
"""

from __future__ import annotations

import copy
import math
import random
import time
from collections import deque
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_ref.training.runner import format_elapsed, mem_suffix

# ---------------------------------------------------------------------------
# MountainCarContinuous environment (Gymnasium-compatible constants)
# ---------------------------------------------------------------------------

MIN_POSITION = -1.2
MAX_POSITION = 0.6
MAX_SPEED = 0.07
GOAL_POSITION = 0.45
GOAL_VELOCITY = 0.0
POWER = 0.0015
MIN_ACTION = -1.0
MAX_ACTION = 1.0
MAX_STEPS = 999  # Gymnasium's default time limit


@dataclass
class MCCState:
    pos: float = -0.5
    vel: float = 0.0


def mcc_step(state: MCCState, action: float) -> tuple[float, MCCState, bool]:
    """One physics step. Returns (reward, next_state, terminated).

    `terminated` is True only when the goal is reached — not when the
    episode is truncated by step count.
    """
    clipped = max(MIN_ACTION, min(MAX_ACTION, action))
    force = clipped * POWER
    vel = state.vel + force - 0.0025 * math.cos(3.0 * state.pos)
    vel = max(-MAX_SPEED, min(MAX_SPEED, vel))
    pos = state.pos + vel
    pos = max(MIN_POSITION, min(MAX_POSITION, pos))
    if pos == MIN_POSITION and vel < 0.0:
        vel = 0.0
    next_state = MCCState(pos=pos, vel=vel)
    terminated = pos >= GOAL_POSITION and vel >= GOAL_VELOCITY
    reward = (100.0 if terminated else 0.0) - 0.1 * clipped * clipped
    return reward, next_state, terminated


def observe(state: MCCState) -> Tensor:
    return torch.tensor([state.pos, state.vel], dtype=torch.float64)


# ---------------------------------------------------------------------------
# Actor: tanh-squashed Gaussian policy
# ---------------------------------------------------------------------------


class Actor(nn.Module):
    def __init__(self, obs_dim: int = 2, hidden: int = 64) -> None:
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
        mean, log_std = self(x)
        std = torch.exp(log_std)
        if rng is None:
            eps = torch.randn_like(mean)
        else:
            eps = torch.tensor(rng.gauss(0.0, 1.0), dtype=torch.float64)
        u = mean + std * eps
        a_squashed = torch.tanh(u)
        action = a_squashed * MAX_ACTION
        log_prob_u = -0.5 * ((u - mean) / std) ** 2 - log_std - 0.5 * math.log(2 * math.pi)
        log_prob = log_prob_u - torch.log(1.0 - a_squashed**2 + 1e-6) - math.log(MAX_ACTION)
        return action, log_prob


# ---------------------------------------------------------------------------
# Q-networks
# ---------------------------------------------------------------------------


class QNet(nn.Module):
    def __init__(self, obs_dim: int = 2, act_dim: int = 1, hidden: int = 64) -> None:
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

    q1_loss = F.mse_loss(q1(obs, actions), target)
    q2_loss = F.mse_loss(q2(obs, actions), target)
    q1_opt.zero_grad()
    q1_loss.backward()
    q1_opt.step()
    q2_opt.zero_grad()
    q2_loss.backward()
    q2_opt.step()

    sampled_action, logp = actor.sample(obs)
    q_min = torch.min(q1(obs, sampled_action), q2(obs, sampled_action))
    actor_loss = (alpha * logp - q_min).mean()
    actor_opt.zero_grad()
    actor_loss.backward()
    actor_opt.step()

    return float(actor_loss.item())


def polyak_update(target: nn.Module, online: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for t, o in zip(target.parameters(), online.parameters(), strict=True):
            t.mul_(1.0 - tau).add_(o, alpha=tau)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_sac(
    total_steps: int = 30000, buffer_capacity: int = 100000, batch_size: int = 64,
    lr: float = 3e-4, gamma: float = 0.99, alpha: float = 0.2,
    warmup_steps: int = 1000, tau: float = 0.005, shaping: float = 10.0,
    seed: int = 42, log_every: int = 2000,
) -> tuple[Actor, list[float]]:
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

    history: list[float] = []
    state = MCCState()
    ep_return = 0.0
    ep_len = 0
    t_start = time.monotonic()
    for step in range(total_steps):
        obs = observe(state)
        if step < warmup_steps:
            action = rng.uniform(-MAX_ACTION, MAX_ACTION)
        else:
            with torch.no_grad():
                a_t, _ = actor.sample(obs)
                action = float(a_t.item())
        raw_reward, next_state, terminated = mcc_step(state, action)
        ep_return += raw_reward
        ep_len += 1
        truncated = ep_len >= MAX_STEPS
        is_done = terminated or truncated
        # Buffer's done flag reflects TRUE termination only (so the
        # Q-target bootstrap continues at truncation boundaries).
        buffer_done = terminated
        shaped = raw_reward + shaping * abs(next_state.vel)
        buffer.push(obs.tolist(), action, shaped, observe(next_state).tolist(), buffer_done)
        state = next_state
        if is_done:
            history.append(ep_return)
            ep_return = 0.0
            ep_len = 0
            state = MCCState()
        if len(buffer) >= max(batch_size, warmup_steps):
            sac_update(
                actor, q1, q2, q1_target, q2_target,
                actor_opt, q1_opt, q2_opt, buffer, batch_size, gamma, alpha, rng,
            )
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
    total = 0.0
    for _ in range(n_episodes):
        state = MCCState()
        ep_return = 0.0
        for _ in range(MAX_STEPS):
            obs = observe(state)
            with torch.no_grad():
                mean, _ = actor(obs)
            action = float(torch.tanh(mean).item()) * MAX_ACTION
            reward, state, terminated = mcc_step(state, action)
            ep_return += reward
            if terminated:
                break
        total += ep_return
    return total / n_episodes


if __name__ == "__main__":
    print("=== SAC on MountainCarContinuous ===")
    actor, history = train_sac()
    avg = evaluate(actor)
    print(f"\nEval (20 episodes, greedy): avg_return={avg:.1f}")
    print(f"RESULT\tavg_return={avg:.1f}\tsteps={30000}\tseed=42")
