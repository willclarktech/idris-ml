"""DQN on MountainCar-v0 with reward shaping.

Uses canonical `gym.make("MountainCar-v0")` for env physics. Sparse
reward (-1/step until goal) is augmented with velocity-magnitude
shaping `r' = r + shaping * |v'|` to provide a dense intermediate
signal — the agent learns to build kinetic energy as the proven
precursor to reaching the goal.

Reset state pinned to (-0.5, 0.0) with float64 to mirror idris-gym's
`Gym.ClassicControl.MountainCar.reset = MkMC (-0.5) 0.0` (canonical
Pendulum randomizes pos ~ U(-0.6, -0.4); both Idris and torch_ref pin
to deterministic center).

Aligned with `Example.MountainCar` (Idris): same architecture (2 -> 64 ->
64 -> 3), same defaults (lr=1e-3, gamma=0.99, batch=64, buffer=50K,
target_sync=200, eps 1.0->0.05 over 50K steps, shaping=10.0), same eval
protocol (30 greedy episodes).
"""

from __future__ import annotations

import copy
import random
import time
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_ref.training.runner import format_elapsed, get_device, get_dtype, mem_suffix

MAX_STEPS = 200  # gymnasium MountainCar-v0 default TimeLimit


def make_mountaincar_env(seed: int) -> gym.Env:
    """Create a seeded MountainCar-v0 env. Use `reset_to_center` to pin
    initial state after each `env.reset()`."""
    env = gym.make("MountainCar-v0")
    env.reset(seed=seed)
    return env


def reset_to_center(env: gym.Env) -> np.ndarray:
    """Pin env state to (-0.5, 0.0) and return obs."""
    env.unwrapped.state = np.array([-0.5, 0.0], dtype=np.float64)  # pyright: ignore[reportAttributeAccessIssue]
    return np.array([-0.5, 0.0], dtype=np.float64)


def obs_tensor(obs: np.ndarray) -> Tensor:
    return torch.tensor(obs, dtype=get_dtype(), device=get_device())


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int = 2, num_actions: int = 3, hidden: int = 64) -> None:
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden, dtype=get_dtype())
        self.fc2 = nn.Linear(hidden, hidden, dtype=get_dtype())
        self.fc3 = nn.Linear(hidden, num_actions, dtype=get_dtype())

    def forward(self, x: Tensor) -> Tensor:
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))


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
        device, dtype = get_device(), get_dtype()
        obs = torch.tensor([b[0] for b in batch], dtype=dtype, device=device)
        actions = torch.tensor([b[1] for b in batch], dtype=torch.long, device=device)
        rewards = torch.tensor([b[2] for b in batch], dtype=dtype, device=device)
        next_obs = torch.tensor([b[3] for b in batch], dtype=dtype, device=device)
        dones = torch.tensor([float(b[4]) for b in batch], dtype=dtype, device=device)
        return obs, actions, rewards, next_obs, dones

    def __len__(self) -> int:
        return len(self.buf)


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
    env: gym.Env,
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
    """One episode. Returns (new_step_count, raw_episodic_return)."""
    env.reset()
    obs_np = reset_to_center(env)
    ep_return = 0.0
    for _ in range(MAX_STEPS):
        obs = obs_tensor(obs_np)
        epsilon = linear_epsilon(step_count, eps_start, eps_end, eps_decay)
        action = eps_greedy_action(q, obs, epsilon, rng)
        next_obs_np, raw_reward, term, trunc, _ = env.step(action)
        next_obs_np = next_obs_np.astype(np.float64)
        done = bool(term or trunc)
        ep_return += float(raw_reward)
        shaped_reward = float(raw_reward) + shaping * abs(float(next_obs_np[1]))
        buffer.push(
            obs_np.tolist(), action, shaped_reward, next_obs_np.tolist(), done
        )
        obs_np = next_obs_np
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
    env = make_mountaincar_env(seed)
    history: list[float] = []
    step_count = 0
    t_start = time.monotonic()
    for ep in range(episodes):
        step_count, ep_return = dqn_episode(
            env, q, target, optimizer, buffer, step_count,
            batch_size, gamma, target_sync_every,
            eps_start, eps_end, eps_decay, shaping, rng,
        )
        history.append(ep_return)
        if (ep + 1) % log_every == 0:
            recent = sum(history[-50:]) / min(len(history), 50)
            print(
                f"  {format_elapsed(t_start)} {ep + 1}\tloss={-ep_return:.6f}"
                f"{mem_suffix()}\treturn={ep_return:.1f}\trecent_50={recent:.1f}"
            )
    return q, history


def evaluate(q: QNetwork, n_episodes: int = 30) -> float:
    env = make_mountaincar_env(seed=0)
    total = 0.0
    for _ in range(n_episodes):
        env.reset()
        obs_np = reset_to_center(env)
        ep_return = 0.0
        for _ in range(MAX_STEPS):
            obs = obs_tensor(obs_np)
            with torch.no_grad():
                action = int(torch.argmax(q(obs)).item())
            next_obs_np, reward, term, trunc, _ = env.step(action)
            ep_return += float(reward)
            if term or trunc:
                break
            obs_np = next_obs_np.astype(np.float64)
        total += ep_return
    return total / n_episodes


if __name__ == "__main__":
    print("=== DQN on MountainCar ===")
    q, history = train_dqn()
    avg = evaluate(q)
    print(f"\nEval (30 episodes, greedy): avg_return={avg:.1f}")
    print(f"RESULT\tavg_return={avg:.1f}\tepisodes={len(history)}\tseed=42")
