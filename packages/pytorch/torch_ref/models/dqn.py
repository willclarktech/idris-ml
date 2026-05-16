"""DQN (Mnih et al. 2015) on CartPole-v0.

Deep Q-Network with experience replay and a target network. Self-contained
CartPole physics imported from `reinforce.py`. Convergence target:
greedy-evaluation return >= 150 on CartPole in 500 episodes.
"""

from __future__ import annotations

import copy
import random
import time
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_ref.models.reinforce import MAX_STEPS, CartPoleState, cartpole_step, observe
from torch_ref.training.runner import format_elapsed, mem_suffix

# ---------------------------------------------------------------------------
# Q-network
# ---------------------------------------------------------------------------


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int = 4, num_actions: int = 2, hidden: int = 64) -> None:
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
        return rng.randrange(2)
    with torch.no_grad():
        return int(torch.argmax(q(obs)).item())


def linear_epsilon(
    step: int, start: float = 1.0, end: float = 0.05, decay_steps: int = 10000
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
    rng: random.Random,
) -> tuple[int, float]:
    """Run one episode. Returns (new_step_count, episodic_return)."""
    state = CartPoleState()
    ep_return = 0.0
    for _ in range(MAX_STEPS):
        obs = observe(state)
        epsilon = linear_epsilon(step_count)
        action = eps_greedy_action(q, obs, epsilon, rng)
        reward, next_state, done = cartpole_step(state, action)
        ep_return += reward
        buffer.push(obs.tolist(), action, reward, observe(next_state).tolist(), done)
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
    episodes: int = 300,
    lr: float = 5e-4,
    gamma: float = 0.99,
    batch_size: int = 64,
    buffer_capacity: int = 10000,
    target_sync_every: int = 100,
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
    t_start = time.monotonic()
    for ep in range(episodes):
        step_count, ep_return = dqn_episode(
            q, target, optimizer, buffer, step_count, batch_size, gamma, target_sync_every, rng
        )
        history.append(ep_return)
        if (ep + 1) % log_every == 0:
            recent = sum(history[-50:]) / min(len(history), 50)
            print(
                f"  {format_elapsed(t_start)} {ep + 1}\tloss={-ep_return:.6f}"
                f"{mem_suffix()}\treturn={ep_return:.1f}\trecent_50={recent:.1f}"
            )
    return q, history


def evaluate(q: QNetwork, n_episodes: int = 50) -> float:
    total = 0.0
    for _ in range(n_episodes):
        state = CartPoleState()
        ep_return = 0.0
        for _ in range(MAX_STEPS):
            obs = observe(state)
            with torch.no_grad():
                action = int(torch.argmax(q(obs)).item())
            reward, state, done = cartpole_step(state, action)
            ep_return += reward
            if done:
                break
        total += ep_return
    return total / n_episodes


if __name__ == "__main__":
    print("=== DQN on CartPole ===")
    q, history = train_dqn()
    avg = evaluate(q)
    print(f"\nEval (50 episodes, greedy): avg_return={avg:.1f}")
    print(f"RESULT\tavg_return={avg:.1f}\tepisodes={len(history)}\tseed=42")
