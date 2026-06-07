"""DQN (Mnih et al. 2015) on CartPole-v1.

Deep Q-Network with experience replay and a target network. Uses
`gym.vector.SyncVectorEnv` for NUM_ENVS parallel CartPole envs
collecting transitions in lockstep (matches Idris-side
`Gym.Vector.VecEnv NumEnvs CPState`). Each "epoch" = env-0's primary
episode; the other envs auto-reset and continue feeding the buffer.

Reset state pinned to (0, 0, 0, 0) to mirror idris-gym.

Convergence target: greedy-evaluation return >= 150 on CartPole in 500
episodes.
"""

from __future__ import annotations

import copy
import random
import time
from collections import deque
from typing import Any, cast

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_ref.models.reinforce import (
    MAX_STEPS,
    make_cartpole_env,
    obs_tensor,
    reset_to_zero,
)
from torch_ref.training.runner import format_elapsed, get_device, get_dtype, mem_suffix

# Parallel envs collecting transitions in lockstep. Matches Idris-side
# `Example.Dqn.NumEnvs`.
NUM_ENVS = 4

# CartPole-v1: Box obs (np.ndarray) and Discrete(2) actions (int).
type CartPoleEnv = gym.Env[np.ndarray, int]


def make_cartpole_vec_env(seed: int, num_envs: int) -> gym.vector.SyncVectorEnv:
    """N independent CartPole envs in a SyncVectorEnv, each reset to all-zero
    to mirror idris-gym's `Gym.ClassicControl.CartPole.reset`."""

    def _make(idx: int):
        def _f() -> CartPoleEnv:
            return make_cartpole_env(seed + idx)

        return _f

    vec = gym.vector.SyncVectorEnv([_make(i) for i in range(num_envs)])
    vec.reset()
    # SyncVectorEnv.envs is untyped upstream (bare `Env`).
    for sub in cast("list[CartPoleEnv]", vec.envs):  # pyright: ignore[reportUnknownMemberType]
        reset_to_zero(sub)
    return vec


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int = 4, num_actions: int = 2, hidden: int = 64) -> None:
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden, dtype=get_dtype())
        self.fc2 = nn.Linear(hidden, hidden, dtype=get_dtype())
        self.fc3 = nn.Linear(hidden, num_actions, dtype=get_dtype())

    def forward(self, x: Tensor) -> Tensor:
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buf: deque[tuple[list[float], int, float, list[float], bool]] = deque(maxlen=capacity)

    def push(
        self,
        obs: list[float],
        action: int,
        reward: float,
        next_obs: list[float],
        done: bool,
    ) -> None:
        self.buf.append((obs, action, reward, next_obs, done))

    def sample(self, n: int, rng: random.Random) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
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
        return rng.randrange(2)
    with torch.no_grad():
        return int(torch.argmax(q(obs)).item())


def eps_greedy_batched(
    q: QNetwork, obs_batch: Tensor, epsilon: float, rng: random.Random
) -> np.ndarray:
    """Batched epsilon-greedy across NUM_ENVS envs. One batched forward,
    then per-env eps-vs-greedy with independent random draws (matches the
    Idris-side one-randomRIO-per-env-per-step convention)."""
    n = obs_batch.shape[0]
    with torch.no_grad():
        q_vals = q(obs_batch)  # [N, 2]
        greedy = torch.argmax(q_vals, dim=-1).cpu().numpy()  # [N]
    actions = np.zeros(n, dtype=np.int64)
    for i in range(n):
        if rng.random() < epsilon:
            actions[i] = rng.randrange(2)
        else:
            actions[i] = int(greedy[i])
    return actions


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
    # torch stubs leave Tensor.backward's params untyped.
    loss.backward()  # pyright: ignore[reportUnknownMemberType]
    torch.nn.utils.clip_grad_norm_(q.parameters(), 10.0)
    optimizer.step()
    return float(loss.item())


def dqn_episode(
    env: CartPoleEnv,
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
    env.reset()
    obs_np = reset_to_zero(env)
    ep_return = 0.0
    for _ in range(MAX_STEPS):
        obs = obs_tensor(obs_np)
        epsilon = linear_epsilon(step_count)
        action = eps_greedy_action(q, obs, epsilon, rng)
        next_obs_np, reward, term, trunc, _ = env.step(action)
        next_obs_np = next_obs_np.astype(np.float64)
        done = bool(term or trunc)
        ep_return += float(reward)
        buffer.push(obs_np.tolist(), action, float(reward), next_obs_np.tolist(), done)
        obs_np = next_obs_np
        step_count += 1

        if len(buffer) >= batch_size:
            dqn_update(q, target, optimizer, buffer, batch_size, gamma, rng)

        if step_count % target_sync_every == 0:
            target.load_state_dict(q.state_dict())

        if done:
            break
    return step_count, ep_return


def dqn_episode_batched(
    vec_env: gym.vector.SyncVectorEnv,
    obs_np: np.ndarray,
    q: QNetwork,
    target: QNetwork,
    optimizer: torch.optim.Optimizer,
    buffer: ReplayBuffer,
    step_count: int,
    batch_size: int,
    gamma: float,
    target_sync_every: int,
    rng: random.Random,
) -> tuple[int, float, np.ndarray]:
    """Batched DQN episode. NUM_ENVS envs collect transitions in lockstep
    via one batched action-selection forward per outer step; NUM_ENVS
    transitions get pushed per outer step; one gradient update per outer
    step (replay-ratio shifts by N — same on both sides for alignment).
    env-0 is the primary; epoch terminates when env-0 hits done.

    Returns (new_step_count, env-0's episode return, new obs_np)."""
    ep_return = 0.0
    # SyncVectorEnv.envs is untyped upstream (bare `Env`).
    envs = cast("list[CartPoleEnv]", vec_env.envs)  # pyright: ignore[reportUnknownMemberType]
    for _ in range(MAX_STEPS):
        obs_t = obs_tensor(obs_np)  # [N, 4]
        epsilon = linear_epsilon(step_count)
        actions_np = eps_greedy_batched(q, obs_t, epsilon, rng)
        # SyncVectorEnv is unparameterized upstream; narrow its step() products.
        next_obs_np, rewards_np, terms_np, truncs_np, _ = cast(
            "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]",
            vec_env.step(actions_np),
        )
        next_obs_np = next_obs_np.astype(np.float64)
        dones_np = np.logical_or(terms_np, truncs_np)
        for i in range(NUM_ENVS):
            buffer.push(
                obs_np[i].tolist(),
                int(actions_np[i]),
                float(rewards_np[i]),
                next_obs_np[i].tolist(),
                bool(dones_np[i]),
            )
        ep_return += float(rewards_np[0])
        # Auto-reset terminated envs back to zero state.
        for i in range(NUM_ENVS):
            if dones_np[i]:
                reset_to_zero(envs[i])
                next_obs_np[i] = 0.0
        obs_np = next_obs_np
        step_count += 1

        if len(buffer) >= batch_size:
            dqn_update(q, target, optimizer, buffer, batch_size, gamma, rng)

        if step_count % target_sync_every == 0:
            target.load_state_dict(q.state_dict())

        if dones_np[0]:
            break
    return step_count, ep_return, obs_np


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
    """Batched DQN. NUM_ENVS envs collect transitions in lockstep via
    `gym.vector.SyncVectorEnv`. Each "episode" = env-0's primary episode."""
    torch.manual_seed(seed)  # pyright: ignore[reportUnknownMemberType] - untyped `seed` param in torch stubs
    rng = random.Random(seed)
    q = QNetwork()
    target = copy.deepcopy(q)
    optimizer = torch.optim.Adam(q.parameters(), lr=lr)
    buffer = ReplayBuffer(buffer_capacity)
    vec_env = make_cartpole_vec_env(seed, NUM_ENVS)
    obs_np = np.zeros((NUM_ENVS, 4), dtype=np.float64)
    history: list[float] = []
    step_count = 0
    t_start = time.monotonic()
    for ep in range(episodes):
        step_count, ep_return, obs_np = dqn_episode_batched(
            vec_env,
            obs_np,
            q,
            target,
            optimizer,
            buffer,
            step_count,
            batch_size,
            gamma,
            target_sync_every,
            rng,
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
    env = make_cartpole_env(seed=0)
    total = 0.0
    for _ in range(n_episodes):
        env.reset()
        obs_np = reset_to_zero(env)
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
    print("=== DQN on CartPole ===")
    q, history = train_dqn()
    avg = evaluate(q)
    print(f"\nEval (50 episodes, greedy): avg_return={avg:.1f}")
    print(f"RESULT\tavg_return={avg:.1f}\tepisodes={len(history)}\tseed=42")
