"""DQN on MountainCar-v0 with reward shaping.

Uses canonical `gym.make("MountainCar-v0")` for env physics. Sparse
reward (-1/step until goal) is augmented with velocity-magnitude
shaping `r' = r + shaping * |v'|` to provide a dense intermediate
signal — the agent learns to build kinetic energy as the proven
precursor to reaching the goal.

Both Idris (`Gym.ClassicControl.MountainCar.reset`) and the PyTorch
reference randomize the initial state per Gymnasium U(-0.6, -0.4) ×
{0.0} — seeded once at trainer start, advanced per episode.

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

# Parallel envs collecting transitions in lockstep. Matches Idris-side
# `Example.MountainCar.NumEnvs`.
NUM_ENVS = 4


def make_mountaincar_env(seed: int) -> gym.Env:
    """Create a MountainCar-v0 env seeded once at construction. Per-episode
    resets advance the env's PRNG and randomize pos ~ U(-0.6, -0.4) per
    Gymnasium, matching idris-gym's randomized `Env.reset`."""
    env = gym.make("MountainCar-v0")
    env.reset(seed=seed)
    return env


def reset_to_center(env: gym.Env) -> np.ndarray:
    """Return obs of the env's current (just-reset) state as float64.

    Previously pinned to (-0.5, 0.0); idris-gym now randomizes per
    Gymnasium and the PyTorch side follows.
    """
    return np.asarray(env.unwrapped.state, dtype=np.float64)  # pyright: ignore[reportAttributeAccessIssue]


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
        return rng.randrange(3)
    with torch.no_grad():
        return int(torch.argmax(q(obs)).item())


def make_mountaincar_vec_env(seed: int, num_envs: int) -> gym.vector.SyncVectorEnv:
    """N MountainCar-v0 envs in a SyncVectorEnv, seeded once at construction
    and randomized per Gymnasium on each reset."""

    def _make(idx: int):
        def _f():
            return make_mountaincar_env(seed + idx)

        return _f

    vec = gym.vector.SyncVectorEnv([_make(i) for i in range(num_envs)])
    vec.reset()
    return vec


def eps_greedy_batched(
    q: QNetwork, obs_batch: Tensor, epsilon: float, rng: random.Random
) -> np.ndarray:
    """Batched epsilon-greedy across NUM_ENVS envs."""
    n = obs_batch.shape[0]
    with torch.no_grad():
        q_vals = q(obs_batch)  # [N, 3]
        greedy = torch.argmax(q_vals, dim=-1).cpu().numpy()
    actions = np.zeros(n, dtype=np.int64)
    for i in range(n):
        if rng.random() < epsilon:
            actions[i] = rng.randrange(3)
        else:
            actions[i] = int(greedy[i])
    return actions


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
        buffer.push(obs_np.tolist(), action, shaped_reward, next_obs_np.tolist(), done)
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
    eps_start: float,
    eps_end: float,
    eps_decay: int,
    shaping: float,
    rng: random.Random,
) -> tuple[int, float, np.ndarray]:
    """Batched DQN episode: NUM_ENVS parallel envs in lockstep. Each outer
    step does one batched action-selection forward → N transitions pushed
    → 1 gradient update. env-0 is primary; epoch terminates when env-0
    done. Returns (new_step_count, env-0 raw episode return, new obs_np)."""
    ep_return = 0.0
    for _ in range(MAX_STEPS):
        obs_t = obs_tensor(obs_np)
        epsilon = linear_epsilon(step_count, eps_start, eps_end, eps_decay)
        actions_np = eps_greedy_batched(q, obs_t, epsilon, rng)
        next_obs_np, raw_rewards, terms_np, truncs_np, _ = vec_env.step(actions_np)
        next_obs_np = next_obs_np.astype(np.float64)
        dones_np = np.logical_or(terms_np, truncs_np)
        ep_return += float(raw_rewards[0])
        for i in range(NUM_ENVS):
            shaped_r = float(raw_rewards[i]) + shaping * abs(float(next_obs_np[i, 1]))
            buffer.push(
                obs_np[i].tolist(),
                int(actions_np[i]),
                shaped_r,
                next_obs_np[i].tolist(),
                bool(dones_np[i]),
            )
        # Gymnasium SyncVectorEnv auto-resets terminated sub-envs; the
        # randomized initial obs is already in next_obs_np[i].
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
    """Batched DQN. NUM_ENVS envs collect transitions in lockstep via
    `gym.vector.SyncVectorEnv`. Each "episode" = env-0's primary episode."""
    torch.manual_seed(seed)
    rng = random.Random(seed)
    q = QNetwork()
    target = copy.deepcopy(q)
    optimizer = torch.optim.Adam(q.parameters(), lr=lr)
    buffer = ReplayBuffer(buffer_capacity)
    vec_env = make_mountaincar_vec_env(seed, NUM_ENVS)
    obs_np = np.tile(np.array([-0.5, 0.0], dtype=np.float64), (NUM_ENVS, 1))
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
            eps_start,
            eps_end,
            eps_decay,
            shaping,
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
