"""SAC (Haarnoja et al. 2018) on MountainCarContinuous-v0.

Continuous-action sibling to the discrete `mountain_car.py`. SAC pairs
naturally with the dense |v|-magnitude reward shaping that worked for
DQN on discrete MountainCar — without it, random Gaussian exploration
almost never reaches the goal in 999 steps.

Uses canonical `gym.make("MountainCarContinuous-v0")` for env physics.
Both Idris (`Gym.ClassicControl.MountainCarContinuous.reset`) and the
PyTorch reference randomize the initial state per Gymnasium
U(-0.6, -0.4) × {0.0} — seeded once at trainer start, advanced per
episode.

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

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_ref.training.runner import format_elapsed, get_device, get_dtype, mem_suffix

MAX_ACTION = 1.0
MAX_STEPS = 999  # Gymnasium's MountainCarContinuous-v0 default TimeLimit

# Parallel envs collecting transitions in lockstep. Matches Idris-side
# `Example.MountainCarCont.NumEnvs`.
NUM_ENVS = 4


def make_mountaincarcont_env(seed: int) -> gym.Env:
    env = gym.make("MountainCarContinuous-v0")
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


def make_mountaincarcont_vec_env(seed: int, num_envs: int) -> gym.vector.SyncVectorEnv:
    """N MountainCarContinuous-v0 envs in a SyncVectorEnv, seeded once at
    construction and randomized per Gymnasium on each reset (mirrors
    Idris-side `Gym.Vector.resetAll`)."""

    def _make(idx: int):
        def _f():
            return make_mountaincarcont_env(seed + idx)

        return _f

    vec = gym.vector.SyncVectorEnv([_make(i) for i in range(num_envs)])
    vec.reset()
    return vec


class Actor(nn.Module):
    def __init__(self, obs_dim: int = 2, hidden: int = 64) -> None:
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden, dtype=get_dtype())
        self.fc2 = nn.Linear(hidden, hidden, dtype=get_dtype())
        self.mean_head = nn.Linear(hidden, 1, dtype=get_dtype())
        self.log_std = nn.Parameter(torch.zeros(1, dtype=get_dtype()))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = F.relu(self.fc2(F.relu(self.fc1(x))))
        mean = self.mean_head(h).squeeze(-1)
        log_std = torch.clamp(self.log_std.squeeze(0) + torch.zeros_like(mean), min=-5.0, max=2.0)
        return mean, log_std

    def sample(self, x: Tensor, rng: random.Random | None = None) -> tuple[Tensor, Tensor]:
        mean, log_std = self(x)
        std = torch.exp(log_std)
        if rng is None:
            eps = torch.randn_like(mean)
        else:
            eps = torch.tensor(rng.gauss(0.0, 1.0), dtype=get_dtype(), device=get_device())
        u = mean + std * eps
        a_squashed = torch.tanh(u)
        action = a_squashed * MAX_ACTION
        log_prob_u = -0.5 * ((u - mean) / std) ** 2 - log_std - 0.5 * math.log(2 * math.pi)
        log_prob = log_prob_u - torch.log(1.0 - a_squashed**2 + 1e-6) - math.log(MAX_ACTION)
        return action, log_prob


class QNet(nn.Module):
    def __init__(self, obs_dim: int = 2, act_dim: int = 1, hidden: int = 64) -> None:
        super().__init__()
        self.fc1 = nn.Linear(obs_dim + act_dim, hidden, dtype=get_dtype())
        self.fc2 = nn.Linear(hidden, hidden, dtype=get_dtype())
        self.head = nn.Linear(hidden, 1, dtype=get_dtype())

    def forward(self, obs: Tensor, action: Tensor) -> Tensor:
        a = action.unsqueeze(-1) if action.dim() == obs.dim() - 1 else action
        x = torch.cat([obs, a], dim=-1)
        h = F.relu(self.fc2(F.relu(self.fc1(x))))
        return self.head(h).squeeze(-1)


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buf: deque[tuple[list[float], float, float, list[float], bool]] = deque(
            maxlen=capacity
        )

    def push(self, obs: list[float], a: float, r: float, next_obs: list[float], done: bool) -> None:
        self.buf.append((obs, a, r, next_obs, done))

    def sample(self, n: int, rng: random.Random) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        batch = rng.sample(self.buf, n)
        device, dtype = get_device(), get_dtype()
        obs = torch.tensor([b[0] for b in batch], dtype=dtype, device=device)
        actions = torch.tensor([b[1] for b in batch], dtype=dtype, device=device)
        rewards = torch.tensor([b[2] for b in batch], dtype=dtype, device=device)
        next_obs = torch.tensor([b[3] for b in batch], dtype=dtype, device=device)
        dones = torch.tensor([float(b[4]) for b in batch], dtype=dtype, device=device)
        return obs, actions, rewards, next_obs, dones

    def __len__(self) -> int:
        return len(self.buf)


def sac_update(
    actor: Actor,
    q1: QNet,
    q2: QNet,
    q1_target: QNet,
    q2_target: QNet,
    actor_opt: torch.optim.Optimizer,
    q1_opt: torch.optim.Optimizer,
    q2_opt: torch.optim.Optimizer,
    buffer: ReplayBuffer,
    batch_size: int,
    gamma: float,
    alpha: float,
    rng: random.Random,
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


def train_sac(
    total_steps: int = 30000,
    buffer_capacity: int = 100000,
    batch_size: int = 64,
    lr: float = 3e-4,
    gamma: float = 0.99,
    alpha: float = 0.2,
    warmup_steps: int = 1000,
    tau: float = 0.005,
    shaping: float = 10.0,
    seed: int = 42,
    log_every: int = 2000,
) -> tuple[Actor, list[float]]:
    """Batched SAC. NUM_ENVS parallel envs in lockstep — each outer step
    pushes NUM_ENVS transitions to the buffer and does one gradient
    update. Total env-steps per `total_steps` outer = total_steps * N."""
    torch.manual_seed(seed)
    rng = random.Random(seed)
    actor = Actor().to(get_device())
    q1 = QNet().to(get_device())
    q2 = QNet().to(get_device())
    q1_target = copy.deepcopy(q1)
    q2_target = copy.deepcopy(q2)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=lr)
    q1_opt = torch.optim.Adam(q1.parameters(), lr=lr)
    q2_opt = torch.optim.Adam(q2.parameters(), lr=lr)
    buffer = ReplayBuffer(buffer_capacity)

    vec_env = make_mountaincarcont_vec_env(seed, NUM_ENVS)
    obs_np = np.tile(np.array([-0.5, 0.0], dtype=np.float64), (NUM_ENVS, 1))
    ep_returns_running = np.zeros(NUM_ENVS, dtype=np.float64)
    history: list[float] = []
    t_start = time.monotonic()
    for step in range(total_steps):
        if step < warmup_steps:
            actions_np = np.array(
                [rng.uniform(-MAX_ACTION, MAX_ACTION) for _ in range(NUM_ENVS)],
                dtype=np.float64,
            )
        else:
            obs_t = obs_tensor(obs_np)
            with torch.no_grad():
                a_t, _ = actor.sample(obs_t)
                actions_np = a_t.cpu().numpy().astype(np.float64)
        next_obs_np, raw_rewards, terms_np, truncs_np, _ = vec_env.step(
            actions_np.astype(np.float32).reshape(NUM_ENVS, 1)
        )
        next_obs_np = next_obs_np.astype(np.float64)
        is_dones = np.logical_or(terms_np, truncs_np)
        ep_returns_running += raw_rewards.astype(np.float64)
        # Per-env push with shaping; buffer done flag reflects TRUE
        # termination only (Q-target bootstrap continues at truncation).
        for i in range(NUM_ENVS):
            shaped_r = float(raw_rewards[i]) + shaping * abs(float(next_obs_np[i, 1]))
            buffer.push(
                obs_np[i].tolist(),
                float(actions_np[i]),
                shaped_r,
                next_obs_np[i].tolist(),
                bool(terms_np[i]),
            )
        # Episode completion + auto-reset. Gymnasium's SyncVectorEnv
        # already auto-resets terminated sub-envs and returns the
        # randomized initial obs in next_obs_np[i].
        for i in range(NUM_ENVS):
            if is_dones[i]:
                history.append(float(ep_returns_running[i]))
                ep_returns_running[i] = 0.0
        obs_np = next_obs_np
        if len(buffer) >= max(batch_size, warmup_steps):
            sac_update(
                actor,
                q1,
                q2,
                q1_target,
                q2_target,
                actor_opt,
                q1_opt,
                q2_opt,
                buffer,
                batch_size,
                gamma,
                alpha,
                rng,
            )
            polyak_update(q1_target, q1, tau)
            polyak_update(q2_target, q2, tau)
        if (step + 1) % log_every == 0:
            recent = history[-20:] or [0.0]
            last_ep = history[-1] if history else 0.0
            print(
                f"  {format_elapsed(t_start)} {step + 1}\tloss={-last_ep:.6f}"
                f"{mem_suffix()}\treturn={last_ep:.1f}"
                f"\trecent_20={sum(recent) / len(recent):.1f}"
            )
    return actor, history


def evaluate(actor: Actor, n_episodes: int = 20) -> float:
    env = make_mountaincarcont_env(seed=0)
    total = 0.0
    for _ in range(n_episodes):
        env.reset()
        obs_np = reset_to_center(env)
        ep_return = 0.0
        for _ in range(MAX_STEPS):
            obs = obs_tensor(obs_np)
            with torch.no_grad():
                mean, _ = actor(obs)
            action = float(torch.tanh(mean).item()) * MAX_ACTION
            next_obs_np, reward, terminated, truncated, _ = env.step(
                np.array([action], dtype=np.float32)
            )
            ep_return += float(reward)
            if terminated or truncated:
                break
            obs_np = next_obs_np.astype(np.float64)
        total += ep_return
    return total / n_episodes


if __name__ == "__main__":
    print("=== SAC on MountainCarContinuous ===")
    actor, history = train_sac()
    avg = evaluate(actor)
    print(f"\nEval (20 episodes, greedy): avg_return={avg:.1f}")
    print(f"RESULT\tavg_return={avg:.1f}\tsteps={30000}\tseed=42")
