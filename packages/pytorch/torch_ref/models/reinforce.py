"""REINFORCE (Williams 1992) on CartPole-v1.

Policy gradient with mean-return baseline. Uses canonical
`gym.make("CartPole-v1")` for env physics. Both Idris
(`Gym.ClassicControl.CartPole.reset`) and the PyTorch reference
randomize each of the 4 initial-state components per Gymnasium
U(-0.05, 0.05) — seeded once at trainer start, advanced per episode.

MAX_STEPS=200 matches the CartPole-v0 episode cap (idris-gym
`cartPoleMaxSteps`). CartPole-v1's wrapper has a 500-step cap but we
truncate at 200 ourselves for paired-side parity.
"""

from __future__ import annotations

import time
from typing import cast

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from torch_ref.init import init_linear_
from torch_ref.training.runner import (
    format_elapsed,
    get_device,
    get_dtype,
    mem_suffix,
    multinomial_safe,
)

MAX_STEPS = 200

# CartPole-v1: Box observations (np.ndarray) and Discrete actions (int).
# gymnasium's `gym.make` stub returns `Env[Unknown, Unknown]`, so call
# sites pin this alias for pyright strict.
CartPoleEnv = gym.Env[np.ndarray, int]


def make_cartpole_env(seed: int) -> CartPoleEnv:
    """Create a CartPole-v1 env seeded once at construction. Per-episode
    resets advance the env's PRNG and randomize the start state per
    Gymnasium's U(-0.05, 0.05)^4, matching idris-gym's randomized
    `Env.reset`."""
    env = cast("CartPoleEnv", gym.make("CartPole-v1"))  # pyright: ignore[reportUnknownMemberType]
    env.reset(seed=seed)
    return env


def reset_to_zero(env: CartPoleEnv) -> np.ndarray:
    """Return the obs of the env's current (just-reset) state as float64.

    Previously pinned env state to (0, 0, 0, 0) to match idris-gym's
    deterministic reset; idris-gym now randomizes per Gymnasium and the
    PyTorch side follows suit. Function name kept for call-site stability.
    """
    # `state` isn't on the Env stub (CartPole-specific attribute).
    return np.asarray(env.unwrapped.state, dtype=np.float64)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownArgumentType]


def obs_tensor(obs: np.ndarray) -> Tensor:
    return torch.tensor(obs, dtype=get_dtype(), device=get_device())


class PolicyNetwork(nn.Module):
    def __init__(self, hidden: int = 128) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, hidden, dtype=get_dtype())
        self.fc2 = nn.Linear(hidden, 2, dtype=get_dtype())
        init_linear_(self)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(torch.tanh(self.fc1(x)))


def collect_episode(
    env: CartPoleEnv, policy: PolicyNetwork, max_steps: int = MAX_STEPS
) -> tuple[list[Tensor], list[float]]:
    """Run one episode, return (log_probs, rewards)."""
    env.reset()
    obs_np = reset_to_zero(env)
    log_probs: list[Tensor] = []
    rewards: list[float] = []
    for _ in range(max_steps):
        obs = obs_tensor(obs_np)
        logits = policy(obs)
        log_p = torch.log_softmax(logits, dim=0)
        probs = torch.exp(log_p)
        action = int(multinomial_safe(probs, 1).item())
        log_probs.append(log_p[action])
        next_obs_np, reward, term, trunc, _ = env.step(action)
        rewards.append(float(reward))
        if term or trunc:
            break
        obs_np = next_obs_np.astype(np.float64)
    return log_probs, rewards


def discounted_returns(rewards: list[float], gamma: float = 0.99) -> list[float]:
    """Compute discounted returns G_t = sum_{k=0}^{T-t-1} gamma^k * r_{t+k}."""
    returns: list[float] = []
    g = 0.0
    for r in reversed(rewards):
        g = r + gamma * g
        returns.append(g)
    returns.reverse()
    return returns


def reinforce_epoch(
    env: CartPoleEnv,
    policy: PolicyNetwork,
    optimizer: torch.optim.Optimizer,
    batch_size: int = 10,
    gamma: float = 0.99,
) -> tuple[float, float]:
    """One REINFORCE update: collect batch of episodes, compute policy gradient.

    Returns (mean episode return, policy-gradient loss scalar).
    """
    all_log_probs: list[Tensor] = []
    all_advantages: list[float] = []
    episode_returns: list[float] = []
    for _ in range(batch_size):
        log_probs, rewards = collect_episode(env, policy)
        returns = discounted_returns(rewards, gamma)
        ep_return = sum(rewards)
        episode_returns.append(ep_return)
        all_log_probs.extend(log_probs)
        all_advantages.extend(returns)
    baseline = sum(episode_returns) / len(episode_returns)
    adjusted = [g - baseline for g in all_advantages]
    optimizer.zero_grad()
    loss = torch.tensor(0.0, dtype=get_dtype(), device=get_device())
    for lp, adv in zip(all_log_probs, adjusted, strict=True):
        loss = loss - lp * adv
    loss = loss / len(all_log_probs)
    loss_val = float(loss.item())
    # torch stub: Tensor.backward's params are unannotated.
    loss.backward()  # pyright: ignore[reportUnknownMemberType]
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    optimizer.step()
    return sum(episode_returns) / len(episode_returns), loss_val


def train_reinforce(
    epochs: int = 3000,
    batch_size: int = 10,
    lr: float = 0.001,
    gamma: float = 0.99,
    seed: int = 42,
    log_every: int = 100,
) -> tuple[PolicyNetwork, list[float]]:
    """Train REINFORCE on CartPole. Returns (policy, history of avg returns)."""
    # torch stub: manual_seed's seed param is unannotated.
    torch.manual_seed(seed)  # pyright: ignore[reportUnknownMemberType]
    policy = PolicyNetwork()
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    env = make_cartpole_env(seed)
    history: list[float] = []
    t_start = time.monotonic()
    for epoch in range(epochs):
        avg_return, loss_val = reinforce_epoch(env, policy, optimizer, batch_size, gamma)
        history.append(avg_return)
        if (epoch + 1) % log_every == 0:
            recent = sum(history[-100:]) / min(len(history), 100)
            print(
                f"  {format_elapsed(t_start)} {epoch + 1}\tloss={loss_val:.6f}"
                f"{mem_suffix()}\treturn={avg_return:.1f}\trecent_100={recent:.1f}"
            )
        if len(history) >= 100:
            recent_avg = sum(history[-100:]) / 100
            if recent_avg >= 195.0:
                print(f"  Solved at epoch {epoch + 1} (recent_100={recent_avg:.1f})")
                break
    return policy, history


def evaluate(policy: PolicyNetwork, n_episodes: int = 100) -> float:
    """Evaluate policy greedily (argmax). Returns mean return."""
    env = make_cartpole_env(seed=0)
    total = 0.0
    for _ in range(n_episodes):
        env.reset()
        obs_np = reset_to_zero(env)
        ep_return = 0.0
        for _ in range(MAX_STEPS):
            obs = obs_tensor(obs_np)
            with torch.no_grad():
                logits = policy(obs)
            action = int(torch.argmax(logits).item())
            next_obs_np, reward, term, trunc, _ = env.step(action)
            ep_return += float(reward)
            if term or trunc:
                break
            obs_np = next_obs_np.astype(np.float64)
        total += ep_return
    return total / n_episodes


if __name__ == "__main__":
    print("=== REINFORCE on CartPole ===")
    policy, history = train_reinforce()
    avg = evaluate(policy)
    print(f"\nEval (100 episodes, greedy): avg_return={avg:.1f}")
    print(f"RESULT  avg_return={avg:.1f}  epochs={len(history)}  seed=42")
