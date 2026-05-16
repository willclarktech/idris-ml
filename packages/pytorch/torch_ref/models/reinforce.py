"""REINFORCE (Williams 1992) on CartPole-v0.

Policy gradient with mean-return baseline. Self-contained CartPole physics
matching Gymnasium's exact constants (no gymnasium dependency needed).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from torch_ref.training.runner import format_elapsed, mem_suffix

# ---------------------------------------------------------------------------
# CartPole environment (Gymnasium-compatible constants, Euler integration)
# ---------------------------------------------------------------------------

GRAVITY = 9.8
MASSCART = 1.0
MASSPOLE = 0.1
TOTAL_MASS = MASSCART + MASSPOLE
LENGTH = 0.5  # half pole length
POLEMASS_LENGTH = MASSPOLE * LENGTH
FORCE_MAG = 10.0
TAU = 0.02  # integration timestep
THETA_THRESHOLD = 12.0 * 2.0 * math.pi / 360.0  # 12 degrees
X_THRESHOLD = 2.4
MAX_STEPS = 200


@dataclass
class CartPoleState:
    x: float = 0.0
    x_dot: float = 0.0
    theta: float = 0.0
    theta_dot: float = 0.0


def cartpole_step(state: CartPoleState, action: int) -> tuple[float, CartPoleState, bool]:
    """One step of CartPole physics. Returns (reward, next_state, done)."""
    force = FORCE_MAG if action == 1 else -FORCE_MAG
    cos_theta = math.cos(state.theta)
    sin_theta = math.sin(state.theta)

    temp = (force + POLEMASS_LENGTH * state.theta_dot**2 * sin_theta) / TOTAL_MASS
    theta_acc = (GRAVITY * sin_theta - cos_theta * temp) / (
        LENGTH * (4.0 / 3.0 - MASSPOLE * cos_theta**2 / TOTAL_MASS)
    )
    x_acc = temp - POLEMASS_LENGTH * theta_acc * cos_theta / TOTAL_MASS

    next_state = CartPoleState(
        x=state.x + TAU * state.x_dot,
        x_dot=state.x_dot + TAU * x_acc,
        theta=state.theta + TAU * state.theta_dot,
        theta_dot=state.theta_dot + TAU * theta_acc,
    )
    done = abs(next_state.x) > X_THRESHOLD or abs(next_state.theta) > THETA_THRESHOLD
    return 1.0, next_state, done


def observe(state: CartPoleState) -> Tensor:
    """State to observation tensor."""
    return torch.tensor([state.x, state.x_dot, state.theta, state.theta_dot], dtype=torch.float64)


# ---------------------------------------------------------------------------
# Policy network
# ---------------------------------------------------------------------------


class PolicyNetwork(nn.Module):
    def __init__(self, hidden: int = 128) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, hidden, dtype=torch.float64)
        self.fc2 = nn.Linear(hidden, 2, dtype=torch.float64)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(torch.tanh(self.fc1(x)))


# ---------------------------------------------------------------------------
# REINFORCE
# ---------------------------------------------------------------------------


def collect_episode(
    policy: PolicyNetwork, max_steps: int = MAX_STEPS
) -> tuple[list[Tensor], list[float]]:
    """Run one episode, return (log_probs, rewards)."""
    state = CartPoleState()
    log_probs: list[Tensor] = []
    rewards: list[float] = []

    for _ in range(max_steps):
        obs = observe(state)
        logits = policy(obs)
        log_p = torch.log_softmax(logits, dim=0)
        probs = torch.exp(log_p)
        action = int(torch.multinomial(probs, 1).item())

        log_probs.append(log_p[action])
        reward, state, done = cartpole_step(state, int(action))
        rewards.append(reward)
        if done:
            break

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
        log_probs, rewards = collect_episode(policy)
        returns = discounted_returns(rewards, gamma)
        ep_return = sum(rewards)
        episode_returns.append(ep_return)

        all_log_probs.extend(log_probs)
        all_advantages.extend(returns)

    # Baseline: mean episodic return
    baseline = sum(episode_returns) / len(episode_returns)
    adjusted = [g - baseline for g in all_advantages]

    # Policy gradient loss: -mean(log_prob * advantage)
    optimizer.zero_grad()
    loss = torch.tensor(0.0, dtype=torch.float64)
    for lp, adv in zip(all_log_probs, adjusted, strict=True):
        loss = loss - lp * adv
    loss = loss / len(all_log_probs)
    loss_val = float(loss.item())
    loss.backward()
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
    torch.manual_seed(seed)
    policy = PolicyNetwork()
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    history: list[float] = []
    t_start = time.monotonic()
    for epoch in range(epochs):
        avg_return, loss_val = reinforce_epoch(policy, optimizer, batch_size, gamma)
        history.append(avg_return)

        if (epoch + 1) % log_every == 0:
            recent = sum(history[-100:]) / min(len(history), 100)
            print(
                f"  {format_elapsed(t_start)} {epoch + 1}\tloss={loss_val:.6f}"
                f"{mem_suffix()}\treturn={avg_return:.1f}\trecent_100={recent:.1f}"
            )

        # Early stop if solved
        if len(history) >= 100:
            recent_avg = sum(history[-100:]) / 100
            if recent_avg >= 195.0:
                print(f"  Solved at epoch {epoch + 1} (recent_100={recent_avg:.1f})")
                break

    return policy, history


def evaluate(policy: PolicyNetwork, n_episodes: int = 100) -> float:
    """Evaluate policy greedily (argmax). Returns mean return."""
    total = 0.0
    for _ in range(n_episodes):
        state = CartPoleState()
        ep_return = 0.0
        for _ in range(MAX_STEPS):
            obs = observe(state)
            with torch.no_grad():
                logits = policy(obs)
            action = torch.argmax(logits).item()
            reward, state, done = cartpole_step(state, int(action))
            ep_return += reward
            if done:
                break
        total += ep_return
    return total / n_episodes


if __name__ == "__main__":
    print("=== REINFORCE on CartPole ===")
    policy, history = train_reinforce()
    avg = evaluate(policy)
    print(f"\nEval (100 episodes, greedy): avg_return={avg:.1f}")
    print(f"RESULT  avg_return={avg:.1f}  epochs={len(history)}  seed=42")
