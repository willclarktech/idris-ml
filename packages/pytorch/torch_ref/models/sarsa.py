"""SARSA (Rummery & Niranjan 1994) on CliffWalking-v1.

On-policy TD(0) control. Same env as Q-learning; the only difference is the
TD target uses the next action actually selected under the current policy
(epsilon-greedy) rather than max over next actions. Classic Sutton & Barto
result: SARSA converges to a safer path further from the cliff than
Q-learning, which walks right along the edge.
"""

from __future__ import annotations

import random
import time
from typing import cast

import gymnasium as gym
import numpy as np
import numpy.typing as npt

from torch_ref.models.q_learning import MAX_STEPS, NUM_ACTIONS, NUM_STATES, eps_greedy
from torch_ref.training.runner import format_elapsed, mem_suffix


def sarsa_episode(
    env: gym.Env[int, int],
    q: npt.NDArray[np.float64],
    alpha: float,
    gamma: float,
    epsilon: float,
    rng: random.Random,
    max_steps: int = MAX_STEPS,
) -> float:
    """Run one episode with SARSA updates. Returns episodic return."""
    obs, _ = env.reset()
    s = int(obs)
    action = eps_greedy(q[s], epsilon, rng)
    total_reward = 0.0
    for _ in range(max_steps):
        next_obs, reward, term, trunc, _ = env.step(action)
        s_next = int(next_obs)
        total_reward += float(reward)
        done = bool(term or trunc)
        if done:
            q[s, action] += alpha * (float(reward) - q[s, action])
            break
        next_action = eps_greedy(q[s_next], epsilon, rng)
        target = float(reward) + gamma * q[s_next, next_action]
        q[s, action] += alpha * (target - q[s, action])
        s = s_next
        action = next_action
    return total_reward


def train_sarsa(
    epochs: int = 1000,
    alpha: float = 0.5,
    gamma: float = 1.0,
    epsilon: float = 0.1,
    seed: int = 42,
    log_every: int = 100,
) -> tuple[npt.NDArray[np.float64], list[float]]:
    """Train SARSA on CliffWalking. Returns (Q-table, history)."""
    rng = random.Random(seed)
    # gym.make returns an unparameterized Env; CliffWalking obs/actions are ints
    env = cast(
        "gym.Env[int, int]",
        gym.make("CliffWalking-v1"),  # pyright: ignore[reportUnknownMemberType]
    )
    env.reset(seed=seed)
    q = np.zeros((NUM_STATES, NUM_ACTIONS), dtype=np.float64)
    history: list[float] = []
    t_start = time.monotonic()
    for epoch in range(epochs):
        ret = sarsa_episode(env, q, alpha, gamma, epsilon, rng)
        history.append(ret)
        if (epoch + 1) % log_every == 0:
            recent = sum(history[-100:]) / min(len(history), 100)
            print(
                f"  {format_elapsed(t_start)} {epoch + 1}\tloss={-ret:.6f}"
                f"{mem_suffix()}\treturn={ret:.1f}\trecent_100={recent:.1f}"
            )
    return q, history


def evaluate(q: npt.NDArray[np.float64], n_episodes: int = 100) -> float:
    """Greedy evaluation. Returns mean return."""
    # gym.make returns an unparameterized Env; CliffWalking obs/actions are ints
    env = cast(
        "gym.Env[int, int]",
        gym.make("CliffWalking-v1"),  # pyright: ignore[reportUnknownMemberType]
    )
    env.reset(seed=0)
    total = 0.0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        s = int(obs)
        ep_return = 0.0
        for _ in range(MAX_STEPS):
            action = int(np.argmax(q[s]))
            next_obs, reward, term, trunc, _ = env.step(action)
            s = int(next_obs)
            ep_return += float(reward)
            if term or trunc:
                break
        total += ep_return
    return total / n_episodes


if __name__ == "__main__":
    print("=== SARSA on CliffWalking ===")
    q, history = train_sarsa()
    avg = evaluate(q)
    print(f"\nEval (100 episodes, greedy): avg_return={avg:.1f}")
    print(f"RESULT\tavg_return={avg:.1f}\tepochs={len(history)}\tseed=42")
