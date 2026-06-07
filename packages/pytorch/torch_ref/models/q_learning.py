"""Q-learning (Watkins 1989) on CliffWalking-v1.

Tabular off-policy TD(0) control. Uses canonical
`gym.make("CliffWalking-v1")` for env physics — 4x12 grid, start
(3,0)→state 36, goal (3,11)→state 47, cliff along row 3 cols 1..10
with -100 reward + reset-to-start, -1/step otherwise. Action mapping
0=up, 1=right, 2=down, 3=left.

Note: CliffWalking-v1 has no TimeLimit wrapper (env.spec.max_episode_steps is None).
We enforce MAX_STEPS=100 via the training loop's `range(max_steps)` cap.
"""

from __future__ import annotations

import random
import time
from typing import cast

import gymnasium as gym
import numpy as np
import numpy.typing as npt

from torch_ref.training.runner import format_elapsed, mem_suffix

NUM_ROWS = 4
NUM_COLS = 12
NUM_STATES = NUM_ROWS * NUM_COLS  # 48
NUM_ACTIONS = 4  # 0=up, 1=right, 2=down, 3=left
MAX_STEPS = 100


def eps_greedy(q_row: npt.NDArray[np.float64], epsilon: float, rng: random.Random) -> int:
    """Epsilon-greedy action selection. Ties broken by first-argmax."""
    if rng.random() < epsilon:
        return rng.randrange(NUM_ACTIONS)
    return int(np.argmax(q_row))


def q_learning_episode(
    env: gym.Env[int, int],
    q: npt.NDArray[np.float64],
    alpha: float,
    gamma: float,
    epsilon: float,
    rng: random.Random,
    max_steps: int = MAX_STEPS,
) -> float:
    """Run one episode with Q-learning updates. Returns episodic return."""
    obs, _ = env.reset()
    s = int(obs)
    total_reward = 0.0
    for _ in range(max_steps):
        action = eps_greedy(q[s], epsilon, rng)
        next_obs, reward, term, trunc, _ = env.step(action)
        s_next = int(next_obs)
        total_reward += float(reward)
        done = bool(term or trunc)
        target = float(reward) if done else float(reward) + gamma * float(np.max(q[s_next]))
        q[s, action] += alpha * (target - q[s, action])
        s = s_next
        if done:
            break
    return total_reward


def train_q_learning(
    epochs: int = 500,
    alpha: float = 0.5,
    gamma: float = 1.0,
    epsilon: float = 0.1,
    seed: int = 42,
    log_every: int = 100,
) -> tuple[npt.NDArray[np.float64], list[float]]:
    """Train Q-learning on CliffWalking. Returns (Q-table, history of returns)."""
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
        ret = q_learning_episode(env, q, alpha, gamma, epsilon, rng)
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
    print("=== Q-learning on CliffWalking ===")
    q, history = train_q_learning()
    avg = evaluate(q)
    print(f"\nEval (100 episodes, greedy): avg_return={avg:.1f}")
    print(f"RESULT\tavg_return={avg:.1f}\tepochs={len(history)}\tseed=42")
