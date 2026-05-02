"""Q-learning on FrozenLake-v1 (slippery 4x4).

Tabular off-policy TD(0) on a stochastic env. Uses canonical
`gym.make("FrozenLake-v1")` for env physics (default 4x4 map,
is_slippery=True) — matches `Gym.ToyText.FrozenLake` on the Idris side
on map layout, slip distribution (1/3 intended, 1/3 each
perpendicular), reward (+1 at goal, 0 elsewhere), action mapping
(0=left, 1=down, 2=right, 3=up), and TimeLimit (100 steps).
"""

from __future__ import annotations

import random
import time

import gymnasium as gym
import numpy as np

from torch_ref.training.runner import format_elapsed, mem_suffix

NUM_ROWS = 4
NUM_COLS = 4
NUM_STATES = NUM_ROWS * NUM_COLS  # 16
NUM_ACTIONS = 4  # 0=left, 1=down, 2=right, 3=up
MAX_STEPS = 100  # gymnasium FrozenLake-v1 default TimeLimit


def eps_greedy(q_row: np.ndarray, epsilon: float, rng: random.Random) -> int:
    if rng.random() < epsilon:
        return rng.randrange(NUM_ACTIONS)
    return int(np.argmax(q_row))


def q_learning_episode(
    env: gym.Env,
    q: np.ndarray,
    alpha: float,
    gamma: float,
    epsilon: float,
    rng: random.Random,
    max_steps: int = MAX_STEPS,
) -> float:
    pos_obs, _ = env.reset()
    pos = int(pos_obs)
    total_reward = 0.0
    for _ in range(max_steps):
        action = eps_greedy(q[pos], epsilon, rng)
        next_obs, reward, term, trunc, _ = env.step(action)
        pos_next = int(next_obs)
        done = bool(term or trunc)
        total_reward += float(reward)
        target = float(reward) if done else float(reward) + gamma * float(np.max(q[pos_next]))
        q[pos, action] += alpha * (target - q[pos, action])
        pos = pos_next
        if done:
            break
    return total_reward


def train_q_learning(
    epochs: int = 10000,
    alpha: float = 0.1,
    gamma: float = 0.99,
    epsilon: float = 0.3,
    seed: int = 42,
    log_every: int = 1000,
) -> tuple[np.ndarray, list[float]]:
    rng = random.Random(seed)
    env = gym.make("FrozenLake-v1")
    env.reset(seed=seed)  # seed the env's slip RNG once
    q = np.zeros((NUM_STATES, NUM_ACTIONS), dtype=np.float64)
    history: list[float] = []
    t_start = time.monotonic()
    for epoch in range(epochs):
        ret = q_learning_episode(env, q, alpha, gamma, epsilon, rng)
        history.append(ret)
        if (epoch + 1) % log_every == 0:
            recent = sum(history[-1000:]) / min(len(history), 1000)
            print(
                f"  {format_elapsed(t_start)} {epoch + 1}\tloss={-ret:.6f}"
                f"{mem_suffix()}\treturn={ret:.1f}\trecent_1000={recent:.3f}"
            )
    return q, history


def evaluate(q: np.ndarray, n_episodes: int = 100, seed: int = 0) -> float:
    """Greedy evaluation. Slip dynamics still apply, so even an optimal
    policy fails some episodes. avg_return == success rate."""
    env = gym.make("FrozenLake-v1")
    env.reset(seed=seed)
    total = 0.0
    for _ in range(n_episodes):
        pos_obs, _ = env.reset()
        pos = int(pos_obs)
        ep_return = 0.0
        for _ in range(MAX_STEPS):
            action = int(np.argmax(q[pos]))
            next_obs, reward, term, trunc, _ = env.step(action)
            pos = int(next_obs)
            ep_return += float(reward)
            if term or trunc:
                break
        total += ep_return
    return total / n_episodes


if __name__ == "__main__":
    print("=== Q-learning on FrozenLake (slippery 4x4) ===")
    q, history = train_q_learning()
    avg = evaluate(q)
    print(f"\nEval (100 episodes, greedy): avg_return={avg:.2f}")
    print(f"RESULT\tavg_return={avg:.2f}\tepochs={len(history)}\tseed=42")
