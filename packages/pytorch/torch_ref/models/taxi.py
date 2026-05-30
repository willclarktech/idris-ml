"""Q-learning on Taxi-v4.

Tabular off-policy TD(0) on a deterministic 5x5 grid. Uses canonical
`gym.make("Taxi-v4")` for env physics — state encoding
`((row*5 + col)*5 + pass_idx)*4 + dest`, four pickup/dropoff locations
R(0,0), G(0,4), Y(4,0), B(4,3), rewards -1/step, +20 successful
dropoff, -10 illegal pickup/dropoff, 200-step TimeLimit.

Reset state is pinned to (taxi=(2,2), pass=R(0), dest=B(3)) — encoded
243 — to mirror `Gym.ToyText.Taxi.defaultStart` on the Idris side
(canonical Taxi-v4 randomizes; both Idris and torch_ref pin to the
same fixed start for paired convergence).

Optimal trajectory under canonical walls from this start is 13
actions → return +8 (4 moves to R + pickup + 7 moves to B + dropoff;
canonical walls add an extra detour vs the suboptimal-shorter path
through the SW corner, but the fixed-start optimal length is
unchanged at 13).
"""

from __future__ import annotations

import random
import time

import gymnasium as gym
import numpy as np

from torch_ref.training.runner import format_elapsed, mem_suffix

NUM_STATES = 500  # 5 rows * 5 cols * 5 pass * 4 dest
NUM_ACTIONS = 6  # 0=down, 1=up, 2=right, 3=left, 4=pickup, 5=dropoff
MAX_STEPS = 200  # gymnasium Taxi-v4 default TimeLimit

START_TAXI_ROW = 2
START_TAXI_COL = 2
START_PASS_IDX = 0  # R
START_DEST_IDX = 3  # B


def _pin_start(env: gym.Env) -> int:
    """Pin env state to the fixed default start (taxi=(2,2), pass=R, dest=B)
    and return the encoded state index."""
    encoded = env.unwrapped.encode(  # pyright: ignore[reportAttributeAccessIssue]
        START_TAXI_ROW, START_TAXI_COL, START_PASS_IDX, START_DEST_IDX
    )
    env.unwrapped.s = encoded  # pyright: ignore[reportAttributeAccessIssue]
    return int(encoded)


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
    env.reset()
    s = _pin_start(env)
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
    epochs: int = 20000,
    alpha: float = 0.1,
    gamma: float = 0.99,
    epsilon: float = 0.1,
    seed: int = 42,
    log_every: int = 2000,
) -> tuple[np.ndarray, list[float]]:
    rng = random.Random(seed)
    env = gym.make("Taxi-v4")
    env.reset(seed=seed)
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
                f"{mem_suffix()}\treturn={ret:.1f}\trecent_1000={recent:.2f}"
            )
    return q, history


def evaluate(q: np.ndarray, n_episodes: int = 100) -> float:
    """Greedy evaluation. Deterministic env + fixed start = single trajectory
    repeated; loop kept for parity with the Idris example's output format."""
    env = gym.make("Taxi-v4")
    env.reset(seed=0)
    total = 0.0
    for _ in range(n_episodes):
        env.reset()
        s = _pin_start(env)
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
    print("=== Q-learning on Taxi-v4 ===")
    q, history = train_q_learning()
    avg = evaluate(q)
    print(f"\nEval (100 episodes, greedy): avg_return={avg:.1f}")
    print(f"RESULT\tavg_return={avg:.1f}\tepochs={len(history)}\tseed=42")
