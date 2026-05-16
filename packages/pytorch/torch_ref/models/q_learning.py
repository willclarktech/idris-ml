"""Q-learning (Watkins 1989) on CliffWalking-v0.

Tabular off-policy TD(0) control. Self-contained CliffWalking grid matching
the Idris `Gym.ToyText.CliffWalking` env (4x12 grid, start (3,0), goal
(3,11), cliff along row 3 cols 1..10 with -100 reward + reset-to-start).
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass

import numpy as np

from torch_ref.training.runner import format_elapsed, mem_suffix

# ---------------------------------------------------------------------------
# CliffWalking environment
# ---------------------------------------------------------------------------

NUM_ROWS = 4
NUM_COLS = 12
NUM_STATES = NUM_ROWS * NUM_COLS  # 48
NUM_ACTIONS = 4  # 0=up, 1=right, 2=down, 3=left
START_ROW = 3
START_COL = 0
GOAL_ROW = 3
GOAL_COL = 11
MAX_STEPS = 100


@dataclass
class CWState:
    row: int = START_ROW
    col: int = START_COL

    def encode(self) -> int:
        return self.row * NUM_COLS + self.col


def on_cliff(row: int, col: int) -> bool:
    return row == 3 and 1 <= col <= 10


def cw_step(state: CWState, action: int) -> tuple[float, CWState, bool]:
    """One step. Returns (reward, next_state, done)."""
    r, c = state.row, state.col
    if action == 0:
        r -= 1
    elif action == 1:
        c += 1
    elif action == 2:
        r += 1
    else:
        c -= 1
    r = max(0, min(NUM_ROWS - 1, r))
    c = max(0, min(NUM_COLS - 1, c))

    if on_cliff(r, c):
        return -100.0, CWState(START_ROW, START_COL), False
    if r == GOAL_ROW and c == GOAL_COL:
        return -1.0, CWState(r, c), True
    return -1.0, CWState(r, c), False


# ---------------------------------------------------------------------------
# Q-learning
# ---------------------------------------------------------------------------


def eps_greedy(q_row: np.ndarray, epsilon: float, rng: random.Random) -> int:
    """Epsilon-greedy action selection. Ties broken by first-argmax."""
    if rng.random() < epsilon:
        return rng.randrange(NUM_ACTIONS)
    return int(np.argmax(q_row))


def q_learning_episode(
    q: np.ndarray,
    alpha: float,
    gamma: float,
    epsilon: float,
    rng: random.Random,
    max_steps: int = MAX_STEPS,
) -> float:
    """Run one episode with Q-learning updates. Returns episodic return."""
    state = CWState()
    total_reward = 0.0
    for _ in range(max_steps):
        s = state.encode()
        action = eps_greedy(q[s], epsilon, rng)
        reward, next_state, done = cw_step(state, action)
        total_reward += reward

        s_next = next_state.encode()
        target = reward if done else reward + gamma * float(np.max(q[s_next]))
        q[s, action] += alpha * (target - q[s, action])

        state = next_state
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
) -> tuple[np.ndarray, list[float]]:
    """Train Q-learning on CliffWalking. Returns (Q-table, history of returns)."""
    rng = random.Random(seed)
    q = np.zeros((NUM_STATES, NUM_ACTIONS), dtype=np.float64)
    history: list[float] = []
    t_start = time.monotonic()
    for epoch in range(epochs):
        ret = q_learning_episode(q, alpha, gamma, epsilon, rng)
        history.append(ret)
        if (epoch + 1) % log_every == 0:
            recent = sum(history[-100:]) / min(len(history), 100)
            print(
                f"  {format_elapsed(t_start)} {epoch + 1}\tloss={-ret:.6f}"
                f"{mem_suffix()}\treturn={ret:.1f}\trecent_100={recent:.1f}"
            )
    return q, history


def evaluate(q: np.ndarray, n_episodes: int = 100) -> float:
    """Greedy evaluation. Returns mean return."""
    total = 0.0
    for _ in range(n_episodes):
        state = CWState()
        ep_return = 0.0
        for _ in range(MAX_STEPS):
            s = state.encode()
            action = int(np.argmax(q[s]))
            reward, state, done = cw_step(state, action)
            ep_return += reward
            if done:
                break
        total += ep_return
    return total / n_episodes


if __name__ == "__main__":
    print("=== Q-learning on CliffWalking ===")
    q, history = train_q_learning()
    avg = evaluate(q)
    print(f"\nEval (100 episodes, greedy): avg_return={avg:.1f}")
    print(f"RESULT\tavg_return={avg:.1f}\tepochs={len(history)}\tseed=42")
