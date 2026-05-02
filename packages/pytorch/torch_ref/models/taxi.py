"""Q-learning on Taxi-v3.

Tabular off-policy TD(0) on a deterministic 5x5 grid with 4 designated
locations (R, G, Y, B). Self-contained env matching the Idris
`Gym.ToyText.Taxi`: state encoding `((row*5 + col)*5 + pass)*4 + dest`,
walls between cols 1-2 in rows 0-1 and between cols 2-3 in rows 3-4,
rewards -1/step, +20 successful dropoff, -10 illegal pickup/dropoff.

Default fixed start: taxi (2,2), passenger R (idx 0), destination B (idx 3).
Optimal trajectory under those walls is 13 actions → return = +8.
"""

from __future__ import annotations

import random

import numpy as np

# ---------------------------------------------------------------------------
# Taxi environment
# ---------------------------------------------------------------------------

NUM_ROWS = 5
NUM_COLS = 5
NUM_STATES = NUM_ROWS * NUM_COLS * 5 * 4  # 500
NUM_ACTIONS = 6  # 0=down, 1=up, 2=right, 3=left, 4=pickup, 5=dropoff
MAX_STEPS = 200

# Locations: R=0(0,0), G=1(0,4), Y=2(4,0), B=3(4,3)
LOC_ROW = (0, 0, 4, 4)
LOC_COL = (0, 4, 0, 3)


def encode(row: int, col: int, pass_idx: int, dest: int) -> int:
    return ((row * NUM_COLS + col) * 5 + pass_idx) * 4 + dest


def blocked(r: int, c: int, r2: int, c2: int) -> bool:
    """Wall check: cannot move horizontally between (r, lo)-(r, hi)."""
    if r != r2:
        return False
    lo, hi = (c, c2) if c < c2 else (c2, c)
    if (r == 0 or r == 1) and lo == 1 and hi == 2:
        return True
    return (r == 3 or r == 4) and lo == 2 and hi == 3


class TaxiState:
    __slots__ = ("row", "col", "pass_idx", "dest")

    def __init__(self, row: int, col: int, pass_idx: int, dest: int):
        self.row = row
        self.col = col
        self.pass_idx = pass_idx
        self.dest = dest

    def encode(self) -> int:
        return encode(self.row, self.col, self.pass_idx, self.dest)

    def copy(self) -> TaxiState:
        return TaxiState(self.row, self.col, self.pass_idx, self.dest)


def default_start() -> TaxiState:
    return TaxiState(2, 2, 0, 3)


def t_step(state: TaxiState, action: int) -> tuple[float, TaxiState, bool]:
    """One step. Returns (reward, next_state, done). Mutates `state`'s copy."""
    s = state.copy()
    if action <= 3:
        # Move action
        dr = (1, -1, 0, 0)[action]
        dc = (0, 0, 1, -1)[action]
        rn = max(0, min(NUM_ROWS - 1, s.row + dr))
        cn = max(0, min(NUM_COLS - 1, s.col + dc))
        if not blocked(s.row, s.col, rn, cn):
            s.row, s.col = rn, cn
        return -1.0, s, False
    if action == 4:
        # Pickup
        if (
            s.pass_idx < 4
            and s.row == LOC_ROW[s.pass_idx]
            and s.col == LOC_COL[s.pass_idx]
        ):
            s.pass_idx = 4
            return -1.0, s, False
        return -10.0, s, False
    # Dropoff (action == 5)
    if (
        s.pass_idx == 4
        and s.row == LOC_ROW[s.dest]
        and s.col == LOC_COL[s.dest]
    ):
        s.pass_idx = s.dest
        return 20.0, s, True
    return -10.0, s, False


# ---------------------------------------------------------------------------
# Q-learning
# ---------------------------------------------------------------------------


def eps_greedy(q_row: np.ndarray, epsilon: float, rng: random.Random) -> int:
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
    state = default_start()
    total_reward = 0.0
    for _ in range(max_steps):
        s = state.encode()
        action = eps_greedy(q[s], epsilon, rng)
        reward, next_state, done = t_step(state, action)
        total_reward += reward
        s_next = next_state.encode()
        target = reward if done else reward + gamma * float(np.max(q[s_next]))
        q[s, action] += alpha * (target - q[s, action])
        state = next_state
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
    q = np.zeros((NUM_STATES, NUM_ACTIONS), dtype=np.float64)
    history: list[float] = []
    for epoch in range(epochs):
        ret = q_learning_episode(q, alpha, gamma, epsilon, rng)
        history.append(ret)
        if (epoch + 1) % log_every == 0:
            recent = sum(history[-1000:]) / min(len(history), 1000)
            print(f"  epoch {epoch + 1:5d}  return={ret:6.1f}  recent_1000={recent:.2f}")
    return q, history


def evaluate(q: np.ndarray, n_episodes: int = 100) -> float:
    """Greedy evaluation. Deterministic env + fixed start = single trajectory
    repeated; loop kept for parity with the Idris example's output format."""
    total = 0.0
    for _ in range(n_episodes):
        state = default_start()
        ep_return = 0.0
        for _ in range(MAX_STEPS):
            action = int(np.argmax(q[state.encode()]))
            reward, state, done = t_step(state, action)
            ep_return += reward
            if done:
                break
        total += ep_return
    return total / n_episodes


if __name__ == "__main__":
    print("=== Q-learning on Taxi-v3 ===")
    q, history = train_q_learning()
    avg = evaluate(q)
    print(f"\nEval (100 episodes, greedy): avg_return={avg:.1f}")
    print(f"RESULT\tavg_return={avg:.1f}\tepochs={len(history)}\tseed=42")
