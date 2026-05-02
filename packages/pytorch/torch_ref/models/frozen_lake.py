"""Q-learning on FrozenLake-v1 (slippery 4x4).

Tabular off-policy TD(0) on a stochastic env. Self-contained 4x4 grid matching
the Idris `Gym.ToyText.FrozenLake` env (Gymnasium 4x4 default map; isSlippery
gives intended direction probability 1/3 + each perpendicular 1/3).

Map layout:
    S F F F
    F H F H
    F F F H
    H F F G
Reward: +1 at goal, 0 elsewhere. avg_return == greedy success rate.
"""

from __future__ import annotations

import random

import numpy as np

# ---------------------------------------------------------------------------
# FrozenLake environment (slippery 4x4)
# ---------------------------------------------------------------------------

NUM_ROWS = 4
NUM_COLS = 4
NUM_STATES = NUM_ROWS * NUM_COLS  # 16
NUM_ACTIONS = 4  # 0=left, 1=down, 2=right, 3=up
MAX_STEPS = 100

# S=0, F=1, H=2, G=3
TILE_HOLE = 2
TILE_GOAL = 3
DEFAULT_MAP = np.array(
    [
        0, 1, 1, 1,
        1, 2, 1, 2,
        1, 1, 1, 2,
        2, 1, 1, 3,
    ]
)


def move_det(pos: int, action: int) -> int:
    r, c = pos // NUM_COLS, pos % NUM_COLS
    if action == 0:
        c -= 1  # left
    elif action == 1:
        r += 1  # down
    elif action == 2:
        c += 1  # right
    else:
        r -= 1  # up
    r = max(0, min(NUM_ROWS - 1, r))
    c = max(0, min(NUM_COLS - 1, c))
    return r * NUM_COLS + c


def slip_action(intended: int, rng: random.Random) -> int:
    """Slippery dynamics: 1/3 intended, 1/3 each perpendicular."""
    choice = rng.randrange(3)
    if choice == 0:
        return intended
    # Perpendicular pair (left-perpendicular, right-perpendicular)
    perp = {
        0: (3, 1),  # left  -> up,    down
        1: (0, 2),  # down  -> left,  right
        2: (1, 3),  # right -> down,  up
        3: (2, 0),  # up    -> right, left
    }[intended]
    return perp[0] if choice == 1 else perp[1]


def fl_step(pos: int, action: int, rng: random.Random) -> tuple[float, int, bool]:
    """One slippery step. Returns (reward, next_pos, done)."""
    actual = slip_action(action, rng)
    pos_next = move_det(pos, actual)
    tile = DEFAULT_MAP[pos_next]
    if tile == TILE_GOAL:
        return 1.0, pos_next, True
    if tile == TILE_HOLE:
        return 0.0, pos_next, True
    return 0.0, pos_next, False


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
    pos = 0
    total_reward = 0.0
    for _ in range(max_steps):
        action = eps_greedy(q[pos], epsilon, rng)
        reward, pos_next, done = fl_step(pos, action, rng)
        total_reward += reward
        target = reward if done else reward + gamma * float(np.max(q[pos_next]))
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
    q = np.zeros((NUM_STATES, NUM_ACTIONS), dtype=np.float64)
    history: list[float] = []
    for epoch in range(epochs):
        ret = q_learning_episode(q, alpha, gamma, epsilon, rng)
        history.append(ret)
        if (epoch + 1) % log_every == 0:
            recent = sum(history[-1000:]) / min(len(history), 1000)
            print(f"  epoch {epoch + 1:5d}  return={ret:.1f}  recent_1000={recent:.3f}")
    return q, history


def evaluate(q: np.ndarray, n_episodes: int = 100, seed: int = 0) -> float:
    """Greedy evaluation. Slip dynamics still apply, so even an optimal
    policy fails some episodes. avg_return == success rate."""
    rng = random.Random(seed)
    total = 0.0
    for _ in range(n_episodes):
        pos = 0
        ep_return = 0.0
        for _ in range(MAX_STEPS):
            action = int(np.argmax(q[pos]))
            reward, pos, done = fl_step(pos, action, rng)
            ep_return += reward
            if done:
                break
        total += ep_return
    return total / n_episodes


if __name__ == "__main__":
    print("=== Q-learning on FrozenLake (slippery 4x4) ===")
    q, history = train_q_learning()
    avg = evaluate(q)
    print(f"\nEval (100 episodes, greedy): avg_return={avg:.2f}")
    print(f"RESULT\tavg_return={avg:.2f}\tepochs={len(history)}\tseed=42")
