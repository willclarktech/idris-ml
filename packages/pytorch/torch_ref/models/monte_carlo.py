"""First-visit MC control on Blackjack-v1.

On-policy first-visit Monte Carlo with epsilon-greedy exploration and
incremental-mean updates (Sutton & Barto chapter 5). Uses canonical
`gym.make("Blackjack-v1")` for env physics — Sutton & Barto rules
(natural=False, sab=True), uniform 13-card suit (A=1/13, 2..9 each
1/13, 10 with weight 4/13 from {10, J, Q, K}), dealer sticks at 17,
+1/-1/0 win/loss/draw reward.

Aligned with idris-gym `Gym.ToyText.Blackjack` (canonical distribution
adopted on both sides in the same commit, replacing the prior
Ace=2/13, 10=3/13 weighting).
"""

from __future__ import annotations

import random

import gymnasium as gym
import numpy as np

NUM_ACTIONS = 2  # 0=stick, 1=hit
# State: (player_sum in 4..21, dealer_show in 1..10, usable in {0,1})
# idx = (ps - 4) * 20 + (ds - 1) * 2 + ua   ->  0..359
NUM_STATES = 400


def encode(player_sum: int, dealer_show: int, usable: bool) -> int:
    ps = max(4, min(21, player_sum)) - 4
    ds = max(1, min(10, dealer_show)) - 1
    ua = 1 if usable else 0
    return ps * 20 + ds * 2 + ua


def play_hand(
    env: gym.Env,
    q: np.ndarray,
    epsilon: float,
    rng: random.Random,
    greedy: bool = False,
) -> tuple[list[tuple[int, int]], float]:
    """Play one Blackjack hand. Returns (trajectory [(state_idx, action)],
    terminal_reward)."""
    obs, _ = env.reset()
    trajectory: list[tuple[int, int]] = []
    reward = 0.0
    while True:
        player_sum, dealer_show, usable = obs
        state_idx = encode(int(player_sum), int(dealer_show), bool(usable))
        if greedy:
            action = int(np.argmax(q[state_idx]))
        elif rng.random() < epsilon:
            action = rng.randrange(NUM_ACTIONS)
        else:
            action = int(np.argmax(q[state_idx]))
        trajectory.append((state_idx, action))
        next_obs, r, term, trunc, _ = env.step(action)
        reward = float(r)
        if term or trunc:
            break
        obs = next_obs
    return trajectory, reward


def mc_episode(
    env: gym.Env,
    q: np.ndarray,
    counts: np.ndarray,
    epsilon: float,
    rng: random.Random,
) -> float:
    """Run one hand, apply first-visit MC updates. Returns terminal reward."""
    trajectory, reward = play_hand(env, q, epsilon, rng, greedy=False)
    visited: set[tuple[int, int]] = set()
    for s, a in trajectory:
        if (s, a) not in visited:
            visited.add((s, a))
            counts[s, a] += 1
            q[s, a] += (reward - q[s, a]) / counts[s, a]
    return reward


def train_mc(
    epochs: int = 50000,
    epsilon: float = 0.1,
    seed: int = 42,
    log_every: int = 10000,
) -> tuple[np.ndarray, list[float]]:
    """Train first-visit MC on Blackjack. Returns (Q, history)."""
    rng = random.Random(seed)
    env = gym.make("Blackjack-v1")
    env.reset(seed=seed)
    q = np.zeros((NUM_STATES, NUM_ACTIONS), dtype=np.float64)
    counts = np.zeros((NUM_STATES, NUM_ACTIONS), dtype=np.int64)
    history: list[float] = []
    for epoch in range(epochs):
        reward = mc_episode(env, q, counts, epsilon, rng)
        history.append(reward)
        if (epoch + 1) % log_every == 0:
            recent = sum(history[-1000:]) / min(len(history), 1000)
            win_rate = sum(1 for r in history[-1000:] if r > 0) / min(len(history), 1000)
            print(
                f"  epoch {epoch + 1:6d}  mean_reward={recent:+.3f}  win_rate={win_rate:.3f}"
            )
    return q, history


def evaluate(q: np.ndarray, n_episodes: int = 10000, seed: int = 0) -> dict[str, float]:
    """Greedy evaluation. Returns {win, draw, loss, avg_reward}."""
    rng = random.Random(seed)
    env = gym.make("Blackjack-v1")
    env.reset(seed=seed)
    wins = draws = losses = 0
    total_reward = 0.0
    for _ in range(n_episodes):
        _, r = play_hand(env, q, 0.0, rng, greedy=True)
        total_reward += r
        if r > 0:
            wins += 1
        elif r < 0:
            losses += 1
        else:
            draws += 1
    return {
        "win": wins / n_episodes,
        "draw": draws / n_episodes,
        "loss": losses / n_episodes,
        "avg_reward": total_reward / n_episodes,
    }


if __name__ == "__main__":
    print("=== First-visit MC on Blackjack ===")
    q, history = train_mc()
    stats = evaluate(q)
    print(
        f"\nEval (10000 episodes, greedy): win={stats['win']:.3f} draw={stats['draw']:.3f} "
        f"loss={stats['loss']:.3f} avg_reward={stats['avg_reward']:+.3f}"
    )
    print(
        f"RESULT\twin_rate={stats['win']:.3f}\tavg_reward={stats['avg_reward']:+.3f}\t"
        f"epochs={len(history)}\tseed=42"
    )
