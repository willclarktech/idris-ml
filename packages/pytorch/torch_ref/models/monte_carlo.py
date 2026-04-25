"""First-visit MC control on Blackjack-v1.

On-policy first-visit Monte Carlo with epsilon-greedy exploration and
incremental-mean updates (Sutton & Barto chapter 5). Self-contained
Blackjack env matching the Idris `Gym.ToyText.Blackjack` env: infinite
deck with replacement, dealer sticks at 17, natural=False.
"""

from __future__ import annotations

import random

import numpy as np

# ---------------------------------------------------------------------------
# Blackjack environment
# ---------------------------------------------------------------------------

# Card value distribution matches Idris: 0->Ace=1; 10,11,12->J,Q,K=10; 1..9 as-is.
# Equivalent to: draw value in 1..10 with 10 having 4x weight.

NUM_ACTIONS = 2  # 0=stick, 1=hit
# State: (player_sum in 4..21, dealer_show in 1..10, usable in {0,1})
# idx = (ps - 4) * 20 + (ds - 1) * 2 + ua   ->  0..359
NUM_STATES = 400  # slight overhead for safety


def draw_card(rng: random.Random) -> int:
    """Draw a card. Distribution matches Idris env: card 1 has weight 2/13
    (n=0 or n=1), cards 2..9 have weight 1/13 each, card 10 has weight 3/13
    (n=10, 11, 12). Not a realistic deck but keeps the envs in lockstep."""
    n = rng.randrange(13)
    if n == 0:
        return 1  # Ace (n=0 special case)
    if n >= 10:
        return 10  # J, Q, K
    return n  # 1..9


def hand_sum(hand: list[int]) -> int:
    total = sum(hand)
    aces = hand.count(1)
    while aces > 0 and total + 10 <= 21:
        total += 10
        aces -= 1
    return total


def usable_ace(hand: list[int]) -> bool:
    return 1 in hand and sum(hand) + 10 <= 21


def encode(player_sum: int, dealer_show: int, usable: bool) -> int:
    ps = max(4, min(21, player_sum)) - 4
    ds = max(1, min(10, dealer_show)) - 1
    ua = 1 if usable else 0
    return ps * 20 + ds * 2 + ua


def play_hand(
    q: np.ndarray, epsilon: float, rng: random.Random, greedy: bool = False
) -> tuple[list[tuple[int, int]], float]:
    """Play one Blackjack hand. Returns (trajectory [(state_idx, action)], terminal_reward)."""
    player = [draw_card(rng), draw_card(rng)]
    dealer = [draw_card(rng), draw_card(rng)]
    trajectory: list[tuple[int, int]] = []
    reward = 0.0

    while True:
        p_sum = hand_sum(player)
        if p_sum > 21:
            break  # shouldn't happen on initial deal, but defensive
        state_idx = encode(p_sum, dealer[0], usable_ace(player))
        if greedy:
            action = int(np.argmax(q[state_idx]))
        elif rng.random() < epsilon:
            action = rng.randrange(NUM_ACTIONS)
        else:
            action = int(np.argmax(q[state_idx]))

        trajectory.append((state_idx, action))

        if action == 0:  # stick
            while hand_sum(dealer) < 17:
                dealer.append(draw_card(rng))
            d_sum = hand_sum(dealer)
            p_sum = hand_sum(player)
            if d_sum > 21 or p_sum > d_sum:
                reward = 1.0
            elif p_sum == d_sum:
                reward = 0.0
            else:
                reward = -1.0
            break
        else:  # hit
            player.append(draw_card(rng))
            if hand_sum(player) > 21:
                reward = -1.0
                break

    return trajectory, reward


# ---------------------------------------------------------------------------
# First-visit MC control
# ---------------------------------------------------------------------------


def mc_episode(
    q: np.ndarray, counts: np.ndarray, epsilon: float, rng: random.Random
) -> float:
    """Run one hand, apply first-visit MC updates. Returns terminal reward."""
    trajectory, reward = play_hand(q, epsilon, rng, greedy=False)
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
    q = np.zeros((NUM_STATES, NUM_ACTIONS), dtype=np.float64)
    counts = np.zeros((NUM_STATES, NUM_ACTIONS), dtype=np.int64)
    history: list[float] = []
    for epoch in range(epochs):
        reward = mc_episode(q, counts, epsilon, rng)
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
    wins = draws = losses = 0
    total_reward = 0.0
    for _ in range(n_episodes):
        _, r = play_hand(q, 0.0, rng, greedy=True)
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
