"""SARSA (Rummery & Niranjan 1994) on CliffWalking-v0.

On-policy TD(0) control. Same env as Q-learning; the only difference is the
TD target uses the next action actually selected under the current policy
(epsilon-greedy) rather than max over next actions. Classic Sutton & Barto
result: SARSA converges to a safer path further from the cliff than
Q-learning, which walks right along the edge.
"""

from __future__ import annotations

import random

import numpy as np

from torch_ref.models.q_learning import (
    MAX_STEPS,
    NUM_ACTIONS,
    NUM_STATES,
    CWState,
    cw_step,
    eps_greedy,
)


def sarsa_episode(
    q: np.ndarray,
    alpha: float,
    gamma: float,
    epsilon: float,
    rng: random.Random,
    max_steps: int = MAX_STEPS,
) -> float:
    """Run one episode with SARSA updates. Returns episodic return."""
    state = CWState()
    s = state.encode()
    action = eps_greedy(q[s], epsilon, rng)
    total_reward = 0.0
    for _ in range(max_steps):
        reward, next_state, done = cw_step(state, action)
        total_reward += reward
        s_next = next_state.encode()
        if done:
            q[s, action] += alpha * (reward - q[s, action])
            break
        next_action = eps_greedy(q[s_next], epsilon, rng)
        target = reward + gamma * q[s_next, next_action]
        q[s, action] += alpha * (target - q[s, action])
        state = next_state
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
) -> tuple[np.ndarray, list[float]]:
    """Train SARSA on CliffWalking. Returns (Q-table, history)."""
    rng = random.Random(seed)
    q = np.zeros((NUM_STATES, NUM_ACTIONS), dtype=np.float64)
    history: list[float] = []
    for epoch in range(epochs):
        ret = sarsa_episode(q, alpha, gamma, epsilon, rng)
        history.append(ret)
        if (epoch + 1) % log_every == 0:
            recent = sum(history[-100:]) / min(len(history), 100)
            print(f"  epoch {epoch + 1:4d}  return={ret:.1f}  recent_100={recent:.1f}")
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
    print("=== SARSA on CliffWalking ===")
    q, history = train_sarsa()
    avg = evaluate(q)
    print(f"\nEval (100 episodes, greedy): avg_return={avg:.1f}")
    print(f"RESULT\tavg_return={avg:.1f}\tepochs={len(history)}\tseed=42")
