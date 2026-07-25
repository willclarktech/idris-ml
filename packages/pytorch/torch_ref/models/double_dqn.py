"""Double DQN (van Hasselt et al. 2016) on CartPole-v1.

Vanilla DQN bootstraps with `max_a Q_target(s', a)`, which both selects
*and* evaluates the next action with the target net — that shared max
overestimates Q (maximization bias). Double DQN decouples the two: the
**online** net selects the next action `a* = argmax_a Q_online(s', a)`,
the **target** net evaluates it `Q_target(s', a*)`. Only the bootstrap
target changes; replay, target sync, eps-greedy, and the batched
`SyncVectorEnv` rollout are identical to `dqn.py` (leaf helpers reused
by import).

Convergence target: greedy-evaluation return >= 150 on CartPole, pass
rate measured across five seeds (per docs/develop/reference-alignment.md).
"""

from __future__ import annotations

import copy
import random
import time
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
import torch.nn.functional as F

from torch_ref.models.dqn import (
    NUM_ENVS,
    QNetwork,
    ReplayBuffer,
    eps_greedy_batched,
    evaluate,
    linear_epsilon,
    make_cartpole_vec_env,
)
from torch_ref.models.reinforce import MAX_STEPS, obs_tensor
from torch_ref.training.runner import format_elapsed, mem_suffix

if TYPE_CHECKING:
    import gymnasium as gym

# Re-export so callers (script, tests) can `from double_dqn import evaluate`.
__all__ = ["double_dqn_update", "double_dqn_episode_batched", "train_double_dqn", "evaluate"]


def double_dqn_update(
    q: QNetwork,
    target: QNetwork,
    optimizer: torch.optim.Optimizer,
    buffer: ReplayBuffer,
    batch_size: int,
    gamma: float,
    rng: random.Random,
) -> float:
    """One gradient step with the Double DQN target. Returns the loss value.

    The single difference from `dqn.dqn_update`: the next action is
    *selected* by the online net and *evaluated* by the target net,
    instead of `target(next_obs).max()` doing both.
    """
    obs, actions, rewards, next_obs, dones = buffer.sample(batch_size, rng)
    q_vals = q(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_actions = q(next_obs).argmax(dim=1, keepdim=True)  # online selects a*
        next_q = target(next_obs).gather(1, next_actions).squeeze(1)  # target evaluates
        target_vals = rewards + gamma * next_q * (1.0 - dones)
    loss = F.mse_loss(q_vals, target_vals)
    optimizer.zero_grad()
    # torch stubs leave Tensor.backward's params untyped.
    loss.backward()  # pyright: ignore[reportUnknownMemberType]
    torch.nn.utils.clip_grad_norm_(q.parameters(), 10.0)
    optimizer.step()
    return float(loss.item())


def double_dqn_episode_batched(
    vec_env: gym.vector.SyncVectorEnv,
    obs_np: np.ndarray,
    q: QNetwork,
    target: QNetwork,
    optimizer: torch.optim.Optimizer,
    buffer: ReplayBuffer,
    step_count: int,
    batch_size: int,
    gamma: float,
    target_sync_every: int,
    rng: random.Random,
) -> tuple[int, float, np.ndarray]:
    """Batched Double DQN episode — `dqn.dqn_episode_batched` with the
    Double update. env-0 is the primary; the epoch ends when env-0 is done.

    Returns (new_step_count, env-0's episode return, new obs_np)."""
    ep_return = 0.0
    for _ in range(MAX_STEPS):
        obs_t = obs_tensor(obs_np)  # [N, 4]
        epsilon = linear_epsilon(step_count)
        actions_np = eps_greedy_batched(q, obs_t, epsilon, rng)
        # SyncVectorEnv is unparameterized upstream; narrow its step() products.
        next_obs_np, rewards_np, terms_np, truncs_np, _ = cast(
            "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]",
            vec_env.step(actions_np),
        )
        next_obs_np = next_obs_np.astype(np.float64)
        dones_np = np.logical_or(terms_np, truncs_np)
        for i in range(NUM_ENVS):
            buffer.push(
                obs_np[i].tolist(),
                int(actions_np[i]),
                float(rewards_np[i]),
                next_obs_np[i].tolist(),
                bool(dones_np[i]),
            )
        ep_return += float(rewards_np[0])
        # SyncVectorEnv auto-resets a terminated sub-env itself, and the obs it
        # returns is the restarted state. Until 2026-08-01 this loop overwrote
        # that with zeros to match the Idris side's pinned reset, which left the
        # policy acting on obs = 0 for one step while the env sat elsewhere.
        # Both sides now reset through the env's own distribution.
        obs_np = next_obs_np
        step_count += 1

        if len(buffer) >= batch_size:
            double_dqn_update(q, target, optimizer, buffer, batch_size, gamma, rng)

        if step_count % target_sync_every == 0:
            target.load_state_dict(q.state_dict())

        if dones_np[0]:
            break
    return step_count, ep_return, obs_np


def train_double_dqn(
    episodes: int = 300,
    lr: float = 5e-4,
    gamma: float = 0.99,
    batch_size: int = 64,
    buffer_capacity: int = 10000,
    target_sync_every: int = 100,
    seed: int = 42,
    log_every: int = 50,
) -> tuple[QNetwork, list[float]]:
    """Batched Double DQN. Identical defaults to `dqn.train_dqn`."""
    torch.manual_seed(seed)  # pyright: ignore[reportUnknownMemberType] - untyped `seed` param in torch stubs
    rng = random.Random(seed)
    q = QNetwork()
    target = copy.deepcopy(q)
    optimizer = torch.optim.Adam(q.parameters(), lr=lr)
    buffer = ReplayBuffer(buffer_capacity)
    vec_env = make_cartpole_vec_env(seed, NUM_ENVS)
    obs_np = np.zeros((NUM_ENVS, 4), dtype=np.float64)
    history: list[float] = []
    step_count = 0
    t_start = time.monotonic()
    for ep in range(episodes):
        step_count, ep_return, obs_np = double_dqn_episode_batched(
            vec_env,
            obs_np,
            q,
            target,
            optimizer,
            buffer,
            step_count,
            batch_size,
            gamma,
            target_sync_every,
            rng,
        )
        history.append(ep_return)
        if (ep + 1) % log_every == 0:
            recent = sum(history[-50:]) / min(len(history), 50)
            print(
                f"  {format_elapsed(t_start)} {ep + 1}\tloss={-ep_return:.6f}"
                f"{mem_suffix()}\treturn={ep_return:.1f}\trecent_50={recent:.1f}"
            )
    return q, history


if __name__ == "__main__":
    print("=== Double DQN on CartPole ===")
    q, history = train_double_dqn()
    avg = evaluate(q)
    print(f"\nEval (50 episodes, greedy): avg_return={avg:.1f}")
    print(f"RESULT\tavg_return={avg:.1f}\tepisodes={len(history)}\tseed=42")
