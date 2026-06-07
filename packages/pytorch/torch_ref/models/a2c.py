"""A2C (synchronous Advantage Actor-Critic) on CartPole-v1.

Aligned with `Example.A2c` (Idris). Separate actor and critic networks
(Idris uses distinct paramId prefixes via `prefixParamId` + `emap` to
register them in the same optimizer without name collisions).

Batched rollouts: `NUM_ENVS` independent CartPole envs stepped in
lockstep via `gym.vector.SyncVectorEnv`, mirroring Idris'
`Gym.Vector.VecEnv NumEnvs CPState`. One batched (actor, critic) forward
per timestep amortises per-op overhead across the N samples. Both sides
use the same N to keep the cross-backend reference comparable.

Reset state pinned to (0, 0, 0, 0) to mirror idris-gym's
`Gym.ClassicControl.CartPole.reset`.
"""

from __future__ import annotations

import time
from typing import Any, cast

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_ref.models.reinforce import (
    MAX_STEPS,
    CartPoleEnv,
    make_cartpole_env,
    obs_tensor,
    reset_to_zero,
)
from torch_ref.training.runner import (
    format_elapsed,
    get_device,
    get_dtype,
    mem_suffix,
    multinomial_safe,
)

# Number of parallel envs run per a2c_update. Matches Idris-side
# `Example.A2c.NumEnvs`. Compile-time on Idris (shape baked into the
# autograd graph); module-level on Python (any-arg-compatible).
NUM_ENVS = 4


class Actor(nn.Module):
    def __init__(self, obs_dim: int = 4, num_actions: int = 2, hidden: int = 64) -> None:
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden, dtype=get_dtype())
        self.fc2 = nn.Linear(hidden, hidden, dtype=get_dtype())
        self.head = nn.Linear(hidden, num_actions, dtype=get_dtype())

    def forward(self, x: Tensor) -> Tensor:
        h = torch.tanh(self.fc2(torch.tanh(self.fc1(x))))
        return self.head(h)


class Critic(nn.Module):
    def __init__(self, obs_dim: int = 4, hidden: int = 64) -> None:
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden, dtype=get_dtype())
        self.fc2 = nn.Linear(hidden, hidden, dtype=get_dtype())
        self.head = nn.Linear(hidden, 1, dtype=get_dtype())

    def forward(self, x: Tensor) -> Tensor:
        h = torch.tanh(self.fc2(torch.tanh(self.fc1(x))))
        return self.head(h).squeeze(-1)


def make_cartpole_vec_env(seed: int, num_envs: int) -> gym.vector.SyncVectorEnv:
    """N independent CartPole envs in a SyncVectorEnv, each reset to all-zero
    to mirror idris-gym's `Gym.ClassicControl.CartPole.reset`."""

    def _make(idx: int):
        def _f():
            return make_cartpole_env(seed + idx)

        return _f

    vec = gym.vector.SyncVectorEnv([_make(i) for i in range(num_envs)])
    vec.reset()
    # SyncVectorEnv.envs is list[Env[Unknown, Unknown]] in gymnasium's stubs.
    for sub in cast("list[CartPoleEnv]", vec.envs):  # pyright: ignore[reportUnknownMemberType]
        reset_to_zero(sub)
    return vec


def collect_rollout(
    actor: Actor,
    critic: Critic,
    vec_env: gym.vector.SyncVectorEnv,
    obs_np: np.ndarray,
    rollout_len: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, np.ndarray]:
    """Batched rollout of `rollout_len` steps across NUM_ENVS parallel envs
    with auto-reset on done (each terminated sub-env resets to all-zero).
    Returned tensors are shaped [T, N, ...]; new_obs is [N, 4]. Mirrors
    Idris-side `rolloutBatched`."""
    obs_list: list[Tensor] = []
    act_list: list[Tensor] = []
    rew_list: list[Tensor] = []
    val_list: list[Tensor] = []
    done_list: list[Tensor] = []
    n = NUM_ENVS
    for _ in range(rollout_len):
        obs_t = obs_tensor(obs_np)  # [N, 4]
        with torch.no_grad():
            logits = actor(obs_t)  # [N, 2]
            values = critic(obs_t)  # [N]
        probs = F.softmax(logits, dim=-1)
        actions_t = multinomial_safe(probs, 1).squeeze(-1)  # [N]
        actions_np = actions_t.cpu().numpy().astype(np.int64)
        # SyncVectorEnv.step's stub returns unsolved TypeVars (Unknown).
        next_obs_np, rewards_np, terms_np, truncs_np, _ = cast(
            "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]",
            vec_env.step(actions_np),
        )
        dones_np = np.logical_or(terms_np, truncs_np)
        device, dtype = get_device(), get_dtype()
        obs_list.append(obs_t)
        act_list.append(actions_t.long())
        rew_list.append(torch.tensor(rewards_np, dtype=dtype, device=device))
        val_list.append(values.detach().to(dtype))
        done_list.append(torch.tensor(dones_np.astype(np.float64), dtype=dtype, device=device))
        # Auto-reset: SyncVectorEnv auto-resets on its own but to a
        # uniform-random state. Override each terminated sub-env back to
        # all-zero so the Idris-side reset semantics match.
        next_obs_np = next_obs_np.astype(np.float64)
        for i in range(n):
            if dones_np[i]:
                reset_to_zero(cast("CartPoleEnv", vec_env.envs[i]))  # pyright: ignore[reportUnknownMemberType]
                next_obs_np[i] = 0.0
        obs_np = next_obs_np
    return (
        torch.stack(obs_list),  # [T, N, 4]
        torch.stack(act_list),  # [T, N]
        torch.stack(rew_list),  # [T, N]
        torch.stack(val_list),  # [T, N]
        torch.stack(done_list),  # [T, N]
        obs_np,  # [N, 4]
    )


def compute_advantages(
    rewards: Tensor,
    values: Tensor,
    dones: Tensor,
    bootstraps: Tensor,
    gamma: float,
    lam: float,
) -> tuple[Tensor, Tensor]:
    """Per-env GAE on batched inputs. rewards/values/dones are [T, N];
    bootstraps is [N]. Returns advantages [T, N], returns [T, N]. Matches
    Idris-side `buildLossBatched`'s per-env GAE chain."""
    t_len, n = rewards.shape
    advantages = torch.zeros_like(rewards)
    for env_idx in range(n):
        gae_val = 0.0
        v_next = float(bootstraps[env_idx].item())
        for t in reversed(range(t_len)):
            mask = 1.0 - float(dones[t, env_idx].item())
            v_t = float(values[t, env_idx].item())
            delta = float(rewards[t, env_idx].item()) + gamma * v_next * mask - v_t
            gae_val = delta + gamma * lam * mask * gae_val
            advantages[t, env_idx] = gae_val
            v_next = float(values[t, env_idx].item())
    returns = advantages + values
    return advantages, returns


def a2c_update(
    actor: Actor,
    critic: Critic,
    optimizer: torch.optim.Optimizer,
    obs: Tensor,
    actions: Tensor,
    advantages: Tensor,
    returns: Tensor,
    entropy_coef: float,
    value_coef: float,
) -> float:
    logits = actor(obs)
    values = critic(obs)
    log_probs = F.log_softmax(logits, dim=-1).gather(1, actions.unsqueeze(1)).squeeze(1)
    probs = F.softmax(logits, dim=-1)
    entropy = -(probs * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
    adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    policy_loss = -(log_probs * adv).mean()
    value_loss = F.mse_loss(values, returns)
    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
    optimizer.zero_grad()
    # torch stub: Tensor.backward's params are unannotated.
    loss.backward()  # pyright: ignore[reportUnknownMemberType]
    torch.nn.utils.clip_grad_norm_(
        list(actor.parameters()) + list(critic.parameters()),
        0.5,
    )
    optimizer.step()
    return float(loss.item())


def train_a2c(
    total_updates: int = 5000,
    rollout_len: int = 20,
    lr: float = 7e-4,
    gamma: float = 0.99,
    lam: float = 0.95,
    entropy_coef: float = 0.01,
    value_coef: float = 0.5,
    seed: int = 42,
    log_every: int = 500,
) -> tuple[Actor, Critic, list[float]]:
    """Hyperparameters match Idris `Example.A2c.defaultConfig`:
    lr=7e-4, entropy=0.01, rollout=20, gamma=0.99, lam=0.95. Batched
    across NUM_ENVS parallel envs (mirrors Idris-side NumEnvs)."""
    # torch stub: manual_seed's seed param is unannotated.
    torch.manual_seed(seed)  # pyright: ignore[reportUnknownMemberType]
    actor = Actor().to(get_device())
    critic = Critic().to(get_device())
    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()),
        lr=lr,
    )
    vec_env = make_cartpole_vec_env(seed, NUM_ENVS)
    obs_np = np.zeros((NUM_ENVS, 4), dtype=np.float64)
    history: list[float] = []
    ep_returns_running = np.zeros(NUM_ENVS, dtype=np.float64)
    t_start = time.monotonic()
    for update in range(total_updates):
        obs, actions, rewards, values, dones, obs_np = collect_rollout(
            actor, critic, vec_env, obs_np, rollout_len
        )
        # Per-env bootstrap: critic value at the post-rollout state,
        # zeroed for any env whose last step terminated.
        with torch.no_grad():
            bootstrap_v = critic(obs_tensor(obs_np))  # [N]
            last_done = dones[-1]  # [N]
            bootstraps = torch.where(
                last_done > 0.5,
                torch.zeros_like(bootstrap_v),
                bootstrap_v,
            )
        advantages, returns = compute_advantages(
            rewards, values, dones, bootstraps, gamma, lam
        )  # both [T, N]
        # Flatten to [T*N, ...] for the update step.
        a2c_update(
            actor,
            critic,
            optimizer,
            obs.reshape(-1, 4),
            actions.reshape(-1),
            advantages.reshape(-1),
            returns.reshape(-1),
            entropy_coef,
            value_coef,
        )
        # Per-env episodic return tracking.
        rewards_np = rewards.cpu().numpy()
        dones_np = dones.cpu().numpy()
        for t in range(rollout_len):
            for env_idx in range(NUM_ENVS):
                ep_returns_running[env_idx] += float(rewards_np[t, env_idx])
                if dones_np[t, env_idx] > 0.5:
                    history.append(float(ep_returns_running[env_idx]))
                    ep_returns_running[env_idx] = 0.0
        if (update + 1) % log_every == 0:
            recent = history[-50:] or [0.0]
            last_ep = history[-1] if history else 0.0
            print(
                f"  {format_elapsed(t_start)} {update + 1}\tloss={-last_ep:.6f}"
                f"{mem_suffix()}\treturn={last_ep:.1f}"
                f"\trecent_50={sum(recent) / len(recent):.1f}"
            )
    return actor, critic, history


def evaluate(actor: Actor, n_episodes: int = 50) -> float:
    env = make_cartpole_env(seed=0)
    total = 0.0
    for _ in range(n_episodes):
        env.reset()
        obs_np = reset_to_zero(env)
        ep_return = 0.0
        for _ in range(MAX_STEPS):
            obs = obs_tensor(obs_np)
            with torch.no_grad():
                logits = actor(obs)
            action = int(torch.argmax(logits, dim=-1).item())
            next_obs_np, reward, term, trunc, _ = env.step(action)
            ep_return += float(reward)
            if term or trunc:
                break
            obs_np = next_obs_np.astype(np.float64)
        total += ep_return
    return total / n_episodes


if __name__ == "__main__":
    print("=== A2C on CartPole (separate actor + critic) ===")
    actor, _critic, history = train_a2c()
    avg = evaluate(actor)
    print(f"\nEval (50 episodes, greedy): avg_return={avg:.1f}")
    print(f"RESULT\tavg_return={avg:.1f}\tupdates={5000}\tseed=42")
