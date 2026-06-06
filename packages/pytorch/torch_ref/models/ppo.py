"""PPO (Schulman et al. 2017) on Acrobot-v1 (discrete-action, categorical policy).

Uses canonical `gym.make("Acrobot-v1")` for env physics (single-step
RK4 with dt=0.2). Both Idris (`Gym.ClassicControl.Acrobot.reset`) and
the PyTorch reference randomize the 4-component initial state per
Gymnasium U(-0.1, 0.1) — seeded once at trainer start, advanced per
episode.

Acrobot is the canonical "PPO clipped-surrogate demonstrates"
benchmark — discrete actions (3: -1/0/+1 torque), sparse reward
(-1/step, 0 at goal), longer horizon, 500-step TimeLimit.

Batched rollouts: NUM_ENVS independent Acrobot envs stepped in lockstep
via `gym.vector.SyncVectorEnv`, mirroring Idris'
`Gym.Vector.VecEnv NumEnvs AState`. One batched (actor, critic) forward
per timestep amortises per-op overhead across the N samples. Both sides
use the same N to keep the cross-backend reference comparable.
"""

from __future__ import annotations

import math
import random
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_ref.training.runner import format_elapsed, get_device, get_dtype, mem_suffix

MAX_STEPS = 500  # gymnasium Acrobot-v1 default TimeLimit

# Number of parallel envs run per ppo_update. Matches Idris-side
# `Example.Ppo.NumEnvs`. Compile-time on Idris; module-level here.
NUM_ENVS = 4


def make_acrobot_env(seed: int) -> gym.Env:
    """Create an Acrobot-v1 env seeded once at construction. Per-episode
    resets advance the env's PRNG and randomize the 4 state components
    ~ U(-0.1, 0.1) per Gymnasium, matching idris-gym's randomized
    `Env.reset`."""
    env = gym.make("Acrobot-v1")
    env.reset(seed=seed)
    return env


def reset_to_zero(env: gym.Env) -> np.ndarray:
    """Return obs [cos(th1), sin(th1), cos(th2), sin(th2), dth1, dth2]
    derived from the env's current (just-reset) state, as float64.

    Previously pinned state to (0, 0, 0, 0); idris-gym now randomizes
    per Gymnasium and the PyTorch side follows.
    """
    th1, th2, dth1, dth2 = env.unwrapped.state  # pyright: ignore[reportAttributeAccessIssue]
    return np.array(
        [math.cos(th1), math.sin(th1), math.cos(th2), math.sin(th2), dth1, dth2],
        dtype=np.float64,
    )


def obs_tensor(obs: np.ndarray) -> Tensor:
    return torch.tensor(obs, dtype=get_dtype(), device=get_device())


class Actor(nn.Module):
    """Categorical policy over 3 actions (-1, 0, +1 torque)."""

    def __init__(self, obs_dim: int = 6, hidden: int = 64, num_actions: int = 3) -> None:
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden, dtype=get_dtype())
        self.fc2 = nn.Linear(hidden, hidden, dtype=get_dtype())
        self.head = nn.Linear(hidden, num_actions, dtype=get_dtype())

    def forward(self, x: Tensor) -> Tensor:
        h = torch.tanh(self.fc2(torch.tanh(self.fc1(x))))
        return self.head(h)


class Critic(nn.Module):
    def __init__(self, obs_dim: int = 6, hidden: int = 64) -> None:
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden, dtype=get_dtype())
        self.fc2 = nn.Linear(hidden, hidden, dtype=get_dtype())
        self.head = nn.Linear(hidden, 1, dtype=get_dtype())

    def forward(self, x: Tensor) -> Tensor:
        h = torch.tanh(self.fc2(torch.tanh(self.fc1(x))))
        return self.head(h).squeeze(-1)


def sample_action(actor: Actor, obs: Tensor, rng: random.Random) -> tuple[int, float]:
    """Sample a discrete action from the categorical policy. Returns (action, log π(a|s))."""
    with torch.no_grad():
        logits = actor(obs)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        u = rng.random()
        cum = 0.0
        a = 0
        for i, p in enumerate(probs.tolist()):
            cum += p
            if u <= cum:
                a = i
                break
        else:
            a = probs.shape[0] - 1
        lp = float(log_probs[a].item())
    return a, lp


def gae(
    rewards: list[float],
    values: list[float],
    dones: list[bool],
    bootstrap: float,
    gamma: float,
    lam: float,
) -> tuple[list[float], list[float]]:
    advantages: list[float] = []
    a_next = 0.0
    v_next = bootstrap
    for t in reversed(range(len(rewards))):
        mask = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * v_next * mask - values[t]
        a = delta + gamma * lam * mask * a_next
        advantages.append(a)
        a_next = a
        v_next = values[t]
    advantages.reverse()
    returns = [adv + v for adv, v in zip(advantages, values, strict=True)]
    return advantages, returns


def gae_batched(
    rewards: Tensor,
    values: Tensor,
    dones: Tensor,
    bootstraps: Tensor,
    gamma: float,
    lam: float,
) -> tuple[Tensor, Tensor]:
    """Per-env GAE on batched [T, N] inputs. bootstraps is [N]. Returns
    advantages [T, N], returns [T, N]. Each env's GAE chain is independent
    — matches Idris-side `prepareRolloutBatched`."""
    t_len, n = rewards.shape
    advantages = torch.zeros_like(rewards)
    for env_idx in range(n):
        a_next = 0.0
        v_next = float(bootstraps[env_idx].item())
        for t in reversed(range(t_len)):
            mask = 1.0 - float(dones[t, env_idx].item())
            delta = (
                float(rewards[t, env_idx].item())
                + gamma * v_next * mask
                - float(values[t, env_idx].item())
            )
            a = delta + gamma * lam * mask * a_next
            advantages[t, env_idx] = a
            a_next = a
            v_next = float(values[t, env_idx].item())
    returns = advantages + values
    return advantages, returns


def make_acrobot_vec_env(seed: int, num_envs: int) -> gym.vector.SyncVectorEnv:
    """N independent Acrobot envs in a SyncVectorEnv, seeded once at
    construction and randomized per Gymnasium on each reset (mirrors
    idris-gym's `Gym.Vector.resetAll`)."""
    def _make(idx: int):
        def _f():
            return make_acrobot_env(seed + idx)
        return _f
    vec = gym.vector.SyncVectorEnv([_make(i) for i in range(num_envs)])
    vec.reset()
    return vec


def collect_rollout(
    actor: Actor,
    critic: Critic,
    vec_env: gym.vector.SyncVectorEnv,
    obs_np: np.ndarray,
    ep_lens: np.ndarray,
    n_steps: int,
    max_ep_len: int,
    rng: random.Random,
) -> tuple[
    Tensor,           # obs [T, N, 6]
    Tensor,           # actions [T, N] long
    Tensor,           # old_log_probs [T, N]
    Tensor,           # rewards [T, N]
    Tensor,           # values [T, N]
    Tensor,           # dones [T, N]
    np.ndarray,       # new_obs [N, 6]
    np.ndarray,       # new_ep_lens [N]
    list[float],      # completed episode returns
]:
    """Batched rollout of `n_steps` steps across NUM_ENVS parallel envs with
    auto-reset on done OR per-env max_ep_len truncation. Returns `[T, N, ...]`
    tensors. Mirrors Idris-side `rolloutBatched` + per-env stepsLeft."""
    n = NUM_ENVS
    device, dtype = get_device(), get_dtype()
    obs_list: list[Tensor] = []
    act_list: list[Tensor] = []
    lp_list: list[Tensor] = []
    rew_list: list[Tensor] = []
    val_list: list[Tensor] = []
    done_list: list[Tensor] = []
    ep_returns: list[float] = []
    ep_sums = np.zeros(n, dtype=np.float64)
    for _ in range(n_steps):
        obs_t = obs_tensor(obs_np)  # [N, 6]
        with torch.no_grad():
            logits = actor(obs_t)            # [N, 3]
            log_probs_t = F.log_softmax(logits, dim=-1)
            probs_t = torch.exp(log_probs_t)
            values_t = critic(obs_t)          # [N]
        # Per-env categorical sampling with the shared rng (matches Idris'
        # one-randomRIO-per-env-per-step convention).
        actions_np = np.zeros(n, dtype=np.int64)
        lps_np = np.zeros(n, dtype=np.float64)
        probs_np = probs_t.cpu().numpy()
        log_probs_np = log_probs_t.cpu().numpy()
        for env_idx in range(n):
            u = rng.random()
            cum = 0.0
            a = probs_np.shape[1] - 1
            for j in range(probs_np.shape[1]):
                cum += float(probs_np[env_idx, j])
                if u <= cum:
                    a = j
                    break
            actions_np[env_idx] = a
            lps_np[env_idx] = float(log_probs_np[env_idx, a])
        # Step all envs.
        next_obs_np, rewards_np, terms_np, truncs_np, _ = vec_env.step(actions_np)
        ep_lens = ep_lens + 1
        gym_dones = np.logical_or(terms_np, truncs_np)
        len_trunc = ep_lens >= max_ep_len
        dones_np = np.logical_or(gym_dones, len_trunc)
        # Record per-step batch.
        obs_list.append(obs_t)
        act_list.append(torch.from_numpy(actions_np).to(device))
        lp_list.append(torch.from_numpy(lps_np.astype(np.float64)).to(device, dtype))
        rew_list.append(torch.from_numpy(rewards_np.astype(np.float64)).to(device, dtype))
        val_list.append(values_t.detach().to(dtype))
        done_list.append(torch.from_numpy(dones_np.astype(np.float64)).to(device, dtype))
        # Per-env episode accounting. Gymnasium's SyncVectorEnv auto-resets
        # any terminated/truncated sub-env and returns the new randomized
        # obs in next_obs_np[env_idx], so we don't need to reset manually.
        ep_sums = ep_sums + rewards_np.astype(np.float64)
        next_obs_np = next_obs_np.astype(np.float64)
        for env_idx in range(n):
            if dones_np[env_idx]:
                ep_returns.append(float(ep_sums[env_idx]))
                ep_sums[env_idx] = 0.0
                ep_lens[env_idx] = 0
        obs_np = next_obs_np
    return (
        torch.stack(obs_list),     # [T, N, 6]
        torch.stack(act_list),     # [T, N]
        torch.stack(lp_list),      # [T, N]
        torch.stack(rew_list),     # [T, N]
        torch.stack(val_list),     # [T, N]
        torch.stack(done_list),    # [T, N]
        obs_np,                    # [N, 6]
        ep_lens,                   # [N]
        ep_returns,
    )


def ppo_update(
    actor: Actor,
    critic: Critic,
    actor_opt: torch.optim.Optimizer,
    critic_opt: torch.optim.Optimizer,
    obs: Tensor,
    actions: Tensor,
    old_log_probs: Tensor,
    advantages: Tensor,
    returns: Tensor,
    clip_eps: float,
    entropy_coef: float,
    k_epochs: int,
    batch_size: int,
    rng: random.Random,
) -> None:
    n = obs.shape[0]
    indices = list(range(n))
    for _ in range(k_epochs):
        rng.shuffle(indices)
        for start in range(0, n, batch_size):
            batch = indices[start : start + batch_size]
            o_b = obs[batch]
            a_b = actions[batch]
            lp_old = old_log_probs[batch]
            adv_b = advantages[batch]
            ret_b = returns[batch]
            logits = actor(o_b)
            log_probs = F.log_softmax(logits, dim=-1)
            lp_new = log_probs.gather(1, a_b.unsqueeze(1)).squeeze(1)
            ratio = torch.exp(lp_new - lp_old)
            s1 = ratio * adv_b
            s2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_b
            policy_loss = -torch.min(s1, s2).mean()
            entropy = -(torch.exp(log_probs) * log_probs).sum(dim=-1).mean()
            actor_loss = policy_loss - entropy_coef * entropy
            actor_opt.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
            actor_opt.step()
            v_b = critic(o_b)
            critic_loss = F.mse_loss(v_b, ret_b)
            critic_opt.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
            critic_opt.step()


def train_ppo(
    total_rollouts: int = 100,
    rollout_steps: int = 256,
    lr: float = 3e-4,
    gamma: float = 0.99,
    lam: float = 0.95,
    clip_eps: float = 0.2,
    entropy_coef: float = 0.01,
    k_epochs: int = 10,
    batch_size: int = 64,
    max_ep_len: int = MAX_STEPS,
    seed: int = 42,
    log_every: int = 10,
) -> tuple[Actor, list[float]]:
    """Batched PPO. Total samples per update = rollout_steps * NUM_ENVS
    (pre-batched used rollout_steps=1024 with one env; default here is
    256 steps × 4 envs = 1024 total to keep the per-update sample budget
    constant, matching Idris-side `Example.Ppo.RolloutLen`)."""
    torch.manual_seed(seed)
    rng = random.Random(seed)
    actor = Actor().to(get_device())
    critic = Critic().to(get_device())
    actor_opt = torch.optim.Adam(actor.parameters(), lr=lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=lr)
    vec_env = make_acrobot_vec_env(seed, NUM_ENVS)
    obs_np = np.tile(
        np.array([1.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64), (NUM_ENVS, 1)
    )
    ep_lens = np.zeros(NUM_ENVS, dtype=np.int64)
    history: list[float] = []
    t_start = time.monotonic()
    for r in range(total_rollouts):
        obs_t, act_t, lp_t, rew_t, val_t, done_t, obs_np, ep_lens, ep_rets = collect_rollout(
            actor, critic, vec_env, obs_np, ep_lens, rollout_steps, max_ep_len, rng
        )
        # Per-env bootstrap: critic value at the post-rollout state,
        # zeroed for any env whose last step terminated.
        with torch.no_grad():
            bootstrap_v = critic(obs_tensor(obs_np))   # [N]
            last_done = done_t[-1]                      # [N]
            bootstraps = torch.where(
                last_done > 0.5,
                torch.zeros_like(bootstrap_v),
                bootstrap_v,
            )
        adv_t, ret_t = gae_batched(rew_t, val_t, done_t, bootstraps, gamma, lam)
        # Flatten [T, N, ...] → [T*N, ...] for the update step.
        flat_obs = obs_t.reshape(-1, 6)
        flat_act = act_t.reshape(-1).long()
        flat_lp = lp_t.reshape(-1)
        flat_adv = adv_t.reshape(-1)
        flat_ret = ret_t.reshape(-1)
        flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)
        ppo_update(
            actor,
            critic,
            actor_opt,
            critic_opt,
            flat_obs,
            flat_act,
            flat_lp,
            flat_adv,
            flat_ret,
            clip_eps,
            entropy_coef,
            k_epochs,
            batch_size,
            rng,
        )
        history.extend(ep_rets)
        if (r + 1) % log_every == 0:
            recent = history[-50:] or [0.0]
            last_ep = history[-1] if history else 0.0
            print(
                f"  {format_elapsed(t_start)} {r + 1}\tloss={-last_ep:.6f}"
                f"{mem_suffix()}\treturn={last_ep:.1f}"
                f"\trecent_50={sum(recent) / len(recent):.1f}"
            )
    return actor, history


def evaluate(actor: Actor, n_episodes: int = 20, max_ep_len: int = MAX_STEPS) -> float:
    env = make_acrobot_env(seed=0)
    total = 0.0
    for _ in range(n_episodes):
        env.reset()
        obs_np = reset_to_zero(env)
        ep_return = 0.0
        for _ in range(max_ep_len):
            obs = obs_tensor(obs_np)
            with torch.no_grad():
                logits = actor(obs)
            action = int(torch.argmax(logits).item())
            next_obs_np, reward, term, trunc, _ = env.step(action)
            ep_return += float(reward)
            if term or trunc:
                break
            obs_np = next_obs_np.astype(np.float64)
        total += ep_return
    return total / n_episodes


if __name__ == "__main__":
    print("=== PPO on Acrobot ===")
    actor, history = train_ppo()
    avg = evaluate(actor)
    print(f"\nEval (20 episodes, greedy): avg_return={avg:.1f}")
    print(f"RESULT\tavg_return={avg:.1f}\trollouts={100}\tseed=42")
