"""SAC (Haarnoja et al. 2018) on Pendulum-v1.

Stochastic tanh-squashed Gaussian actor, twin Q-networks, Polyak-averaged
target Q-networks, fixed entropy temperature α. Uses canonical
`gym.make("Pendulum-v1")` for env physics. Both Idris
(`Gym.ClassicControl.Pendulum.reset`) and the PyTorch reference
randomize the initial state per Gymnasium θ ~ U(-π, π), θ̇ ~ U(-1, 1) —
seeded once at trainer start, advanced per episode.

Aligned with `Example.Sac` (Idris): separate actor + Q1 + Q2 networks
registered under distinct paramId scope prefixes on the Idris side, and
three separate Adam optimizers (one per network). Polyak soft target
update τ=0.005 applied every step, matching the Idris `polyakBlend`
wrapper that calls the C-backend `polyak_blend` FFI.
"""

from __future__ import annotations

import copy
import math
import random
import time
from collections import deque
from typing import Any, cast

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_ref.init_manifest import maybe_dump_init
from torch_ref.init import init_linear_
from torch_ref.training.runner import format_elapsed, get_device, get_dtype, mem_suffix

MAX_ACTION = 2.0  # Pendulum torque range
MAX_STEPS = 200  # gymnasium Pendulum-v1 default TimeLimit

# Parallel envs collecting transitions in lockstep. Matches Idris-side
# `Example.Sac.NumEnvs`.
NUM_ENVS = 4

# Pendulum-v1: Box observations and Box actions (both np.ndarray).
# gymnasium's `gym.make` stub returns `Env[Unknown, Unknown]`, so call
# sites pin this alias for pyright strict.
PendulumEnv = gym.Env[np.ndarray, np.ndarray]


def _reset_to_pi(env: PendulumEnv) -> np.ndarray:
    """Return obs [cos(theta), sin(theta), theta_dot] of the env's current
    (just-reset) state as float64.

    Previously pinned state to (theta=π, theta_dot=0) to match idris-gym's
    deterministic reset; idris-gym now randomizes per Gymnasium (theta ~
    U(-π, π), theta_dot ~ U(-1, 1)) and the PyTorch side follows.
    """
    # `state` isn't on the Env stub (Pendulum-specific 2-float array).
    th, dth = cast(
        "tuple[float, float]",
        env.unwrapped.state,  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
    )
    return np.array([math.cos(th), math.sin(th), dth], dtype=np.float64)


def _obs_tensor(obs: np.ndarray) -> Tensor:
    return torch.tensor(obs, dtype=get_dtype(), device=get_device())


def make_pendulum_vec_env(seed: int, num_envs: int) -> tuple[gym.vector.SyncVectorEnv, np.ndarray]:
    """N Pendulum-v1 envs in a SyncVectorEnv, seeded once at construction
    and randomized per Gymnasium on each reset."""

    def _make(idx: int):
        def _f() -> PendulumEnv:
            return cast("PendulumEnv", gym.make("Pendulum-v1"))  # pyright: ignore[reportUnknownMemberType]

        return _f

    # Same-step autoreset, matching idris-gym's `Gym.Vector.stepAutoReset`
    # (gymnasium 1.x defaults to NEXT_STEP, which inserts a filler transition
    # whose action is ignored and whose reward is 0).
    vec = gym.vector.SyncVectorEnv(
        [_make(i) for i in range(num_envs)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    # SyncVectorEnv.reset's stub returns unsolved TypeVars (Unknown).
    obs0, _info = cast("tuple[np.ndarray, dict[str, Any]]", vec.reset(seed=seed))
    return vec, np.asarray(obs0, dtype=np.float64)


# ---------------------------------------------------------------------------
# Actor: tanh-squashed Gaussian policy
# ---------------------------------------------------------------------------


class Actor(nn.Module):
    def __init__(self, obs_dim: int = 3, hidden: int = 64) -> None:
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden, dtype=get_dtype())
        self.fc2 = nn.Linear(hidden, hidden, dtype=get_dtype())
        self.mean_head = nn.Linear(hidden, 1, dtype=get_dtype())
        # 0-dim, matching the Idris side's `tparamScalar "actor_log_std"`.
        # A shape-(1,) parameter broadcasts the same but reads as a different
        # parameter to the init-manifest gate.
        self.log_std = nn.Parameter(torch.zeros((), dtype=get_dtype()))
        init_linear_(self)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = F.relu(self.fc2(F.relu(self.fc1(x))))
        mean = self.mean_head(h).squeeze(-1)
        log_std = torch.clamp(self.log_std + torch.zeros_like(mean), min=-5.0, max=2.0)
        return mean, log_std

    def sample(self, x: Tensor, rng: random.Random | None = None) -> tuple[Tensor, Tensor]:
        """Reparameterized sample: a = tanh(mean + std * eps) * MAX_ACTION.

        Returns (action, log_prob), both with gradient flow through the actor
        when x requires grad (via reparameterization trick).
        """
        mean, log_std = self(x)
        std = torch.exp(log_std)
        if rng is None:
            eps = torch.randn_like(mean)
        else:
            eps = torch.tensor(rng.gauss(0.0, 1.0), dtype=get_dtype(), device=get_device())
        u = mean + std * eps  # pre-tanh
        a_squashed = torch.tanh(u)
        action = a_squashed * MAX_ACTION
        # Gaussian log-prob of u, corrected for tanh squash + action scaling:
        #   log_prob = gaussian_log_prob(u) - log(1 - tanh(u)^2 + ε) - log(MAX_ACTION)
        log_prob_u = -0.5 * ((u - mean) / std) ** 2 - log_std - 0.5 * math.log(2 * math.pi)
        log_prob = log_prob_u - torch.log(1.0 - a_squashed**2 + 1e-6) - math.log(MAX_ACTION)
        return action, log_prob


# ---------------------------------------------------------------------------
# Q-networks: (obs, action) → scalar
# ---------------------------------------------------------------------------


class QNet(nn.Module):
    def __init__(self, obs_dim: int = 3, act_dim: int = 1, hidden: int = 64) -> None:
        super().__init__()
        self.fc1 = nn.Linear(obs_dim + act_dim, hidden, dtype=get_dtype())
        self.fc2 = nn.Linear(hidden, hidden, dtype=get_dtype())
        self.head = nn.Linear(hidden, 1, dtype=get_dtype())
        init_linear_(self)

    def forward(self, obs: Tensor, action: Tensor) -> Tensor:
        a = action.unsqueeze(-1) if action.dim() == obs.dim() - 1 else action
        x = torch.cat([obs, a], dim=-1)
        h = F.relu(self.fc2(F.relu(self.fc1(x))))
        return self.head(h).squeeze(-1)


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buf: deque[tuple[list[float], float, float, list[float], bool]] = deque(
            maxlen=capacity
        )

    def push(self, obs: list[float], a: float, r: float, next_obs: list[float], done: bool) -> None:
        self.buf.append((obs, a, r, next_obs, done))

    def sample(self, n: int, rng: random.Random) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        # Uniform iid draws WITH replacement (Mnih-classic; cleanrl's buffers
        # do the same), matching Idris `Ml.RL.ReplayBuffer.sampleN`. Was
        # `rng.sample` (without replacement) until 2026-08-03; see
        # reference-alignment.md.
        batch = [self.buf[rng.randrange(len(self.buf))] for _ in range(n)]
        device, dtype = get_device(), get_dtype()
        obs = torch.tensor([b[0] for b in batch], dtype=dtype, device=device)
        actions = torch.tensor([b[1] for b in batch], dtype=dtype, device=device)
        rewards = torch.tensor([b[2] for b in batch], dtype=dtype, device=device)
        next_obs = torch.tensor([b[3] for b in batch], dtype=dtype, device=device)
        dones = torch.tensor([float(b[4]) for b in batch], dtype=dtype, device=device)
        return obs, actions, rewards, next_obs, dones

    def __len__(self) -> int:
        return len(self.buf)


# ---------------------------------------------------------------------------
# SAC update
# ---------------------------------------------------------------------------


def sac_update(
    actor: Actor,
    q1: QNet,
    q2: QNet,
    q1_target: QNet,
    q2_target: QNet,
    actor_opt: torch.optim.Optimizer,
    q1_opt: torch.optim.Optimizer,
    q2_opt: torch.optim.Optimizer,
    buffer: ReplayBuffer,
    batch_size: int,
    gamma: float,
    alpha: float,
    rng: random.Random,
) -> float:
    obs, actions, rewards, next_obs, dones = buffer.sample(batch_size, rng)
    with torch.no_grad():
        next_action, next_logp = actor.sample(next_obs)
        # nn.Module.__call__ is untyped in torch's stubs, so torch.min's
        # overload resolution yields Unknown — pin the Tensor result.
        target_q = cast(
            "Tensor",
            torch.min(q1_target(next_obs, next_action), q2_target(next_obs, next_action)),
        )
        target = rewards + gamma * (1.0 - dones) * (target_q - alpha * next_logp)

    # Q losses (Bellman MSE)
    q1_loss = F.mse_loss(q1(obs, actions), target)
    q2_loss = F.mse_loss(q2(obs, actions), target)
    q1_opt.zero_grad()
    # torch stub: Tensor.backward's params are unannotated.
    q1_loss.backward()  # pyright: ignore[reportUnknownMemberType]
    q1_opt.step()
    q2_opt.zero_grad()
    q2_loss.backward()  # pyright: ignore[reportUnknownMemberType]
    q2_opt.step()

    # Actor loss: E[α * log π(a|s) - min(Q1(s,a), Q2(s,a))]
    sampled_action, logp = actor.sample(obs)
    q_min = cast("Tensor", torch.min(q1(obs, sampled_action), q2(obs, sampled_action)))
    actor_loss = (alpha * logp - q_min).mean()
    actor_opt.zero_grad()
    actor_loss.backward()  # pyright: ignore[reportUnknownMemberType]
    actor_opt.step()

    return float(actor_loss.item())


def polyak_update(target: nn.Module, online: nn.Module, tau: float) -> None:
    """target ← (1-τ)·target + τ·online, in-place. Matches the Idris
    backend's `polyak_blend` FFI semantics."""
    with torch.no_grad():
        for t, o in zip(target.parameters(), online.parameters(), strict=True):
            t.mul_(1.0 - tau).add_(o, alpha=tau)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_sac(
    total_steps: int = 30000,
    buffer_capacity: int = 100000,
    batch_size: int = 64,
    lr: float = 3e-4,
    gamma: float = 0.99,
    alpha: float = 0.2,
    warmup_steps: int = 1000,
    tau: float = 0.005,
    seed: int = 42,
    log_every: int = 2000,
) -> tuple[Actor, list[float]]:
    """Polyak soft update τ=0.005 every step, matching the Idris port which
    calls `polyakBlend` after each gradient step."""
    # torch stub: manual_seed's seed param is unannotated.
    torch.manual_seed(seed)  # pyright: ignore[reportUnknownMemberType]
    rng = random.Random(seed)
    actor = Actor().to(get_device())
    q1 = QNet().to(get_device())
    q2 = QNet().to(get_device())
    q1_target = copy.deepcopy(q1)
    q2_target = copy.deepcopy(q2)
    # Construction lives here rather than in scripts/sac.py, so the dump sees
    # the same objects training uses. Inert unless IDRISML_DUMP_INIT is set.
    maybe_dump_init(actor, q1, q2, q1_target, q2_target)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=lr)
    q1_opt = torch.optim.Adam(q1.parameters(), lr=lr)
    q2_opt = torch.optim.Adam(q2.parameters(), lr=lr)
    buffer = ReplayBuffer(buffer_capacity)

    vec_env, obs_np = make_pendulum_vec_env(seed, NUM_ENVS)
    ep_lens = np.zeros(NUM_ENVS, dtype=np.int64)
    ep_returns_running = np.zeros(NUM_ENVS, dtype=np.float64)
    history: list[float] = []
    t_start = time.monotonic()
    for step in range(total_steps):
        if step < warmup_steps:
            actions_np = np.array(
                [rng.uniform(-MAX_ACTION, MAX_ACTION) for _ in range(NUM_ENVS)],
                dtype=np.float64,
            )
        else:
            obs_t = _obs_tensor(obs_np)
            with torch.no_grad():
                a_t, _ = actor.sample(obs_t)
                actions_np = a_t.cpu().numpy().astype(np.float64)
        # SyncVectorEnv.step's stub returns unsolved TypeVars (Unknown).
        next_obs_np, rewards_np, _terms, _truncs, _ = cast(
            "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]",
            vec_env.step(actions_np.astype(np.float32).reshape(NUM_ENVS, 1)),
        )
        next_obs_np = next_obs_np.astype(np.float64)
        ep_lens += 1
        is_dones = ep_lens >= MAX_STEPS  # Pendulum truncates; no termination
        ep_returns_running += rewards_np.astype(np.float64)
        for i in range(NUM_ENVS):
            buffer.push(
                obs_np[i].tolist(),
                float(actions_np[i]),
                float(rewards_np[i]),
                next_obs_np[i].tolist(),
                bool(is_dones[i]),
            )
        for i in range(NUM_ENVS):
            if is_dones[i]:
                history.append(float(ep_returns_running[i]))
                ep_returns_running[i] = 0.0
                ep_lens[i] = 0
                # SyncVectorEnv auto-resets terminated sub-envs; the
                # randomized initial obs is already in next_obs_np[i].
        obs_np = next_obs_np
        if len(buffer) >= max(batch_size, warmup_steps):
            sac_update(
                actor,
                q1,
                q2,
                q1_target,
                q2_target,
                actor_opt,
                q1_opt,
                q2_opt,
                buffer,
                batch_size,
                gamma,
                alpha,
                rng,
            )
            # Polyak soft update every step (matches Idris).
            polyak_update(q1_target, q1, tau)
            polyak_update(q2_target, q2, tau)
        if (step + 1) % log_every == 0:
            recent = history[-20:] or [0.0]
            last_ep = history[-1] if history else 0.0
            print(
                f"  {format_elapsed(t_start)} {step + 1}\tloss={-last_ep:.6f}"
                f"{mem_suffix()}\treturn={last_ep:.1f}"
                f"\trecent_20={sum(recent) / len(recent):.1f}"
            )
    return actor, history


def evaluate(actor: Actor, n_episodes: int = 20) -> float:
    env = cast("PendulumEnv", gym.make("Pendulum-v1"))  # pyright: ignore[reportUnknownMemberType]
    env.reset(seed=0)
    total = 0.0
    for _ in range(n_episodes):
        env.reset()
        obs_np = _reset_to_pi(env)
        ep_return = 0.0
        for _ in range(MAX_STEPS):
            obs = _obs_tensor(obs_np)
            with torch.no_grad():
                mean, _ = actor(obs)
            action = float(torch.tanh(mean).item()) * MAX_ACTION
            obs_np, reward, term, trunc, _ = env.step(np.array([action], dtype=np.float32))
            obs_np = obs_np.astype(np.float64)
            ep_return += float(reward)
            if term or trunc:
                break
        total += ep_return
    return total / n_episodes
