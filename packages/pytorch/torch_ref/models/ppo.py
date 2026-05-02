"""PPO (Schulman et al. 2017) on Acrobot-v1 (discrete-action, categorical policy).

Pendulum (continuous-action, Gaussian policy) was the original env but
didn't converge at CPU-feasible rollout sizes; Acrobot is the canonical
"PPO clipped-surrogate demonstrates" benchmark — discrete actions,
sparse reward, longer horizon.

Self-contained Acrobot physics matching `Gym.ClassicControl.Acrobot`
(semi-implicit Euler with 4 substeps of dt=0.05, NOT Gymnasium's RK4).
Trajectories diverge numerically from the Gymnasium reference but the
task and termination condition are identical.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Acrobot environment (matches Idris Gym.ClassicControl.Acrobot)
# ---------------------------------------------------------------------------

LINK_LEN_1 = 1.0
LINK_COM_1 = 0.5
LINK_COM_2 = 0.5
LINK_MASS_1 = 1.0
LINK_MASS_2 = 1.0
LINK_MOI = 1.0
MAX_VEL_1 = 4.0 * math.pi
MAX_VEL_2 = 9.0 * math.pi
GRAVITY = 9.8
DT = 0.05
MAX_STEPS = 500


@dataclass
class AcrobotState:
    th1: float = 0.0
    th2: float = 0.0
    dth1: float = 0.0
    dth2: float = 0.0


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)


def _wrap_angle(x: float) -> float:
    two_pi = 2.0 * math.pi
    return x - two_pi * math.floor((x + math.pi) / two_pi)


def _dsdt(torque: float, s: AcrobotState) -> tuple[float, float, float, float]:
    m1, m2 = LINK_MASS_1, LINK_MASS_2
    l1 = LINK_LEN_1
    lc1, lc2 = LINK_COM_1, LINK_COM_2
    i1, i2 = LINK_MOI, LINK_MOI
    th1, th2, dth1, dth2 = s.th1, s.th2, s.dth1, s.dth2
    cos_th2 = math.cos(th2)
    sin_th2 = math.sin(th2)
    d1 = (
        m1 * lc1 * lc1
        + m2 * (l1 * l1 + lc2 * lc2 + 2.0 * l1 * lc2 * cos_th2)
        + i1 + i2
    )
    d2 = m2 * (lc2 * lc2 + l1 * lc2 * cos_th2) + i2
    phi2 = m2 * lc2 * GRAVITY * math.cos(th1 + th2 - math.pi / 2.0)
    phi1 = (
        -(m2 * l1 * lc2 * dth2 * dth2 * sin_th2)
        - 2.0 * m2 * l1 * lc2 * dth2 * dth1 * sin_th2
        + (m1 * lc1 + m2 * l1) * GRAVITY * math.cos(th1 - math.pi / 2.0)
        + phi2
    )
    ddth2 = (
        torque + d2 / d1 * phi1
        - m2 * l1 * lc2 * dth1 * dth1 * sin_th2 - phi2
    ) / (m2 * lc2 * lc2 + i2 - d2 * d2 / d1)
    ddth1 = -((d2 * ddth2 + phi1) / d1)
    return dth1, dth2, ddth1, ddth2


def _euler_step(torque: float, s: AcrobotState) -> AcrobotState:
    _, _, ddth1, ddth2 = _dsdt(torque, s)
    dth1p = s.dth1 + DT * ddth1
    dth2p = s.dth2 + DT * ddth2
    th1p = s.th1 + DT * dth1p
    th2p = s.th2 + DT * dth2p
    return AcrobotState(th1p, th2p, dth1p, dth2p)


def acrobot_step(s: AcrobotState, action: int) -> tuple[float, AcrobotState, bool]:
    """One physics step. Action 0 = -1 torque, 1 = 0 torque, 2 = +1 torque."""
    torque = float(action) - 1.0
    s1 = _euler_step(torque, s)
    s2 = _euler_step(torque, s1)
    s3 = _euler_step(torque, s2)
    s4 = _euler_step(torque, s3)
    th1 = _wrap_angle(s4.th1)
    th2 = _wrap_angle(s4.th2)
    dth1 = _clamp(s4.dth1, -MAX_VEL_1, MAX_VEL_1)
    dth2 = _clamp(s4.dth2, -MAX_VEL_2, MAX_VEL_2)
    sp = AcrobotState(th1, th2, dth1, dth2)
    terminated = -math.cos(th1) - math.cos(th2 + th1) > 1.0
    reward = 0.0 if terminated else -1.0
    return reward, sp, terminated


def observe(s: AcrobotState) -> Tensor:
    return torch.tensor(
        [math.cos(s.th1), math.sin(s.th1),
         math.cos(s.th2), math.sin(s.th2),
         s.dth1, s.dth2],
        dtype=torch.float64,
    )


# ---------------------------------------------------------------------------
# Actor-critic (separate nets, categorical policy)
# ---------------------------------------------------------------------------


class Actor(nn.Module):
    """Categorical policy over 3 actions (-1, 0, +1 torque)."""

    def __init__(self, obs_dim: int = 6, hidden: int = 64, num_actions: int = 3) -> None:
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden, dtype=torch.float64)
        self.fc2 = nn.Linear(hidden, hidden, dtype=torch.float64)
        self.head = nn.Linear(hidden, num_actions, dtype=torch.float64)

    def forward(self, x: Tensor) -> Tensor:
        h = torch.tanh(self.fc2(torch.tanh(self.fc1(x))))
        return self.head(h)


class Critic(nn.Module):
    def __init__(self, obs_dim: int = 6, hidden: int = 64) -> None:
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden, dtype=torch.float64)
        self.fc2 = nn.Linear(hidden, hidden, dtype=torch.float64)
        self.head = nn.Linear(hidden, 1, dtype=torch.float64)

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


# ---------------------------------------------------------------------------
# GAE
# ---------------------------------------------------------------------------


def gae(
    rewards: list[float], values: list[float], dones: list[bool], bootstrap: float,
    gamma: float, lam: float,
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


# ---------------------------------------------------------------------------
# PPO training
# ---------------------------------------------------------------------------


def collect_rollout(
    actor: Actor, critic: Critic, state: AcrobotState, n_steps: int, max_ep_len: int,
    rng: random.Random,
) -> tuple[
    list[Tensor], list[int], list[float], list[float], list[float], list[bool],
    AcrobotState, list[float],
]:
    obs_list: list[Tensor] = []
    act_list: list[int] = []
    lp_list: list[float] = []
    rew_list: list[float] = []
    val_list: list[float] = []
    done_list: list[bool] = []
    ep_returns: list[float] = []
    ep_sum = 0.0
    ep_len = 0
    for _ in range(n_steps):
        obs = observe(state)
        action, lp = sample_action(actor, obs, rng)
        with torch.no_grad():
            v = float(critic(obs).item())
        reward, state, terminated = acrobot_step(state, action)
        ep_sum += reward
        ep_len += 1
        truncate = ep_len >= max_ep_len
        done = terminated or truncate
        obs_list.append(obs)
        act_list.append(action)
        lp_list.append(lp)
        rew_list.append(reward)
        val_list.append(v)
        done_list.append(done)
        if done:
            ep_returns.append(ep_sum)
            ep_sum = 0.0
            ep_len = 0
            state = AcrobotState()
    return obs_list, act_list, lp_list, rew_list, val_list, done_list, state, ep_returns


def ppo_update(
    actor: Actor, critic: Critic, actor_opt: torch.optim.Optimizer,
    critic_opt: torch.optim.Optimizer, obs: Tensor, actions: Tensor,
    old_log_probs: Tensor, advantages: Tensor, returns: Tensor, clip_eps: float,
    entropy_coef: float, k_epochs: int, batch_size: int, rng: random.Random,
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
    total_rollouts: int = 100, rollout_steps: int = 1024, lr: float = 3e-4,
    gamma: float = 0.99, lam: float = 0.95, clip_eps: float = 0.2,
    entropy_coef: float = 0.01, k_epochs: int = 10, batch_size: int = 64,
    max_ep_len: int = MAX_STEPS, seed: int = 42, log_every: int = 10,
) -> tuple[Actor, list[float]]:
    torch.manual_seed(seed)
    rng = random.Random(seed)
    actor = Actor()
    critic = Critic()
    actor_opt = torch.optim.Adam(actor.parameters(), lr=lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=lr)
    state = AcrobotState()
    history: list[float] = []
    for r in range(total_rollouts):
        obs_l, act_l, lp_l, rew_l, val_l, done_l, state, ep_rets = collect_rollout(
            actor, critic, state, rollout_steps, max_ep_len, rng,
        )
        with torch.no_grad():
            bootstrap = (
                0.0 if (done_l and done_l[-1])
                else float(critic(observe(state)).item())
            )
        advs, rets = gae(rew_l, val_l, done_l, bootstrap, gamma, lam)
        obs_t = torch.stack(obs_l)
        act_t = torch.tensor(act_l, dtype=torch.long)
        lp_t = torch.tensor(lp_l, dtype=torch.float64)
        adv_t = torch.tensor(advs, dtype=torch.float64)
        ret_t = torch.tensor(rets, dtype=torch.float64)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        ppo_update(
            actor, critic, actor_opt, critic_opt, obs_t, act_t, lp_t, adv_t, ret_t,
            clip_eps, entropy_coef, k_epochs, batch_size, rng,
        )
        history.extend(ep_rets)
        if (r + 1) % log_every == 0:
            recent = history[-50:] or [0.0]
            print(
                f"  rollout {r + 1:4d}  eps={len(ep_rets):2d}  "
                f"recent_50_return={sum(recent)/len(recent):.1f}"
            )
    return actor, history


def evaluate(actor: Actor, n_episodes: int = 20, max_ep_len: int = MAX_STEPS) -> float:
    total = 0.0
    for _ in range(n_episodes):
        state = AcrobotState()
        ep_return = 0.0
        for _ in range(max_ep_len):
            obs = observe(state)
            with torch.no_grad():
                logits = actor(obs)
            action = int(torch.argmax(logits).item())
            reward, state, terminated = acrobot_step(state, action)
            ep_return += reward
            if terminated:
                break
        total += ep_return
    return total / n_episodes


if __name__ == "__main__":
    print("=== PPO on Acrobot ===")
    actor, history = train_ppo()
    avg = evaluate(actor)
    print(f"\nEval (20 episodes, greedy): avg_return={avg:.1f}")
    print(f"RESULT\tavg_return={avg:.1f}\trollouts={100}\tseed=42")
