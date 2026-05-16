"""A2C (synchronous Advantage Actor-Critic) on CartPole-v0.

Aligned with `Example.A2c` (Idris). Separate actor and critic networks
(Idris uses distinct paramId prefixes via `prefixParamId` + `emap` to
register them in the same optimizer without name collisions). Sequential
single-env rollouts with auto-reset, matching Idris.
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_ref.models.reinforce import MAX_STEPS, CartPoleState, cartpole_step, observe
from torch_ref.training.runner import format_elapsed, mem_suffix


class Actor(nn.Module):
    def __init__(self, obs_dim: int = 4, num_actions: int = 2, hidden: int = 64) -> None:
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden, dtype=torch.float64)
        self.fc2 = nn.Linear(hidden, hidden, dtype=torch.float64)
        self.head = nn.Linear(hidden, num_actions, dtype=torch.float64)

    def forward(self, x: Tensor) -> Tensor:
        h = torch.tanh(self.fc2(torch.tanh(self.fc1(x))))
        return self.head(h)


class Critic(nn.Module):
    def __init__(self, obs_dim: int = 4, hidden: int = 64) -> None:
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden, dtype=torch.float64)
        self.fc2 = nn.Linear(hidden, hidden, dtype=torch.float64)
        self.head = nn.Linear(hidden, 1, dtype=torch.float64)

    def forward(self, x: Tensor) -> Tensor:
        h = torch.tanh(self.fc2(torch.tanh(self.fc1(x))))
        return self.head(h).squeeze(-1)


def collect_rollout(
    actor: Actor, critic: Critic, state: CartPoleState, rollout_len: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, CartPoleState]:
    """Single-env sequential rollout of exactly `rollout_len` steps with
    auto-reset on done. Matches Idris exactly."""
    obs_list: list[Tensor] = []
    act_list: list[int] = []
    rew_list: list[float] = []
    val_list: list[float] = []
    done_list: list[float] = []
    for _ in range(rollout_len):
        obs = observe(state)
        with torch.no_grad():
            logits = actor(obs)
            value = critic(obs)
        probs = F.softmax(logits, dim=-1)
        action = int(torch.multinomial(probs, 1).item())
        reward, next_state, done = cartpole_step(state, action)
        obs_list.append(obs)
        act_list.append(action)
        rew_list.append(reward)
        val_list.append(float(value.item()))
        done_list.append(1.0 if done else 0.0)
        state = CartPoleState() if done else next_state
    return (
        torch.stack(obs_list),
        torch.tensor(act_list, dtype=torch.long),
        torch.tensor(rew_list, dtype=torch.float64),
        torch.tensor(val_list, dtype=torch.float64),
        torch.tensor(done_list, dtype=torch.float64),
        state,
    )


def compute_advantages(
    rewards: Tensor, values: Tensor, dones: Tensor, bootstrap: float,
    gamma: float, lam: float,
) -> tuple[Tensor, Tensor]:
    """GAE, single-env, inputs [T]. Returns (advantages[T], returns[T])."""
    t_len = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    gae_val = 0.0
    v_next = bootstrap
    for t in reversed(range(t_len)):
        mask = 1.0 - float(dones[t].item())
        delta = float(rewards[t].item()) + gamma * v_next * mask - float(values[t].item())
        gae_val = delta + gamma * lam * mask * gae_val
        advantages[t] = gae_val
        v_next = float(values[t].item())
    returns = advantages + values
    return advantages, returns


def a2c_update(
    actor: Actor, critic: Critic, optimizer: torch.optim.Optimizer,
    obs: Tensor, actions: Tensor, advantages: Tensor, returns: Tensor,
    entropy_coef: float, value_coef: float,
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
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        list(actor.parameters()) + list(critic.parameters()), 0.5,
    )
    optimizer.step()
    return float(loss.item())


def train_a2c(
    total_updates: int = 5000, rollout_len: int = 10, lr: float = 7e-4,
    gamma: float = 0.99, lam: float = 0.95, entropy_coef: float = 0.01,
    value_coef: float = 0.5, seed: int = 42, log_every: int = 500,
) -> tuple[Actor, Critic, list[float]]:
    """Hyperparameters match Idris `Example.A2c.defaultConfig`:
    lr=7e-4, entropy=0.01, rollout=10, gamma=0.99, lam=0.95."""
    torch.manual_seed(seed)
    actor = Actor()
    critic = Critic()
    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()), lr=lr,
    )
    state = CartPoleState()
    history: list[float] = []
    ep_return = 0.0
    t_start = time.monotonic()
    for update in range(total_updates):
        obs, actions, rewards, values, dones, new_state = collect_rollout(
            actor, critic, state, rollout_len
        )
        with torch.no_grad():
            bootstrap_v = critic(observe(new_state))
            bootstrap = 0.0 if dones[-1].item() > 0.5 else float(bootstrap_v.item())
        advantages, returns = compute_advantages(rewards, values, dones, bootstrap, gamma, lam)
        a2c_update(
            actor, critic, optimizer, obs, actions, advantages, returns,
            entropy_coef, value_coef,
        )
        for t in range(rollout_len):
            ep_return += float(rewards[t].item())
            if dones[t].item() > 0.5:
                history.append(ep_return)
                ep_return = 0.0
        state = new_state
        if (update + 1) % log_every == 0:
            recent = history[-50:] or [0.0]
            last_ep = history[-1] if history else 0.0
            print(
                f"  {format_elapsed(t_start)} {update + 1}\tloss={-last_ep:.6f}"
                f"{mem_suffix()}\treturn={last_ep:.1f}"
                f"\trecent_50={sum(recent)/len(recent):.1f}"
            )
    return actor, critic, history


def evaluate(actor: Actor, n_episodes: int = 50) -> float:
    total = 0.0
    for _ in range(n_episodes):
        state = CartPoleState()
        ep_return = 0.0
        for _ in range(MAX_STEPS):
            obs = observe(state)
            with torch.no_grad():
                logits = actor(obs)
            action = int(torch.argmax(logits, dim=-1).item())
            reward, state, done = cartpole_step(state, action)
            ep_return += reward
            if done:
                break
        total += ep_return
    return total / n_episodes


if __name__ == "__main__":
    print("=== A2C on CartPole (separate actor + critic) ===")
    actor, _critic, history = train_a2c()
    avg = evaluate(actor)
    print(f"\nEval (50 episodes, greedy): avg_return={avg:.1f}")
    print(f"RESULT\tavg_return={avg:.1f}\tupdates={5000}\tseed=42")
