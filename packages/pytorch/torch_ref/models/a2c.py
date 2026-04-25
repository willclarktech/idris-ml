"""A2C (synchronous Advantage Actor-Critic) on CartPole-v0.

Actor-critic with shared trunk, n-step returns, entropy bonus. Rollouts
are collected across N parallel env copies (sequential "sync" style —
pure Python, no threading). Self-contained CartPole physics imported
from `reinforce.py`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_ref.models.reinforce import MAX_STEPS, CartPoleState, cartpole_step, observe


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int = 4, num_actions: int = 2, hidden: int = 64) -> None:
        super().__init__()
        self.trunk1 = nn.Linear(obs_dim, hidden, dtype=torch.float64)
        self.trunk2 = nn.Linear(hidden, hidden, dtype=torch.float64)
        self.actor_head = nn.Linear(hidden, num_actions, dtype=torch.float64)
        self.critic_head = nn.Linear(hidden, 1, dtype=torch.float64)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = torch.tanh(self.trunk2(torch.tanh(self.trunk1(x))))
        logits = self.actor_head(h)
        value = self.critic_head(h).squeeze(-1)
        return logits, value


def collect_rollout(
    ac: ActorCritic,
    states: list[CartPoleState],
    rollout_len: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, list[CartPoleState]]:
    """Roll out `rollout_len` steps across N envs in lockstep.

    Returns (obs, actions, rewards, dones, log_probs, final_states).
    Shapes: [T, N, ...] for trajectory arrays.
    """
    obs_t: list[Tensor] = []
    action_t: list[list[int]] = []
    reward_t: list[list[float]] = []
    done_t: list[list[float]] = []
    log_prob_t: list[Tensor] = []

    for _ in range(rollout_len):
        obs = torch.stack([observe(s) for s in states])  # [N, obs_dim]
        logits, _ = ac(obs)
        probs = F.softmax(logits, dim=-1)
        actions = torch.multinomial(probs, 1).squeeze(-1)  # [N]
        log_probs = F.log_softmax(logits, dim=-1).gather(1, actions.unsqueeze(1)).squeeze(1)

        rewards: list[float] = []
        dones: list[float] = []
        new_states: list[CartPoleState] = []
        for i, s in enumerate(states):
            r, s_next, done = cartpole_step(s, int(actions[i].item()))
            rewards.append(r)
            dones.append(1.0 if done else 0.0)
            new_states.append(CartPoleState() if done else s_next)
        states = new_states

        obs_t.append(obs)
        action_t.append([int(a.item()) for a in actions])
        reward_t.append(rewards)
        done_t.append(dones)
        log_prob_t.append(log_probs)

    obs_tensor = torch.stack(obs_t)  # [T, N, obs_dim]
    action_tensor = torch.tensor(action_t, dtype=torch.long)
    reward_tensor = torch.tensor(reward_t, dtype=torch.float64)
    done_tensor = torch.tensor(done_t, dtype=torch.float64)
    log_prob_tensor = torch.stack(log_prob_t)  # [T, N]
    return obs_tensor, action_tensor, reward_tensor, done_tensor, log_prob_tensor, states


def compute_advantages(
    rewards: Tensor, values: Tensor, dones: Tensor, bootstrap: Tensor, gamma: float, lam: float
) -> tuple[Tensor, Tensor]:
    """GAE advantages + return targets. Shapes: rewards/values/dones [T, N],
    bootstrap [N]. Returns (advantages, returns) each [T, N]."""
    t_len = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros_like(bootstrap)
    for t in reversed(range(t_len)):
        next_v = bootstrap if t == t_len - 1 else values[t + 1]
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_v * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        advantages[t] = gae
    returns = advantages + values
    return advantages, returns


def a2c_update(
    ac: ActorCritic,
    optimizer: torch.optim.Optimizer,
    obs: Tensor,
    actions: Tensor,
    rewards: Tensor,
    dones: Tensor,
    final_states: list[CartPoleState],
    gamma: float,
    lam: float,
    entropy_coef: float,
    value_coef: float,
) -> float:
    """One A2C gradient step. Returns loss value."""
    t_len, n_envs = rewards.shape

    # Re-forward to get grad-tracked logits + values
    flat_obs = obs.reshape(t_len * n_envs, -1)
    logits, values = ac(flat_obs)
    logits = logits.reshape(t_len, n_envs, -1)
    values = values.reshape(t_len, n_envs)

    # Bootstrap: V(s_T) at final states
    final_obs = torch.stack([observe(s) for s in final_states])  # [N, obs_dim]
    with torch.no_grad():
        _, bootstrap_v = ac(final_obs)

    advantages, returns = compute_advantages(
        rewards, values.detach(), dones, bootstrap_v, gamma, lam
    )
    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    log_probs = F.log_softmax(logits, dim=-1).gather(2, actions.unsqueeze(-1)).squeeze(-1)
    probs = F.softmax(logits, dim=-1)
    entropy = -(probs * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()

    policy_loss = -(log_probs * advantages).mean()
    value_loss = F.mse_loss(values, returns)
    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(ac.parameters(), 0.5)
    optimizer.step()
    return float(loss.item())


def train_a2c(
    total_updates: int = 1000,
    n_envs: int = 8,
    rollout_len: int = 20,
    lr: float = 7e-4,
    gamma: float = 0.99,
    lam: float = 0.95,
    entropy_coef: float = 0.01,
    value_coef: float = 0.5,
    seed: int = 42,
    log_every: int = 100,
) -> tuple[ActorCritic, list[float]]:
    torch.manual_seed(seed)
    ac = ActorCritic()
    optimizer = torch.optim.Adam(ac.parameters(), lr=lr)
    states = [CartPoleState() for _ in range(n_envs)]
    history: list[float] = []
    # Track episodic returns per env
    ep_returns = [0.0] * n_envs
    recent_returns: list[float] = []
    for update in range(total_updates):
        obs, actions, rewards, dones, _, new_states = collect_rollout(ac, states, rollout_len)
        a2c_update(
            ac, optimizer, obs, actions, rewards, dones, new_states,
            gamma, lam, entropy_coef, value_coef,
        )
        # Track per-env episodic returns (rewards here are per-step; episodes end when done=1)
        for t in range(rollout_len):
            for i in range(n_envs):
                ep_returns[i] += float(rewards[t, i].item())
                if dones[t, i].item() > 0.5:
                    recent_returns.append(ep_returns[i])
                    ep_returns[i] = 0.0
        states = new_states
        history.append(sum(recent_returns[-50:]) / max(1, min(len(recent_returns), 50)))
        if (update + 1) % log_every == 0:
            print(f"  update {update + 1:4d}  recent_50_return={history[-1]:.1f}")
    return ac, history


def evaluate(ac: ActorCritic, n_episodes: int = 50) -> float:
    total = 0.0
    for _ in range(n_episodes):
        state = CartPoleState()
        ep_return = 0.0
        for _ in range(MAX_STEPS):
            obs = observe(state).unsqueeze(0)
            with torch.no_grad():
                logits, _ = ac(obs)
            action = int(torch.argmax(logits, dim=-1).item())
            reward, state, done = cartpole_step(state, action)
            ep_return += reward
            if done:
                break
        total += ep_return
    return total / n_episodes


if __name__ == "__main__":
    print("=== A2C on CartPole ===")
    ac, history = train_a2c()
    avg = evaluate(ac)
    print(f"\nEval (50 episodes, greedy): avg_return={avg:.1f}")
    print(f"RESULT\tavg_return={avg:.1f}\tupdates={len(history)}\tseed=42")
