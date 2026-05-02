"""PPO on Acrobot training script with --lr-find support.

Mirrors `Example.Ppo` (Idris): each "epoch" = one rollout (1024 steps) +
K=10 mini-batch passes. Loss reported to lr_find is `-avg_episode_return`
over episodes completed in that rollout (matching `ppoEpoch`).

Each iter is heavy (1024 env steps + K=10 PPO updates), so lr_find
defaults to 30 iters (matching the Idris-side default).

Usage:
    python -m torch_ref.scripts.ppo [--lr 3e-4] [--epochs 100] [--seed 42] [--lr-find]
"""

from __future__ import annotations

import argparse
import random
import sys
import time

import torch

from torch_ref.models.ppo import (
    AcrobotState,
    Actor,
    Critic,
    collect_rollout,
    evaluate,
    gae,
    ppo_update,
)
from torch_ref.training.lr_finder import LrFindConfig, lr_find
from torch_ref.training.runner import format_result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=100,
                        help="number of PPO rollouts")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lambda", dest="lam", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--entropy", type=float, default=0.01)
    parser.add_argument("--rollout", type=int, default=1024)
    parser.add_argument("--k-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-ep-len", type=int, default=500)
    parser.add_argument(
        "--lr-find",
        action="store_true",
        help="Run lr_find (LR-range test) instead of training, then exit.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    print("=== PPO on Acrobot (separate actor + critic, categorical policy) ===")
    print(
        f"Config: lr={args.lr} epochs={args.epochs} rollout={args.rollout}"
        f" k_epochs={args.k_epochs} batch={args.batch_size}"
        f" gamma={args.gamma} lambda={args.lam} clip={args.clip_eps}"
        f" entropy={args.entropy} seed={args.seed}"
    )

    actor = Actor()
    critic = Critic()
    actor_opt = torch.optim.Adam(actor.parameters(), lr=args.lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=args.lr)
    print()

    state = [AcrobotState()]

    def epoch_fn() -> float:
        """One PPO rollout + update. Returns -avg_ep_return (matches Idris)."""
        obs_l, act_l, lp_l, rew_l, val_l, done_l, new_state, ep_rets = collect_rollout(
            actor, critic, state[0], args.rollout, args.max_ep_len, rng,
        )
        with torch.no_grad():
            from torch_ref.models.ppo import observe
            bootstrap = (
                0.0 if (done_l and done_l[-1])
                else float(critic(observe(new_state)).item())
            )
        advs, rets = gae(rew_l, val_l, done_l, bootstrap, args.gamma, args.lam)
        obs_t = torch.stack(obs_l)
        act_t = torch.tensor(act_l, dtype=torch.long)
        lp_t = torch.tensor(lp_l, dtype=torch.float64)
        adv_t = torch.tensor(advs, dtype=torch.float64)
        ret_t = torch.tensor(rets, dtype=torch.float64)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        ppo_update(
            actor, critic, actor_opt, critic_opt, obs_t, act_t, lp_t, adv_t, ret_t,
            args.clip_eps, args.entropy, args.k_epochs, args.batch_size, rng,
        )
        state[0] = new_state

        avg_ep = sum(ep_rets) / len(ep_rets) if ep_rets else float(sum(rew_l))
        return -avg_ep  # Idris returns `negate avgEp`

    if args.lr_find:
        # Single optimizer view for lr_find: pass actor_opt; critic_opt's LR
        # is set in lockstep below via a small wrapper.
        class _BothOpts:
            def __init__(self, a: torch.optim.Optimizer, c: torch.optim.Optimizer) -> None:
                self.param_groups = a.param_groups + c.param_groups
        lr_find(LrFindConfig(num_iters=30), epoch_fn, _BothOpts(actor_opt, critic_opt))
        print()
        print("Done — re-run without --lr-find at the recommended LR.")
        sys.exit(0)

    print("Training...")
    t0 = time.time()
    for epoch in range(args.epochs):
        loss = epoch_fn()
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            elapsed = time.time() - t0
            print(f"  [{elapsed:07.2f}s] {epoch}\tloss={loss:.6f}")

    elapsed = time.time() - t0
    s_per_ep = elapsed / args.epochs
    print(f"Completed in {elapsed:.0f}s ({args.epochs} rollouts, {s_per_ep:.2f}s/rollout)")

    print()
    print("Eval (20 episodes, greedy):")
    avg_return = evaluate(actor, n_episodes=20)
    print(f"  avg_return={avg_return:.1f}")

    print()
    print(format_result([
        ("avg_return", f"{avg_return:.1f}"),
        ("epochs", str(args.epochs)),
        ("seed", str(args.seed)),
    ]))


if __name__ == "__main__":
    main()
