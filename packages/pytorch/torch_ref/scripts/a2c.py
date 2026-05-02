"""A2C on CartPole training script with --lr-find support.

Output format and epoch semantics align with `Example.A2c` (Idris):
each "epoch" = one rollout + one a2c_update, and the "loss" reported
to lr_find is `-avg_episode_return` (matching the Idris convention so
the cross-backend lr_find comparison is meaningful).

Usage:
    python -m torch_ref.scripts.a2c [--lr 7e-4] [--epochs 5000] [--seed 42]
                                     [--lr-find]
"""

from __future__ import annotations

import argparse
import sys
import time

import torch

from torch_ref.models.a2c import (
    Actor,
    Critic,
    a2c_update,
    collect_rollout,
    compute_advantages,
    evaluate,
)
from torch_ref.models.reinforce import CartPoleState, observe
from torch_ref.training.lr_finder import LrFindConfig, lr_find
from torch_ref.training.runner import format_result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--epochs", type=int, default=5000,
                        help="number of A2C updates")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lambda", dest="lam", type=float, default=0.95)
    parser.add_argument("--entropy", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--rollout", type=int, default=10)
    parser.add_argument(
        "--lr-find",
        action="store_true",
        help="Run lr_find (LR-range test) instead of training, then exit.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    print("=== A2C on CartPole (separate actor + critic) ===")
    print(
        f"Config: lr={args.lr} epochs={args.epochs} rollout={args.rollout}"
        f" gamma={args.gamma} lambda={args.lam} entropy={args.entropy}"
        f" seed={args.seed}"
    )

    actor = Actor()
    critic = Critic()
    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()), lr=args.lr,
    )
    print()

    # Stateful epoch context: env state + running episodic return.
    state = [CartPoleState()]
    running_return = [0.0]

    def epoch_fn() -> float:
        """One A2C update. Returns -avg_episodic_return (matches Idris loss)."""
        obs, actions, rewards, values, dones, new_state = collect_rollout(
            actor, critic, state[0], args.rollout
        )
        with torch.no_grad():
            bootstrap_v = critic(observe(new_state))
            bootstrap = (
                0.0 if dones[-1].item() > 0.5 else float(bootstrap_v.item())
            )
        advantages, returns = compute_advantages(
            rewards, values, dones, bootstrap, args.gamma, args.lam,
        )
        a2c_update(
            actor, critic, optimizer, obs, actions, advantages, returns,
            args.entropy, args.value_coef,
        )

        # Track per-episode returns within the rollout (Idris convention).
        ep_returns: list[float] = []
        run = running_return[0]
        for t in range(args.rollout):
            run += float(rewards[t].item())
            if dones[t].item() > 0.5:
                ep_returns.append(run)
                run = 0.0
        running_return[0] = run
        state[0] = new_state

        last_terminated = dones[-1].item() > 0.5
        sum_rew = float(rewards.sum().item())
        if last_terminated and ep_returns:
            reported = ep_returns[-1]
        else:
            reported = sum_rew
        return -reported  # Idris reports `negate avg_return`

    if args.lr_find:
        lr_find(LrFindConfig(num_iters=100), epoch_fn, optimizer)
        print()
        print("Done — re-run without --lr-find at the recommended LR.")
        sys.exit(0)

    print("Training...")
    t0 = time.time()
    for epoch in range(args.epochs):
        loss = epoch_fn()
        if epoch % 500 == 0 or epoch == args.epochs - 1:
            elapsed = time.time() - t0
            print(f"  [{elapsed:07.2f}s] {epoch}\tloss={loss:.6f}")

    elapsed = time.time() - t0
    ms_per_ep = elapsed / args.epochs * 1000
    print(f"Completed in {elapsed:.0f}s ({args.epochs} updates, {ms_per_ep:.0f}ms/update)")

    print()
    print("Eval (50 episodes, greedy):")
    avg_return = evaluate(actor, n_episodes=50)
    print(f"  avg_return={avg_return:.1f}")

    print()
    print(format_result([
        ("avg_return", f"{avg_return:.1f}"),
        ("epochs", str(args.epochs)),
        ("seed", str(args.seed)),
    ]))


if __name__ == "__main__":
    main()
