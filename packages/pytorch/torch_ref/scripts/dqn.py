"""DQN on CartPole training script with --lr-find support.

Output format and epoch semantics align with `Example.Dqn` (Idris):
each "epoch" = one full episode + intra-episode replay updates, and
the "loss" reported to lr_find is `-episode_return` (matching the
Idris convention so the cross-backend lr_find comparison is meaningful).

Usage:
    python -m torch_ref.scripts.dqn [--lr 5e-4] [--epochs 300] [--seed 42]
                                     [--lr-find]
"""

from __future__ import annotations

import argparse
import copy
import random
import sys
import time

import torch

from torch_ref.init_manifest import maybe_dump_init
from torch_ref.models.dqn import (
    NUM_ENVS,
    QNetwork,
    ReplayBuffer,
    dqn_episode_batched,
    evaluate,
    make_cartpole_vec_env,
)
from torch_ref.training.lr_finder import LrFindConfig, lr_find
from torch_ref.training.runner import format_elapsed, format_result, mem_suffix, set_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=300, help="number of episodes")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--buffer-cap", dest="buffer", type=int, default=10000)
    parser.add_argument("--target-sync", type=int, default=100)
    parser.add_argument(
        "--lr-find",
        action="store_true",
        help="Run lr_find (LR-range test) instead of training, then exit.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "mps", "cuda"],
        help="Device for tensor ops (default: cpu)",
    )
    args = parser.parse_args()

    set_device(args.device)
    torch.manual_seed(args.seed)  # pyright: ignore[reportUnknownMemberType] - untyped `seed` param in torch stubs
    rng = random.Random(args.seed)

    print("=== DQN on CartPole ===")
    print(
        f"Config: lr={args.lr} epochs={args.epochs} gamma={args.gamma}"
        f" batch={args.batch} buffer={args.buffer}"
        f" target_sync={args.target_sync} seed={args.seed}"
    )

    q = QNetwork().to(args.device)
    target = copy.deepcopy(q)
    maybe_dump_init(q, target)
    optimizer = torch.optim.Adam(q.parameters(), lr=args.lr)
    buffer = ReplayBuffer(args.buffer)
    vec_env, obs0 = make_cartpole_vec_env(args.seed, NUM_ENVS)
    obs_state = [obs0]
    step_count = [0]
    print()

    def epoch_fn() -> float:
        """One batched DQN episode. Returns -env-0_episode_return
        (matches Idris loss)."""
        new_step, ep_return, new_obs = dqn_episode_batched(
            vec_env,
            obs_state[0],
            q,
            target,
            optimizer,
            buffer,
            step_count[0],
            args.batch,
            args.gamma,
            args.target_sync,
            rng,
        )
        step_count[0] = new_step
        obs_state[0] = new_obs
        return -ep_return  # Idris reports `negate ret`

    if args.lr_find:
        # 30 iters: each iter is a full episode (up to 200 steps), heavier
        # than supervised so we use the same count as Idris.
        lr_find(LrFindConfig(num_iters=30), epoch_fn, optimizer)
        print()
        print("Done — re-run without --lr-find at the recommended LR.")
        sys.exit(0)

    print("Training...")
    t_start = time.monotonic()
    history: list[float] = []
    for epoch in range(args.epochs):
        loss_val = epoch_fn()
        ep_return = -loss_val
        history.append(ep_return)
        if (epoch + 1) % 50 == 0:
            recent = sum(history[-50:]) / min(len(history), 50)
            print(
                f"  {format_elapsed(t_start)} {epoch + 1}\tloss={loss_val:.6f}"
                f"{mem_suffix()}\treturn={ep_return:.1f}\trecent_50={recent:.1f}"
            )

    elapsed = time.monotonic() - t_start
    ms_per_ep = elapsed / args.epochs * 1000
    print(f"Completed in {elapsed:.0f}s ({args.epochs} episodes, {ms_per_ep:.0f}ms/episode)")
    print(f"PERF_MS_PER_EP={ms_per_ep:.6f}")

    print()
    print("Eval (50 episodes, greedy):")
    avg_return = evaluate(q, n_episodes=50)
    print(f"  avg_return={avg_return:.1f}")

    print()
    print(
        format_result(
            [
                ("avg_return", f"{avg_return:.1f}"),
                ("epochs", str(args.epochs)),
                ("seed", str(args.seed)),
            ]
        )
    )


if __name__ == "__main__":
    main()
