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

import numpy as np
import torch

from torch_ref.models.ppo import (
    NUM_ENVS,
    Actor,
    Critic,
    collect_rollout,
    evaluate,
    gae_batched,
    make_acrobot_vec_env,
    obs_tensor,
    ppo_update,
)
from torch_ref.training.lr_finder import LrFindConfig, lr_find
from torch_ref.training.runner import (
    format_elapsed,
    format_result,
    mem_suffix,
    set_device,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=100, help="number of PPO rollouts")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lambda", dest="lam", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--entropy", type=float, default=0.01)
    parser.add_argument(
        "--rollout",
        type=int,
        default=256,
        help="Per-env rollout steps (total samples = rollout * NUM_ENVS).",
    )
    parser.add_argument("--k-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-ep-len", type=int, default=500)
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
    # torch stub: manual_seed's seed param is unannotated.
    torch.manual_seed(args.seed)  # pyright: ignore[reportUnknownMemberType]
    rng = random.Random(args.seed)

    print("=== PPO on Acrobot (separate actor + critic, categorical policy) ===")
    print(
        f"Config: lr={args.lr} epochs={args.epochs} rollout={args.rollout}"
        f" k_epochs={args.k_epochs} batch={args.batch_size}"
        f" gamma={args.gamma} lambda={args.lam} clip={args.clip_eps}"
        f" entropy={args.entropy} seed={args.seed}"
    )

    actor = Actor().to(args.device)
    critic = Critic().to(args.device)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=args.lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=args.lr)
    print()

    vec_env = make_acrobot_vec_env(args.seed, NUM_ENVS)
    obs_state = [np.tile(np.array([1.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64), (NUM_ENVS, 1))]
    ep_lens_state = [np.zeros(NUM_ENVS, dtype=np.int64)]

    def epoch_fn() -> float:
        """One batched PPO rollout + update across NUM_ENVS envs. Returns
        -avg_ep_return (matches Idris)."""
        obs_t, act_t, lp_t, rew_t, val_t, done_t, new_obs, new_ep_lens, ep_rets = collect_rollout(
            actor,
            critic,
            vec_env,
            obs_state[0],
            ep_lens_state[0],
            args.rollout,
            args.max_ep_len,
            rng,
        )
        with torch.no_grad():
            bootstrap_v = critic(obs_tensor(new_obs))  # [N]
            last_done = done_t[-1]  # [N]
            bootstraps = torch.where(
                last_done > 0.5,
                torch.zeros_like(bootstrap_v),
                bootstrap_v,
            )
        adv_t, ret_t = gae_batched(rew_t, val_t, done_t, bootstraps, args.gamma, args.lam)
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
            args.clip_eps,
            args.entropy,
            args.k_epochs,
            args.batch_size,
            rng,
        )
        obs_state[0] = new_obs
        ep_lens_state[0] = new_ep_lens

        sum_rew_per_env = float(rew_t.sum().item()) / NUM_ENVS
        avg_ep = sum(ep_rets) / len(ep_rets) if ep_rets else sum_rew_per_env
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
    t_start = time.monotonic()
    history: list[float] = []
    for epoch in range(args.epochs):
        loss_val = epoch_fn()
        reported = -loss_val
        history.append(reported)
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            recent = sum(history[-50:]) / min(len(history), 50)
            print(
                f"  {format_elapsed(t_start)} {epoch}\tloss={loss_val:.6f}"
                f"{mem_suffix()}\treturn={reported:.1f}\trecent_50={recent:.1f}"
            )

    elapsed = time.monotonic() - t_start
    s_per_ep = elapsed / args.epochs
    ms_per_ep = s_per_ep * 1000
    print(f"Completed in {elapsed:.0f}s ({args.epochs} rollouts, {s_per_ep:.2f}s/rollout)")
    print(f"PERF_MS_PER_EP={ms_per_ep:.6f}")

    print()
    print("Eval (20 episodes, greedy):")
    avg_return = evaluate(actor, n_episodes=20)
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
