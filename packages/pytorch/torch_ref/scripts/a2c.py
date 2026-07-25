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

import numpy as np
import torch

from torch_ref.init_manifest import (
    maybe_dump_after_step,
    maybe_dump_init,
    maybe_load_oracle_tensors,
)
from torch_ref.models.a2c import (
    NUM_ENVS,
    Actor,
    Critic,
    a2c_update,
    collect_rollout,
    compute_advantages,
    evaluate,
    make_cartpole_vec_env,
)
from torch_ref.models.reinforce import obs_tensor
from torch_ref.training.lr_finder import LrFindConfig, lr_find
from torch_ref.training.runner import (
    format_elapsed,
    format_result,
    mem_suffix,
    set_device,
)

# Idris registry name -> this script's parameter name, model-index prefixed
# (0 = actor, 1 = critic). Mirrors the entry in scripts/paired_examples.py,
# which check-step-oracle.py cross-checks.
PAIRED_PARAMS = {
    "actor.linear_0.bias": "0.fc1.bias",
    "actor.linear_0.weight": "0.fc1.weight",
    "actor.linear_1.bias": "0.fc2.bias",
    "actor.linear_1.weight": "0.fc2.weight",
    "actor.linear_2.bias": "0.head.bias",
    "actor.linear_2.weight": "0.head.weight",
    "critic.linear_0.bias": "1.fc1.bias",
    "critic.linear_0.weight": "1.fc1.weight",
    "critic.linear_1.bias": "1.fc2.bias",
    "critic.linear_1.weight": "1.fc2.weight",
    "critic.linear_2.bias": "1.head.bias",
    "critic.linear_2.weight": "1.head.weight",
}

# What Example.A2c registers under IDRISML_ORACLE_DUMP. Laid out per env
# ([NumEnvs, RolloutLen]) because that is the order the Idris side flattens in;
# `compute_advantages` wants [T, N], hence the transpose below.
ORACLE_ROLLOUT = (
    "__oracle.obs",
    "__oracle.actions",
    "__oracle.rewards",
    "__oracle.values",
    "__oracle.dones",
    "__oracle.bootstraps",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--epochs", type=int, default=5000, help="number of A2C updates")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lambda", dest="lam", type=float, default=0.95)
    parser.add_argument("--entropy", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--rollout", type=int, default=20)
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

    print("=== A2C on CartPole (separate actor + critic) ===")
    print(
        f"Config: lr={args.lr} epochs={args.epochs} rollout={args.rollout}"
        f" gamma={args.gamma} lambda={args.lam} entropy={args.entropy}"
        f" seed={args.seed}"
    )

    actor = Actor().to(args.device)
    critic = Critic().to(args.device)
    maybe_dump_init(actor, critic)
    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()),
        lr=args.lr,
    )
    print()

    # Oracle run: take Idris' init weights and the rollout it just collected,
    # recompute advantages, take exactly one update, dump. The environment and
    # the action sampler are never touched — their two RNG streams differ by
    # design — so what is under test is GAE, the advantage normalization, the
    # policy/value/entropy loss, the clip and Adam.
    rollout = maybe_load_oracle_tensors((actor, critic), PAIRED_PARAMS, ORACLE_ROLLOUT)
    if rollout is not None:
        n_envs = rollout["__oracle.bootstraps"].shape[0]

        def per_env(name: str) -> torch.Tensor:
            """[N*T, 1] in env-major order -> [T, N], which GAE wants."""
            return rollout[name].reshape(n_envs, -1).t().contiguous()

        adv, ret = compute_advantages(
            per_env("__oracle.rewards"),
            per_env("__oracle.values"),
            per_env("__oracle.dones"),
            rollout["__oracle.bootstraps"].reshape(-1),
            args.gamma,
            args.lam,
        )

        # Back to the Idris side's env-major flattening so every row of the
        # batch still lines up with its observation.
        def flat(t: torch.Tensor) -> torch.Tensor:
            return t.t().contiguous().reshape(-1)

        a2c_update(
            actor,
            critic,
            optimizer,
            rollout["__oracle.obs"],
            rollout["__oracle.actions"].reshape(-1).long(),
            flat(adv),
            flat(ret),
            args.entropy,
            args.value_coef,
        )
        maybe_dump_after_step(actor, critic)

    # Stateful epoch context: NUM_ENVS parallel envs + per-env running
    # episodic returns. Matches Idris-side `A2CState.{envRef, retRef}`.
    vec_env = make_cartpole_vec_env(args.seed, NUM_ENVS)
    obs_state = [np.zeros((NUM_ENVS, 4), dtype=np.float64)]
    running_returns = [np.zeros(NUM_ENVS, dtype=np.float64)]

    def epoch_fn() -> float:
        """One batched A2C update across NUM_ENVS envs. Returns
        -avg_episodic_return (matches Idris loss)."""
        obs, actions, rewards, values, dones, new_obs = collect_rollout(
            actor, critic, vec_env, obs_state[0], args.rollout
        )
        with torch.no_grad():
            bootstrap_v = critic(obs_tensor(new_obs))  # [N]
            last_done = dones[-1]  # [N]
            bootstraps = torch.where(
                last_done > 0.5,
                torch.zeros_like(bootstrap_v),
                bootstrap_v,
            )
        advantages, returns = compute_advantages(
            rewards,
            values,
            dones,
            bootstraps,
            args.gamma,
            args.lam,
        )
        a2c_update(
            actor,
            critic,
            optimizer,
            obs.reshape(-1, 4),
            actions.reshape(-1),
            advantages.reshape(-1),
            returns.reshape(-1),
            args.entropy,
            args.value_coef,
        )

        ep_returns: list[float] = []
        run = running_returns[0]
        rewards_np = rewards.cpu().numpy()
        dones_np = dones.cpu().numpy()
        for t in range(args.rollout):
            for env_idx in range(NUM_ENVS):
                run[env_idx] += float(rewards_np[t, env_idx])
                if dones_np[t, env_idx] > 0.5:
                    ep_returns.append(float(run[env_idx]))
                    run[env_idx] = 0.0
        running_returns[0] = run
        obs_state[0] = new_obs

        sum_rew = float(rewards.sum().item()) / NUM_ENVS
        reported = sum(ep_returns) / len(ep_returns) if ep_returns else sum_rew
        return -reported  # Idris reports `negate avg_return`

    if args.lr_find:
        lr_find(LrFindConfig(num_iters=100), epoch_fn, optimizer)
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
        if epoch % 500 == 0 or epoch == args.epochs - 1:
            recent = sum(history[-50:]) / min(len(history), 50)
            print(
                f"  {format_elapsed(t_start)} {epoch}\tloss={loss_val:.6f}"
                f"{mem_suffix()}\treturn={reported:.1f}\trecent_50={recent:.1f}"
            )

    elapsed = time.monotonic() - t_start
    ms_per_ep = elapsed / args.epochs * 1000
    print(f"Completed in {elapsed:.0f}s ({args.epochs} updates, {ms_per_ep:.0f}ms/update)")
    print(f"PERF_MS_PER_EP={ms_per_ep:.6f}")

    print()
    print("Eval (50 episodes, greedy):")
    avg_return = evaluate(actor, n_episodes=50)
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
