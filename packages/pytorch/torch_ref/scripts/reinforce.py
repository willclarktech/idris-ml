"""REINFORCE on CartPole training script.

Output format matches the Idris Example.Reinforce exactly.

Usage:
    python -m torch_ref.scripts.reinforce [--lr 0.001] [--epochs 2000] [--seed 42]
"""

import argparse
import os
import sys
import time
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor

from torch_ref.init_manifest import (
    maybe_dump_after_step,
    maybe_dump_init,
    maybe_dump_oracle,
)
from torch_ref.models.reinforce import (
    MAX_STEPS,
    CartPoleEnv,
    PolicyNetwork,
    discounted_returns,
    evaluate,
    make_cartpole_env,
    obs_tensor,
    reinforce_epoch,
)
from torch_ref.replay import write_replay
from torch_ref.training.lr_finder import LrFindConfig, lr_find
from torch_ref.training.runner import (
    format_elapsed,
    format_result,
    get_device,
    get_dtype,
    mem_suffix,
    multinomial_safe,
    set_device,
)

# Idris registry name -> this script's parameter name, model-index prefixed.
# Mirrors the entry in scripts/paired_examples.py, which check-step-oracle.py
# cross-checks.
PAIRED_PARAMS = {
    "linear_0.bias": "0.fc1.bias",
    "linear_0.weight": "0.fc1.weight",
    "linear_1.bias": "0.fc2.bias",
    "linear_1.weight": "0.fc2.weight",
}


def _internal_state(env: "CartPoleEnv") -> np.ndarray:
    """The sub-env's exact float64 state. The oracle feeds these to the
    policy instead of Gymnasium's float32-rounded observations — idris-gym
    observes full-precision states (see a2c's oracle rollout)."""
    return np.asarray(cast("Any", env).unwrapped.state, dtype=np.float64)


def _oracle_episode(
    env: "CartPoleEnv", policy: PolicyNetwork, env_uniforms: list[float], choices: list[int]
) -> tuple[list[Tensor], list[float]]:
    """collect_episode's oracle twin: exact-state obs, and every draw logged
    in the order the Idris side consumes it — the reset's four uniforms
    (inverted through CartPole's U(-0.05, 0.05) affine), then one action
    decision per step."""
    env.reset()
    state = _internal_state(env)
    env_uniforms.extend(float(u) for u in (state + 0.05) / 0.1)
    log_probs: list[Tensor] = []
    rewards: list[float] = []
    for _ in range(MAX_STEPS):
        obs = obs_tensor(state)
        logits = policy(obs)
        log_p = torch.log_softmax(logits, dim=0)
        probs = torch.exp(log_p)
        action = int(multinomial_safe(probs, 1).item())
        choices.append(action)
        log_probs.append(log_p[action])
        _, reward, term, trunc, _ = env.step(action)
        rewards.append(float(reward))
        if term or trunc:
            break
        state = _internal_state(env)
    return log_probs, rewards


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch", type=int, default=10)
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

    print("=== REINFORCE on CartPole ===")
    print(
        f"Config: lr={args.lr} epochs={args.epochs}"
        f" gamma={args.gamma} batch={args.batch} seed={args.seed}"
    )

    policy = PolicyNetwork(hidden=128).to(args.device)
    maybe_dump_init(policy)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)  # setup

    print("Architecture: Linear(4->128)->Tanh->Linear(128->2)")
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"Parameters: {n_params}")
    print()

    env = make_cartpole_env(args.seed)

    # Oracle run: publish the parameters and every draw the epoch makes —
    # per episode the reset state (as the uniforms that produced it) and the
    # action decisions — take exactly one REINFORCE update and publish the
    # result. Idris replays the draws through --replay and regenerates the
    # identical episodes; the discounted returns, baseline, policy loss,
    # clip and Adam are all under test.
    if os.environ.get("IDRISML_ORACLE_DUMP"):
        env_uniforms: list[float] = []
        choices: list[int] = []
        all_log_probs: list[Tensor] = []
        all_advantages: list[float] = []
        episode_returns: list[float] = []
        for _ in range(args.batch):
            log_probs, rewards = _oracle_episode(env, policy, env_uniforms, choices)
            episode_returns.append(sum(rewards))
            all_log_probs.extend(log_probs)
            all_advantages.extend(discounted_returns(rewards, args.gamma))
        maybe_dump_oracle((policy,), PAIRED_PARAMS)
        write_replay(
            os.environ["IDRISML_ORACLE_DUMP"] + ".replay", env=env_uniforms, choices=choices
        )
        baseline = sum(episode_returns) / len(episode_returns)
        adjusted = [g - baseline for g in all_advantages]
        optimizer.zero_grad()
        loss = torch.tensor(0.0, dtype=get_dtype(), device=get_device())
        for lp, adv in zip(all_log_probs, adjusted, strict=True):
            loss = loss - lp * adv
        loss = loss / len(all_log_probs)
        loss.backward()  # pyright: ignore[reportUnknownMemberType]
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()  # pyright: ignore[reportUnknownMemberType]
        maybe_dump_after_step((policy,), PAIRED_PARAMS)

    if args.lr_find:

        def epoch_fn() -> float:
            # `reinforce_epoch` returns (mean episodic return, policy loss).
            # `lr_find` wants a "lower is better" scalar; the Idris example
            # reports `negate avg_return` to runTraining, so we match.
            avg_ret, _ = reinforce_epoch(env, policy, optimizer, args.batch, args.gamma)
            return -avg_ret

        lr_find(LrFindConfig(num_iters=100), epoch_fn, optimizer)
        print()
        print("Done — re-run without --lr-find at the recommended LR.")
        sys.exit(0)

    print("Training...")
    t_start = time.monotonic()
    history: list[float] = []
    for epoch in range(args.epochs):
        avg_return, loss_val = reinforce_epoch(env, policy, optimizer, args.batch, args.gamma)
        history.append(avg_return)
        if epoch % 100 == 0 or epoch == args.epochs - 1:
            recent = sum(history[-100:]) / min(len(history), 100)
            print(
                f"  {format_elapsed(t_start)} {epoch}\tloss={loss_val:.6f}"
                f"{mem_suffix()}\treturn={avg_return:.1f}\trecent_100={recent:.1f}"
            )

    elapsed = time.monotonic() - t_start
    ms_per_ep = elapsed / args.epochs * 1000
    print(f"Completed in {elapsed:.0f}s ({args.epochs} epochs, {ms_per_ep:.0f}ms/epoch)")
    print(f"PERF_MS_PER_EP={ms_per_ep:.6f}")

    print()
    print("Eval (100 episodes, greedy):")
    avg_return = evaluate(policy, n_episodes=100)
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
