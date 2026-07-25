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
import os
import random
import sys
import time
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from torch_ref.init_manifest import (
    maybe_dump_after_step,
    maybe_dump_init,
    maybe_dump_oracle,
)
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
from torch_ref.replay import write_replay
from torch_ref.training.lr_finder import LrFindConfig, lr_find
from torch_ref.training.runner import (
    format_elapsed,
    format_result,
    mem_suffix,
    multinomial_safe,
    set_device,
)

if TYPE_CHECKING:
    from collections.abc import MutableSequence

    import gymnasium as gym

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


def _internal_states(vec: gym.vector.SyncVectorEnv) -> np.ndarray:
    """The sub-envs' exact float64 states [N, 4] (th1, th2, dth1, dth2).

    The oracle rollout derives observations from these instead of Gymnasium's
    observation pipeline: that pipeline rounds through float32, idris-gym's
    does not, and the rounding would put a noise floor under the comparison
    far above what the replayed rollout otherwise reaches."""
    return np.array(
        [np.asarray(e.unwrapped.state, dtype=np.float64) for e in cast("Any", vec).envs]
    )


def _reset_uniforms(states: np.ndarray) -> list[float]:
    """Invert Acrobot's U(-0.1, 0.1) start draw: the uniform that produces
    `state` under `Random.Dist.uniform` (`lo + u * (hi - lo)`). Exact up to
    one rounding of the division."""
    return [float(u) for u in (np.asarray(states).reshape(-1) + 0.1) / 0.2]


def _obs_from_states(states: np.ndarray) -> Tensor:
    """[N, 4] exact states -> [N, 6] observations, float64 throughout —
    the same [cos th1, sin th1, cos th2, sin th2, dth1, dth2] transform
    `Gym.ClassicControl.Acrobot.aObserve` applies."""
    th1, th2 = states[:, 0], states[:, 1]
    obs = np.stack(
        [np.cos(th1), np.sin(th1), np.cos(th2), np.sin(th2), states[:, 2], states[:, 3]],
        axis=1,
    )
    return obs_tensor(obs)


def _oracle_rollout(
    actor: Actor, critic: Critic, vec: gym.vector.SyncVectorEnv, rollout_len: int
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, np.ndarray, list[float]]:
    """`collect_rollout`'s oracle twin: obs come from the exact internal
    states, and every reset draw (initial and same-step auto-reset) is logged
    as the uniforms that produced it, in the order the Idris side consumes its
    env `Source`. Returns the [T, N] rollout, the final states and the logged
    uniforms. TimeLimit truncation stands in for Idris' stepsLeft counter:
    both fire at 500 steps, beyond any first rollout of 256."""
    states = _internal_states(vec)
    env_uniforms = _reset_uniforms(states)
    obs_list: list[Tensor] = []
    act_list: list[Tensor] = []
    lp_list: list[Tensor] = []
    rew_list: list[Tensor] = []
    val_list: list[Tensor] = []
    done_list: list[Tensor] = []
    for _ in range(rollout_len):
        obs_t = _obs_from_states(states)  # [N, 6]
        with torch.no_grad():
            logits = actor(obs_t)
            log_probs_t = F.log_softmax(logits, dim=-1)
            values = critic(obs_t)
        actions_t = multinomial_safe(torch.exp(log_probs_t), 1).squeeze(-1)
        actions_np = actions_t.cpu().numpy().astype(np.int64)
        lps_t = log_probs_t.gather(1, actions_t.unsqueeze(1)).squeeze(1)
        # SyncVectorEnv.step's stub returns unsolved TypeVars (Unknown).
        _, rewards_np, terms_np, truncs_np, _ = cast(
            "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]",
            vec.step(actions_np),
        )
        dones_np = np.logical_or(terms_np, truncs_np)
        # Same-step autoreset: a terminated sub-env's internal state is
        # already its fresh reset draw. Log those draws env-ascending, the
        # order Idris' stepAllAutoResetTrunc consumes them.
        new_states = _internal_states(vec)
        for i in np.flatnonzero(dones_np):
            env_uniforms += _reset_uniforms(new_states[int(i)])
        dtype = obs_t.dtype
        obs_list.append(obs_t)
        act_list.append(actions_t.long())
        lp_list.append(lps_t.detach().to(dtype))
        rew_list.append(torch.tensor(rewards_np, dtype=dtype))
        val_list.append(values.detach().to(dtype))
        done_list.append(torch.tensor(dones_np.astype(np.float64), dtype=dtype))
        states = new_states
    return (
        torch.stack(obs_list),  # [T, N, 6]
        torch.stack(act_list),  # [T, N]
        torch.stack(lp_list),  # [T, N]
        torch.stack(rew_list),  # [T, N]
        torch.stack(val_list),  # [T, N]
        torch.stack(done_list),  # [T, N]
        states,  # [N, 4]
        env_uniforms,
    )


class _RecordingRandom(random.Random):
    """`random.Random` that also keeps each `shuffle`'s resulting permutation.

    `ppo_update` runs unmodified with this as its rng; the recorded
    permutations are read back into replay tags afterwards."""

    def __init__(self, seed: int) -> None:
        super().__init__(seed)
        self.perms: list[list[int]] = []

    def shuffle(self, x: MutableSequence[Any]) -> None:
        super().shuffle(x)
        self.perms.append([int(i) for i in cast("list[int]", list(x))])


def _shuffle_tags(perms: list[list[int]], t_len: int, n_envs: int) -> list[float]:
    """Tags that make the Idris tag-sort shuffle reproduce each recorded
    permutation.

    `Example.Ppo.shuffleIO` draws one uniform per element of its prepped list
    and sorts by it, so any target permutation is reachable by handing element
    `i` the tag `position(i) / total`. The two sides flatten differently —
    Idris env-major (element i = env * T + t), the reference time-major (flat
    index j = t * N + env) — so element i's position is where its time-major
    twin landed in the recorded permutation."""
    tags: list[float] = []
    total = t_len * n_envs
    for perm in perms:
        pos = [0] * total
        for p, j in enumerate(perm):
            pos[j] = p
        for i in range(total):
            env_idx, t = divmod(i, t_len)
            tags.append(pos[t * n_envs + env_idx] / total)
    return tags


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=100, help="number of PPO rollouts")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lambda", dest="lam", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--entropy", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
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
    maybe_dump_init(actor, critic)
    # Single Adam over both nets — the combined-loss composition
    # `ppo_update` applies, matching Idris' one registry-wide optimizer.
    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()),
        lr=args.lr,
    )
    print()

    # Oracle run: collect one rollout, publish the fixture this side started
    # from — the parameters, plus every draw the epoch made, recorded to a
    # replay file — then take exactly one full PPO update (K epochs of
    # minibatches) and publish the result. The Idris side replays the draws
    # through `--replay`: actions land on its `Rng.choice` channel, reset
    # states on its env `Source` (as the uniforms that produced them) and the
    # minibatch permutations on its uniform channel (as tags its tag-sort
    # shuffle orders identically), so its own physics and forward passes
    # regenerate the identical rollout, and GAE, the advantage normalization,
    # the clipped surrogate, the entropy and value terms, the clip and Adam
    # are all under test.
    if os.environ.get("IDRISML_ORACLE_DUMP"):
        vec0, _obs0 = make_acrobot_vec_env(args.seed, NUM_ENVS)
        o, a, lp, r, v, d, final_states, env_uniforms = _oracle_rollout(
            actor, critic, vec0, args.rollout
        )
        with torch.no_grad():
            boot_v = critic(_obs_from_states(final_states))
            boots = torch.where(d[-1] > 0.5, torch.zeros_like(boot_v), boot_v)
        maybe_dump_oracle((actor, critic), PAIRED_PARAMS)
        adv, ret = gae_batched(r, v, d, boots, args.gamma, args.lam)
        flat_adv = adv.reshape(-1)
        flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)
        rec = _RecordingRandom(args.seed)
        ppo_update(
            actor,
            critic,
            optimizer,
            o.reshape(-1, 6),
            a.reshape(-1).long(),
            lp.reshape(-1),
            flat_adv,
            ret.reshape(-1),
            args.clip_eps,
            args.entropy,
            args.value_coef,
            args.k_epochs,
            args.batch_size,
            rec,
        )
        write_replay(
            os.environ["IDRISML_ORACLE_DUMP"] + ".replay",
            env=env_uniforms,
            # [T, N] flattened row-major = the order sampleActionFromBatch
            # consumes choices: per timestep, env 0..N-1. (tolist is
            # list[Unknown] in the torch stub.)
            choices=[int(x) for x in cast("list[int]", a.reshape(-1).tolist())],  # pyright: ignore[reportUnknownMemberType]
            uniforms=_shuffle_tags(rec.perms, args.rollout, NUM_ENVS),
        )
        maybe_dump_after_step((actor, critic), PAIRED_PARAMS)

    vec_env, obs0 = make_acrobot_vec_env(args.seed, NUM_ENVS)
    obs_state = [obs0]
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
            optimizer,
            flat_obs,
            flat_act,
            flat_lp,
            flat_adv,
            flat_ret,
            args.clip_eps,
            args.entropy,
            args.value_coef,
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
        lr_find(LrFindConfig(num_iters=30), epoch_fn, optimizer)
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
