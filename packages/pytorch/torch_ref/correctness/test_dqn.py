"""Correctness tests for DQN on CartPole."""

from torch_ref.models.dqn import ReplayBuffer, evaluate, train_dqn


class TestDqn:
    def test_returns_improve(self) -> None:
        """Late returns should be higher than early returns."""
        _, history = train_dqn(episodes=300, seed=42, log_every=10000)
        early = sum(history[:50]) / 50
        late = sum(history[-50:]) / 50
        assert late > early + 50, f"Expected large improvement; early={early:.1f} late={late:.1f}"

    def test_converges(self) -> None:
        """DQN should solve CartPole on most seeds (greedy eval >= 150).

        Single-seed convergence asserts are fragile for DQN — the
        policy oscillates and an unlucky seed can land episode 300 in
        a dip: measured 2026-06-10, seeds 41/43/44/45 all hit the 200
        eval cap while seed 42 lands at 132. Assert the pass rate
        across five seeds instead (measured 4/5), per the multi-seed
        convergence policy in docs/develop/reference-alignment.md.
        """
        seeds = (41, 42, 43, 44, 45)
        results: list[float] = []
        for seed in seeds:
            q, _ = train_dqn(episodes=300, seed=seed, log_every=10000)
            results.append(evaluate(q, n_episodes=30))
        passes = sum(1 for avg in results if avg >= 150.0)
        assert passes >= 3, f"Expected >=3/5 seeds with avg >= 150, got {passes}/5: {results}"

    def test_buffer_ring(self) -> None:
        """ReplayBuffer should overwrite oldest when full."""
        import random

        buf = ReplayBuffer(capacity=3)
        for i in range(5):
            buf.push([float(i)], i, float(i), [float(i + 1)], False)
        assert len(buf) == 3
        obs, _, _, _, _ = buf.sample(3, random.Random(0))
        # Values 0, 1 should be gone; {2, 3, 4} remain.
        vals = {int(o[0]) for o in obs}
        assert vals.issubset({2, 3, 4})
