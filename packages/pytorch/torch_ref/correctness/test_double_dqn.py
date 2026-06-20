"""Correctness tests for Double DQN on CartPole."""

from torch_ref.models.double_dqn import evaluate, train_double_dqn


class TestDoubleDqn:
    def test_returns_improve(self) -> None:
        """Late returns should be higher than early returns."""
        _, history = train_double_dqn(episodes=300, seed=42, log_every=10000)
        early = sum(history[:50]) / 50
        late = sum(history[-50:]) / 50
        assert late > early + 50, f"Expected large improvement; early={early:.1f} late={late:.1f}"

    def test_converges(self) -> None:
        """Double DQN should solve CartPole on most seeds (greedy eval >= 150).

        Single-seed convergence asserts are fragile for DQN-family methods
        — the policy oscillates and an unlucky seed can land episode 300 in
        a dip. Assert the pass rate across five seeds instead, per the
        multi-seed convergence policy in docs/develop/reference-alignment.md
        (same bar as vanilla DQN's test_converges).
        """
        seeds = (41, 42, 43, 44, 45)
        results: list[float] = []
        for seed in seeds:
            q, _ = train_double_dqn(episodes=300, seed=seed, log_every=10000)
            results.append(evaluate(q, n_episodes=30))
        passes = sum(1 for avg in results if avg >= 150.0)
        assert passes >= 3, f"Expected >=3/5 seeds with avg >= 150, got {passes}/5: {results}"
