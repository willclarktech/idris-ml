"""Correctness tests for first-visit MC on Blackjack."""

from torch_ref.models.monte_carlo import evaluate, train_mc


class TestMonteCarlo:
    def test_converges(self) -> None:
        """MC control should find a near-optimal policy (win rate >= 0.38; random ~0.28)."""
        q, _ = train_mc(epochs=50000, seed=42, log_every=100000)
        stats = evaluate(q, n_episodes=5000)
        assert stats["win"] >= 0.38, f"Expected win_rate >= 0.38, got {stats['win']:.3f}"

    def test_win_rate_beats_random(self) -> None:
        """Learned policy should beat random (random ~0.28 win)."""
        q, _ = train_mc(epochs=20000, seed=42, log_every=100000)
        stats = evaluate(q, n_episodes=2000)
        assert stats["win"] > 0.32, f"Expected win > 0.32, got {stats['win']:.3f}"
