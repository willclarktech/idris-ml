"""Correctness tests for A2C on CartPole."""

from torch_ref.models.a2c import evaluate, train_a2c


class TestA2c:
    def test_returns_improve(self) -> None:
        """Recent-returns history should grow over training."""
        _, history = train_a2c(total_updates=1000, seed=42, log_every=10000)
        assert len(history) == 1000
        # First 100 updates have very short episodes; last 100 should be much longer.
        early = sum(history[:100]) / 100
        late = sum(history[-100:]) / 100
        assert late > early + 30, f"Expected improvement; early={early:.1f} late={late:.1f}"

    def test_converges(self) -> None:
        """A2C should solve CartPole-ish (greedy eval >= 100)."""
        ac, _ = train_a2c(total_updates=1000, seed=42, log_every=10000)
        avg = evaluate(ac, n_episodes=30)
        assert avg >= 100.0, f"Expected avg >= 100, got {avg:.1f}"
