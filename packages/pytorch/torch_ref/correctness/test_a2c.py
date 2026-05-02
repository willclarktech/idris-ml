"""Correctness tests for A2C on CartPole."""

from torch_ref.models.a2c import evaluate, train_a2c


class TestA2c:
    def test_returns_improve(self) -> None:
        """Recent-returns history should grow over training."""
        _actor, _critic, history = train_a2c(total_updates=1000, seed=42, log_every=10000)
        assert len(history) >= 10
        # First few episodes are short; later ones should be much longer.
        n = max(1, len(history) // 4)
        early = sum(history[:n]) / n
        late = sum(history[-n:]) / n
        assert late > early + 30, f"Expected improvement; early={early:.1f} late={late:.1f}"

    def test_converges(self) -> None:
        """A2C should solve CartPole-ish (greedy eval >= 100)."""
        actor, _critic, _history = train_a2c(total_updates=1000, seed=42, log_every=10000)
        avg = evaluate(actor, n_episodes=30)
        assert avg >= 100.0, f"Expected avg >= 100, got {avg:.1f}"
