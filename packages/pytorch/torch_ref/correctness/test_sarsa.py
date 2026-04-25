"""Correctness tests for SARSA on CliffWalking."""

from torch_ref.models.sarsa import evaluate, train_sarsa


class TestSarsa:
    def test_returns_improve(self) -> None:
        """Returns should improve from early to late episodes."""
        _, history = train_sarsa(epochs=1000, seed=42, log_every=10000)
        assert len(history) == 1000
        early_avg = sum(history[:50]) / 50
        late_avg = sum(history[-50:]) / 50
        assert late_avg > early_avg, (
            f"Expected improvement; early={early_avg:.1f} late={late_avg:.1f}"
        )

    def test_converges(self) -> None:
        """SARSA should find a near-optimal greedy policy (safer path ≈ -15 to -17)."""
        q, _ = train_sarsa(epochs=1000, seed=42, log_every=10000)
        avg = evaluate(q, n_episodes=50)
        assert avg >= -20.0, f"Expected avg >= -20, got {avg:.1f}"
