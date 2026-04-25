"""Correctness tests for Q-learning on CliffWalking."""

from torch_ref.models.q_learning import evaluate, train_q_learning


class TestQLearning:
    def test_returns_improve(self) -> None:
        """Returns should improve from early to late episodes."""
        _, history = train_q_learning(epochs=500, seed=42, log_every=1000)
        assert len(history) == 500
        early_avg = sum(history[:50]) / 50
        late_avg = sum(history[-50:]) / 50
        assert late_avg > early_avg, (
            f"Expected improvement; early={early_avg:.1f} late={late_avg:.1f}"
        )

    def test_converges(self) -> None:
        """Q-learning should find near-optimal greedy policy (optimal ~= -13)."""
        q, _ = train_q_learning(epochs=500, seed=42, log_every=1000)
        avg = evaluate(q, n_episodes=50)
        assert avg >= -20.0, f"Expected avg >= -20, got {avg:.1f}"

    def test_q_table_shape(self) -> None:
        """Q-table should be |S| x |A| = 48 x 4."""
        q, _ = train_q_learning(epochs=50, seed=42, log_every=1000)
        assert q.shape == (48, 4)
