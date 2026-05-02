"""Correctness tests for PPO on Acrobot."""

from torch_ref.models.ppo import evaluate, train_ppo


class TestPpo:
    def test_returns_improve(self) -> None:
        """Episodic returns should improve substantially from start to end."""
        _, history = train_ppo(total_rollouts=50, log_every=10000)
        assert len(history) > 0
        early = sum(history[:5]) / max(1, min(len(history), 5))
        late = sum(history[-10:]) / max(1, min(len(history), 10))
        # Acrobot returns range from ~-500 (random/timeout) to ~-100 (solved).
        # Early returns are typically -500 or -400; late returns should be
        # at least 100 better at convergence.
        assert late > early + 100, (
            f"Expected substantial improvement; early={early:.1f} late={late:.1f}"
        )

    def test_converges(self) -> None:
        """PPO should reach greedy avg_return >= -200 on Acrobot (random ~-500)."""
        actor, _ = train_ppo(total_rollouts=100, log_every=10000)
        avg = evaluate(actor, n_episodes=10)
        assert avg >= -200.0, f"Expected avg >= -200, got {avg:.1f}"
