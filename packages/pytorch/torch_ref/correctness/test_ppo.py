"""Correctness tests for PPO on Pendulum."""

from torch_ref.models.ppo import evaluate, train_ppo


class TestPpo:
    def test_returns_improve(self) -> None:
        """Episodic returns should improve substantially from start to end."""
        _, history = train_ppo(total_rollouts=100, log_every=10000)
        assert len(history) > 0
        early = sum(history[:10]) / max(1, min(len(history), 10))
        late = sum(history[-20:]) / max(1, min(len(history), 20))
        assert late > early + 300, (
            f"Expected substantial improvement; early={early:.1f} late={late:.1f}"
        )

    def test_converges(self) -> None:
        """PPO should reach greedy avg_return >= -500 on Pendulum (random ~-1200)."""
        actor, _ = train_ppo(total_rollouts=200, log_every=10000)
        avg = evaluate(actor, n_episodes=10)
        assert avg >= -500.0, f"Expected avg >= -500, got {avg:.1f}"
