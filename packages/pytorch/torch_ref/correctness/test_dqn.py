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
        """DQN should solve CartPole (greedy eval >= 150)."""
        q, _ = train_dqn(episodes=300, seed=42, log_every=10000)
        avg = evaluate(q, n_episodes=30)
        assert avg >= 150.0, f"Expected avg >= 150, got {avg:.1f}"

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
