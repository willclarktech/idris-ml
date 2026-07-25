"""Vector-env autoreset and reset-observation contracts.

idris-gym's `Gym.Vector.stepAutoReset` resets a terminated sub-env within the
same step: the returned obs is the fresh start state and the step's real
reward is kept. gymnasium 1.x defaults SyncVectorEnv to NEXT_STEP autoreset
instead, which inserts a filler transition (action ignored, reward 0) after
every termination — a rollout structure the Idris side cannot produce. Every
reference vec-env constructor must therefore pin SAME_STEP.

Each constructor also returns the reset observation, and the training loops
start their first rollout from it — the Idris side observes its own reset
draw, never a pinned placeholder.
"""

from collections.abc import Callable
from typing import Any, cast

import numpy as np
import pytest
from gymnasium.vector import AutoresetMode, SyncVectorEnv

from torch_ref.models.a2c import make_cartpole_vec_env
from torch_ref.models.dqn import make_cartpole_vec_env as make_dqn_vec_env
from torch_ref.models.mountain_car import make_mountaincar_vec_env
from torch_ref.models.mountain_car_cont import make_mountaincarcont_vec_env
from torch_ref.models.ppo import make_acrobot_vec_env
from torch_ref.models.sac import make_pendulum_vec_env

MakeVec = Callable[[int, int], tuple[SyncVectorEnv, np.ndarray]]

ALL_MAKERS: list[MakeVec] = [
    make_cartpole_vec_env,
    make_dqn_vec_env,
    make_acrobot_vec_env,
    make_mountaincar_vec_env,
    make_mountaincarcont_vec_env,
    make_pendulum_vec_env,
]


@pytest.mark.parametrize("make_vec", ALL_MAKERS)
def test_every_vec_env_pins_same_step_autoreset(make_vec: MakeVec) -> None:
    vec, _obs0 = make_vec(42, 2)
    assert vec.autoreset_mode is AutoresetMode.SAME_STEP


@pytest.mark.parametrize("make_vec", ALL_MAKERS)
def test_constructor_returns_the_reset_observation(make_vec: MakeVec) -> None:
    """The returned obs is the just-reset sub-envs' observation as float64,
    not a pinned placeholder — the first rollout must start from it."""
    _vec, obs0 = make_vec(42, 2)
    assert obs0.dtype == np.float64
    assert obs0.shape[0] == 2
    # Two differently-seeded sub-envs draw different start states; a pinned
    # placeholder (zeros or a tiled constant row) would make the rows equal.
    assert not np.array_equal(obs0[0], obs0[1])


def test_terminating_step_returns_fresh_reset_state() -> None:
    """On the step a sub-env terminates, the real reward is kept and the
    returned obs is the newly-reset state — the `stepAutoReset` contract."""
    vec, obs0 = make_cartpole_vec_env(42, 2)
    # The reset obs matches the sub-envs' internal float64 states (up to the
    # float32 rounding of Gymnasium's observation pipeline).
    states0 = np.array([e.unwrapped.state for e in cast("Any", vec).envs])
    np.testing.assert_allclose(obs0, states0, rtol=1e-6, atol=1e-9)
    for _ in range(500):
        # SyncVectorEnv.step's stub returns unsolved TypeVars (Unknown).
        obs, rew, term, trunc, _info = cast(
            "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]",
            vec.step(np.array([1, 1])),
        )
        done = np.logical_or(term, trunc)
        if done[0]:
            assert rew[0] == 1.0, "terminating step must keep its real reward"
            # The sub-env's internal state is already the fresh reset draw,
            # and the returned obs is that state (a CartPole start state is
            # inside U(-0.05, 0.05)^4; a terminal one is far outside).
            state = np.asarray(cast("Any", vec).envs[0].unwrapped.state, dtype=np.float64)
            np.testing.assert_allclose(obs[0], state, rtol=1e-6)
            assert np.all(np.abs(state) <= 0.05)
            return
    pytest.fail("CartPole never terminated under a constant action")
