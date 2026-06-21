# idris-gym

Pure-Idris reinforcement-learning environments with a [Gymnasium](https://gymnasium.farama.org/)-parity
API. Classic-control and toy-text domains, composable wrappers, and seeded reproducibility — no
Python, no FFI, just a typed `Env` interface. Depends only on `contrib`.

## The `Env` interface

Every environment implements `Gym.Env`, parameterised over its state, action, and observation
types:

```idris
interface Env state action obs where
  reset       : Seed -> (state, Seed)                          -- env.reset(seed=...)
  step        : state -> action -> (Double, state, Outcome, Info)  -- (reward, next, outcome, info)
  observe     : state -> obs
  actionSpace : Space
  obsSpace    : Space
  defaultTimeLimit : Maybe Nat                                 -- enforced by the TimeLimit wrapper
```

`reset` threads the PRNG `Seed` (returns it advanced) for reproducible rollouts and multi-env
replay buffers; deterministic envs pass it through unchanged. `Outcome` is
`Continue | Terminated | Truncated` — the Gymnasium v0.26+ split that matters for value-function
bootstrapping (bootstrap on the next state for `Truncated`, not `Terminated`).

## Environment catalogue

| Family | Environments |
| --- | --- |
| `Gym.ClassicControl.*` | CartPole, MountainCar, MountainCarCont, Pendulum, Acrobot |
| `Gym.ToyText.*` | CliffWalking, Taxi, FrozenLake, Blackjack |

## Wrappers

Wrappers compose over any `Env` via the interface: `Gym.Wrapper.TimeLimit` (truncation),
`Normalize` (observation scaling), `Record` (episode logging), `Action` (action remapping).
`Gym.Vector` batches independent env instances.

## Usage

Implement `Env` to add your own environment, or drive an existing one in a rollout loop. The
reinforcement-learning examples in [idris-ml-examples](../idris-ml-examples/) (REINFORCE, A2C,
PPO, DQN, SAC, Q-learning, SARSA, …) train [idris-ml](../idris-ml/) policies against these envs —
those are the worked references for the rollout + training-loop pattern.
