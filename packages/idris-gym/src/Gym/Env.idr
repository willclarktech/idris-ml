module Gym.Env

import public Gym.Rng
import public Gym.Space

----------------------------------------------------------------------
-- Outcome
----------------------------------------------------------------------

||| How an episode step concluded.
||| Continue  = episode still running.
||| Terminated = natural end (pole fell, goal reached, agent died).
||| Truncated = artificial end (time limit, wrapper intervention).
|||
||| The split matches Gymnasium v0.26+: value-function bootstrapping
||| uses the next state on Truncated but not on Terminated.
public export
data Outcome = Continue | Terminated | Truncated

||| Is the episode over, for either reason?
public export
done : Outcome -> Bool
done Continue = False
done _        = True

public export
Eq Outcome where
  Continue   == Continue   = True
  Terminated == Terminated = True
  Truncated  == Truncated  = True
  _ == _                   = False

public export
Show Outcome where
  show Continue   = "Continue"
  show Terminated = "Terminated"
  show Truncated  = "Truncated"

----------------------------------------------------------------------
-- Info
----------------------------------------------------------------------

||| Auxiliary key-value info returned alongside a step result.
||| Matches Gymnasium's info dict but uses plain strings for pragmatism
||| (Train.idr's formatResult uses the same List (String, String) shape).
public export
Info : Type
Info = List (String, String)

----------------------------------------------------------------------
-- Env interface
----------------------------------------------------------------------

||| Type-safe environment interface for reinforcement learning.
|||
||| @state  Internal environment state (deterministic envs: physics only;
|||         stochastic envs: include a PRNG seed field).
||| @action Action type (Nat for Discrete, Double / Vect k Double for Box).
||| @obs    Observation type (what the agent sees).
public export
interface Env state action obs where
  ||| Initial state for a new episode. Takes a Seed and returns the
  ||| initial state plus the advanced Seed. Matches Gymnasium's
  ||| `env.reset(seed=...)` contract: each call consumes randomness
  ||| from the caller-side PRNG and returns it advanced. Deterministic
  ||| envs (e.g. CliffWalking) pass the Seed through unchanged.
  reset : Seed -> (state, Seed)
  ||| Advance the environment by one step.
  ||| Returns (reward, next state, outcome, info).
  step : state -> action -> (Double, state, Outcome, Info)
  ||| Extract observation from internal state.
  observe : state -> obs
  ||| Descriptor of the action space (metadata for wrappers/loggers).
  actionSpace : Space
  ||| Descriptor of the observation space.
  obsSpace : Space
  ||| Default truncation step count, if this env has a standard one.
  ||| Truncation itself is enforced by the TimeLimited wrapper, not here.
  defaultTimeLimit : Maybe Nat
  defaultTimeLimit = Nothing
