module Gym.Wrapper.Record

import Gym.Env

----------------------------------------------------------------------
-- Recorded wrapper (episode statistics)
----------------------------------------------------------------------

||| Wraps any environment state with cumulative reward + step count.
||| On Terminated/Truncated, emits final values into Info under keys
||| "episode_return" and "episode_length".
public export
record Recorded state where
  constructor MkRecorded
  inner : state
  totalReward : Double
  epLength : Nat

export
recorded : state -> Recorded state
recorded st = MkRecorded st 0.0 Z

||| Step the inner env, updating accumulators and appending
||| episode_return/episode_length to Info when the episode ends.
export
recordedStep : {state, action, obs : Type} ->
               Env state action obs =>
               Recorded state -> action ->
               (Double, Recorded state, Outcome, Info)
recordedStep (MkRecorded s tot len) act =
  let (r, s', out, info) = step {state} {action} {obs} s act
      tot' = tot + r
      len' = S len
      info' = case out of
                Continue => info
                _ => ("episode_return", show tot') ::
                     ("episode_length", show len') ::
                     info
  in (r, MkRecorded s' tot' len', out, info')

||| Observation from the wrapped state.
export
recordedObserve : {state, action, obs : Type} ->
                  Env state action obs =>
                  Recorded state -> obs
recordedObserve (MkRecorded s _ _) = observe {state} {action} {obs} s
