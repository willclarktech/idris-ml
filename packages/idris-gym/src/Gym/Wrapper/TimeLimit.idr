module Gym.Wrapper.TimeLimit

import Gym.Env

----------------------------------------------------------------------
-- TimeLimited wrapper
----------------------------------------------------------------------

||| Wraps any environment state with a step-count truncation limit.
||| On step, elapsed increments; Continue becomes Truncated when
||| elapsed >= limit. Terminated from the inner env is preserved.
|||
||| Wrappers are exposed as helper functions rather than Env instances
||| to sidestep name-shadowing in nested interface resolution. Callers
||| thread the wrapper state manually via timeLimitedStep.
public export
record TimeLimited state where
  constructor MkTimeLimited
  inner   : state
  elapsed : Nat
  limit   : Nat

||| Build a wrapped state from an inner state and explicit step limit.
||| limit = 0 disables truncation.
export
timeLimited : Nat -> state -> TimeLimited state
timeLimited lim st = MkTimeLimited st Z lim

||| Build a wrapped state using the env's defaultTimeLimit.
||| Falls back to the provided default when the env reports Nothing.
export
withDefaultTimeLimit : Env state action obs =>
                       Nat -> state -> TimeLimited state
withDefaultTimeLimit fallback st =
  let lim = case defaultTimeLimit {state} {action} {obs} of
              Just n  => n
              Nothing => fallback
  in MkTimeLimited st Z lim

||| Step the inner env and apply truncation.
export
timeLimitedStep : {state, action, obs : Type} ->
                  Env state action obs =>
                  TimeLimited state -> action ->
                  (Double, TimeLimited state, Outcome, Info)
timeLimitedStep (MkTimeLimited s e lim) act =
  let (r, s', out, info) = step {state} {action} {obs} s act
      e'   = S e
      out' = case out of
               Continue => if lim > 0 && e' >= lim then Truncated else Continue
               o        => o
  in (r, MkTimeLimited s' e' lim, out', info)

||| Extract the observation from a wrapped state.
export
timeLimitedObserve : {state, action, obs : Type} ->
                     Env state action obs =>
                     TimeLimited state -> obs
timeLimitedObserve (MkTimeLimited s _ _) = observe {state} {action} {obs} s
