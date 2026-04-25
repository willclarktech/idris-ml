module Gym.Vector

import Data.Vect
import Gym.Env


----------------------------------------------------------------------
-- VecEnv: n independent copies of an env running in lockstep
----------------------------------------------------------------------

||| A batch of n independent env states. "Parallel" here is just a map
||| — pure Idris has no process-level threads. Matches Gymnasium's
||| SyncVectorEnv semantics.
public export
record VecEnv (n : Nat) state where
  constructor MkVecEnv
  envs : Vect n state

||| Reset all environments to the Env's reset state.
export
resetAll : {n : Nat} ->
           {state, action, obs : Type} ->
           Env state action obs =>
           VecEnv n state
resetAll = MkVecEnv (replicate n (reset {state} {action} {obs}))

||| Step every environment with its corresponding action.
||| Returns a Vect of per-env step tuples.
export
stepAll : {n : Nat} ->
          {state, action, obs : Type} ->
          Env state action obs =>
          VecEnv n state -> Vect n action ->
          (VecEnv n state, Vect n (Double, Outcome, Info))
stepAll (MkVecEnv ss) acts =
  let results = zipWith (\s, a => step {state} {action} {obs} s a) ss acts
      newStates = map (\(_, s', _, _) => s') results
      trimmed   = map (\(r, _, o, i) => (r, o, i)) results
  in (MkVecEnv newStates, trimmed)

||| Step every env and auto-reset any that ended. The "observation"
||| returned for a reset env is the observation of the newly-reset state.
export
stepAutoReset : {n : Nat} ->
                {state, action, obs : Type} ->
                Env state action obs =>
                VecEnv n state -> Vect n action ->
                (VecEnv n state, Vect n Double, Vect n obs, Vect n Outcome)
stepAutoReset (MkVecEnv ss) acts =
  let tuples = zipWith (\s, a => step {state} {action} {obs} s a) ss acts
      advanced = map resetIfDone tuples
      newStates = map fst advanced
      outs      = map (\(s, _, _) => observe {state} {action} {obs} s) advanced
      rewards   = map (\(_, r, _) => r) advanced
      outcomes  = map (\(_, _, o) => o) advanced
  in (MkVecEnv newStates, rewards, outs, outcomes)
  where
    resetIfDone : (Double, state, Outcome, Info) ->
                  (state, Double, Outcome)
    resetIfDone (r, s', out, _) =
      case out of
        Continue => (s', r, out)
        _        => (reset {state} {action} {obs}, r, out)
