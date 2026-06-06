module Gym.Vector

import Data.Vect
import Gym.Env
import Gym.Rng


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

||| Reset all environments, threading the caller-side Seed through n
||| sub-resets. Returns the populated VecEnv and the advanced Seed.
export
resetAll : {n : Nat} ->
           {state, action, obs : Type} ->
           Env state action obs =>
           Seed -> (VecEnv n state, Seed)
resetAll {n=Z}   seed = (MkVecEnv [], seed)
resetAll {n=S k} seed =
  let (s,  seed')  = reset {state} {action} {obs} seed
      (vk, seed'') = resetAll {n=k} {state} {action} {obs} seed'
  in (MkVecEnv (s :: vk.envs), seed'')

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
||| Threads a Seed through the per-env auto-resets (advanced once per
||| terminated env).
export
stepAutoReset : {n : Nat} ->
                {state, action, obs : Type} ->
                Env state action obs =>
                Seed -> VecEnv n state -> Vect n action ->
                (VecEnv n state, Vect n Double, Vect n obs, Vect n Outcome, Seed)
stepAutoReset seed0 (MkVecEnv ss) acts =
  let tuples = zipWith (\s, a => step {state} {action} {obs} s a) ss acts
      (newStates, outs, rewards, outcomes, seed') = walk seed0 tuples
  in (MkVecEnv newStates, rewards, outs, outcomes, seed')
  where
    walk : {k : Nat} -> Seed -> Vect k (Double, state, Outcome, Info) ->
           (Vect k state, Vect k obs, Vect k Double, Vect k Outcome, Seed)
    walk seed [] = ([], [], [], [], seed)
    walk seed ((r, s', out, _) :: rest) =
      let (s'', seedNext) =
            case out of
              Continue => (s', seed)
              _        => reset {state} {action} {obs} seed
          o = observe {state} {action} {obs} s''
          (ss', os, rs, outs, seedEnd) = walk seedNext rest
      in (s'' :: ss', o :: os, r :: rs, out :: outs, seedEnd)
