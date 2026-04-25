module Gym.Env

import Data.Vect


||| Type-safe environment interface for reinforcement learning.
|||
||| @state  Internal environment state
||| @action Discrete action type
||| @obs    Observation type (what the agent sees)
public export
interface Env state action obs where
  ||| Initial state for a new episode.
  reset : state
  ||| Advance the environment by one step.
  ||| Returns (reward, next state, done flag).
  step : state -> action -> (Double, state, Bool)
  ||| Extract observation from internal state.
  observe : state -> obs
  ||| Maximum steps per episode.
  maxSteps : Nat
