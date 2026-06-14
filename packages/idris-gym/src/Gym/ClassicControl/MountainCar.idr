module Gym.ClassicControl.MountainCar

import Data.Vect
import Gym.Env

----------------------------------------------------------------------
-- MountainCar-v0 (Gymnasium-compatible constants)
----------------------------------------------------------------------

MinPosition  : Double; MinPosition = -1.2
MaxPosition  : Double; MaxPosition = 0.6
MaxSpeed     : Double; MaxSpeed    = 0.07
GoalPosition : Double; GoalPosition = 0.5
MCForce      : Double; MCForce       = 0.001
Gravity      : Double; Gravity     = 0.0025

||| MountainCar state: position and velocity.
public export
record MCState where
  constructor MkMC
  mcPos, mcVel : Double

clamp : Double -> Double -> Double -> Double
clamp lo hi x = if x < lo then lo else if x > hi then hi else x

||| One physics step. Action 0 = push left, 1 = no push, 2 = push right.
export
mcStep : MCState -> Nat -> (Double, MCState, Outcome, Info)
mcStep s action =
  let a = cast {to=Double} (cast {to=Integer} action) - 1.0   -- -1, 0, +1
      vel1 = s.mcVel + a * MCForce - prim__doubleCos (3.0 * s.mcPos) * Gravity
      vel2 = clamp (negate MaxSpeed) MaxSpeed vel1
      pos1 = s.mcPos + vel2
      pos2 = clamp MinPosition MaxPosition pos1
      -- If we hit the left wall, velocity resets to 0
      vel3       = if pos2 == MinPosition && vel2 < 0.0 then 0.0 else vel2
      terminated = pos2 >= GoalPosition
  in (-1.0, MkMC pos2 vel3,
      if terminated then Terminated else Continue,
      [])

||| Observation: [position, velocity].
export
mcObserve : MCState -> Vect 2 Double
mcObserve s = [s.mcPos, s.mcVel]

||| Initial state with position drawn uniformly from (-0.6, -0.4) and
||| velocity 0, matching Gymnasium's MountainCar-v0 reset distribution.
export
mcReset : Seed -> (MCState, Seed)
mcReset s0 =
  let (pos, s1) = nextUniform s0 (-0.6) (-0.4)
  in (MkMC pos 0.0, s1)

public export
Env MCState Nat (Vect 2 Double) where
  reset            = mcReset
  step             = mcStep
  observe          = mcObserve
  actionSpace      = Discrete 3
  obsSpace         = Box [MinPosition, negate MaxSpeed] [MaxPosition, MaxSpeed]
  defaultTimeLimit = Just 200
