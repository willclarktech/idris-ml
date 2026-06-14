module Gym.ClassicControl.MountainCarCont

import Data.Vect
import Gym.Env

----------------------------------------------------------------------
-- MountainCarContinuous-v0 (Gymnasium-compatible constants)
----------------------------------------------------------------------

MinPosition  : Double; MinPosition   = -1.2
MaxPosition  : Double; MaxPosition   = 0.6
MaxSpeed     : Double; MaxSpeed      = 0.07
GoalPosition : Double; GoalPosition  = 0.45
GoalVelocity : Double; GoalVelocity  = 0.0
Power        : Double; Power         = 0.0015
MinAction    : Double; MinAction     = -1.0
MaxAction    : Double; MaxAction     = 1.0

public export
record MCCState where
  constructor MkMCC
  mccPos, mccVel : Double

clamp : Double -> Double -> Double -> Double
clamp lo hi x = if x < lo then lo else if x > hi then hi else x

||| One physics step. Action is a scalar force in [-1.0, 1.0].
export
mccStep : MCCState -> Double -> (Double, MCCState, Outcome, Info)
mccStep s action =
  let clippedAction = clamp MinAction MaxAction action
      force = clippedAction * Power
      vel1 = s.mccVel + force - 0.0025 * prim__doubleCos (3.0 * s.mccPos)
      vel2 = clamp (negate MaxSpeed) MaxSpeed vel1
      pos1 = s.mccPos + vel2
      pos2 = clamp MinPosition MaxPosition pos1
      vel3 = if pos2 == MinPosition && vel2 < 0.0 then 0.0 else vel2
      terminated = pos2 >= GoalPosition && vel3 >= GoalVelocity
      reward = (if terminated then 100.0 else 0.0)
             - 0.1 * clippedAction * clippedAction
  in (reward, MkMCC pos2 vel3,
      if terminated then Terminated else Continue,
      [])

export
mccObserve : MCCState -> Vect 2 Double
mccObserve s = [s.mccPos, s.mccVel]

||| Initial state with position drawn uniformly from (-0.6, -0.4) and
||| velocity 0, matching Gymnasium's MountainCarContinuous-v0 reset
||| distribution.
export
mccReset : Seed -> (MCCState, Seed)
mccReset s0 =
  let (pos, s1) = nextUniform s0 (-0.6) (-0.4)
  in (MkMCC pos 0.0, s1)

public export
Env MCCState Double (Vect 2 Double) where
  reset = mccReset
  step = mccStep
  observe = mccObserve
  actionSpace = Box [MinAction] [MaxAction]
  obsSpace = Box [MinPosition, negate MaxSpeed] [MaxPosition, MaxSpeed]
  defaultTimeLimit = Just 999
