module Gym.ClassicControl.Pendulum

import Data.Vect
import Gym.Env


----------------------------------------------------------------------
-- Pendulum-v1 (Gymnasium-compatible constants)
----------------------------------------------------------------------

Gravity    : Double; Gravity    = 10.0
MassPole   : Double; MassPole   = 1.0
PoleLen    : Double; PoleLen    = 1.0
MaxTorque  : Double; MaxTorque  = 2.0
MaxSpeed   : Double; MaxSpeed   = 8.0
Dt         : Double; Dt         = 0.05
Pi         : Double; Pi         = 3.141592653589793

||| Internal state: angle (theta) in radians, angular velocity.
public export
record PState where
  constructor MkP
  pTheta, pThetaDot : Double


clamp : Double -> Double -> Double -> Double
clamp lo hi x = if x < lo then lo else if x > hi then hi else x

||| Wrap an angle to [-pi, pi].
angleNormalize : Double -> Double
angleNormalize x =
  let twoPi = 2.0 * Pi
      wrapped = x - twoPi * prim__doubleFloor ((x + Pi) / twoPi)
  in wrapped

||| One step. Action is torque in [-2.0, 2.0].
export
pStep : PState -> Double -> (Double, PState, Outcome, Info)
pStep s action =
  let torque = clamp (negate MaxTorque) MaxTorque action
      th = s.pTheta
      dth = s.pThetaDot
      -- Use the angle-normalized theta for the reward only.
      thNorm = angleNormalize th
      reward = negate (thNorm * thNorm
                     + 0.1 * dth * dth
                     + 0.001 * torque * torque)
      -- Dynamics (matches Gymnasium Pendulum-v1)
      dth1 = dth + (3.0 * Gravity / (2.0 * PoleLen) * prim__doubleSin th
                  + 3.0 / (MassPole * PoleLen * PoleLen) * torque) * Dt
      dth2 = clamp (negate MaxSpeed) MaxSpeed dth1
      th1 = th + dth2 * Dt
  in (reward, MkP th1 dth2, Continue, [])

||| Observation: [cos(theta), sin(theta), theta_dot].
export
pObserve : PState -> Vect 3 Double
pObserve s = [ prim__doubleCos s.pTheta
             , prim__doubleSin s.pTheta
             , s.pThetaDot
             ]

public export
Env PState Double (Vect 3 Double) where
  reset = MkP Pi 0.0   -- hangs-down start (Gymnasium randomizes; we pick the extreme)
  step = pStep
  observe = pObserve
  actionSpace = Box [negate MaxTorque] [MaxTorque]
  obsSpace = Box [-1.0, -1.0, negate MaxSpeed] [1.0, 1.0, MaxSpeed]
  defaultTimeLimit = Just 200
