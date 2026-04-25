module Gym.ClassicControl.Acrobot

import Data.Vect
import Gym.Env


----------------------------------------------------------------------
-- Acrobot-v1 (Gymnasium-compatible constants)
--
-- Gymnasium uses RK4 with dt = 0.2 total. We use semi-implicit Euler
-- with 4 substeps of dt = 0.05 for simpler implementation. Trajectories
-- diverge numerically from the Gymnasium reference but the task and
-- termination condition are identical.
----------------------------------------------------------------------

LinkLen1 : Double; LinkLen1 = 1.0
LinkCom1 : Double; LinkCom1 = 0.5
LinkCom2 : Double; LinkCom2 = 0.5
LinkMass1 : Double; LinkMass1 = 1.0
LinkMass2 : Double; LinkMass2 = 1.0
LinkMOI : Double; LinkMOI = 1.0
MaxVel1 : Double; MaxVel1 = 4.0 * 3.141592653589793
MaxVel2 : Double; MaxVel2 = 9.0 * 3.141592653589793
Gravity : Double; Gravity = 9.8
Pi : Double; Pi = 3.141592653589793
Dt : Double; Dt = 0.05

||| Acrobot state: two joint angles and angular velocities.
public export
record AState where
  constructor MkA
  aTh1, aTh2, aDth1, aDth2 : Double


clamp : Double -> Double -> Double -> Double
clamp lo hi x = if x < lo then lo else if x > hi then hi else x

||| Wrap an angle to [-pi, pi].
wrapAngle : Double -> Double
wrapAngle x =
  let twoPi = 2.0 * Pi
  in x - twoPi * prim__doubleFloor ((x + Pi) / twoPi)

-- dtheta1, dtheta2, ddtheta1, ddtheta2 from the augmented state and torque.
dsdt : Double -> AState -> (Double, Double, Double, Double, Double)
dsdt torque s =
  let m1 = LinkMass1
      m2 = LinkMass2
      l1 = LinkLen1
      lc1 = LinkCom1
      lc2 = LinkCom2
      i1 = LinkMOI
      i2 = LinkMOI
      th1 = s.aTh1
      th2 = s.aTh2
      dth1 = s.aDth1
      dth2 = s.aDth2
      cosTh2 = prim__doubleCos th2
      sinTh2 = prim__doubleSin th2
      d1 = m1 * lc1 * lc1
         + m2 * (l1 * l1 + lc2 * lc2 + 2.0 * l1 * lc2 * cosTh2)
         + i1 + i2
      d2 = m2 * (lc2 * lc2 + l1 * lc2 * cosTh2) + i2
      phi2 = m2 * lc2 * Gravity * prim__doubleCos (th1 + th2 - Pi / 2.0)
      phi1 = negate (m2 * l1 * lc2 * dth2 * dth2 * sinTh2)
           - 2.0 * m2 * l1 * lc2 * dth2 * dth1 * sinTh2
           + (m1 * lc1 + m2 * l1) * Gravity * prim__doubleCos (th1 - Pi / 2.0)
           + phi2
      ddth2 = (torque + d2 / d1 * phi1
             - m2 * l1 * lc2 * dth1 * dth1 * sinTh2 - phi2)
             / (m2 * lc2 * lc2 + i2 - d2 * d2 / d1)
      ddth1 = negate ((d2 * ddth2 + phi1) / d1)
  in (dth1, dth2, ddth1, ddth2, 0.0)

-- One semi-implicit Euler substep.
eulerStep : Double -> AState -> AState
eulerStep torque s =
  let (_, _, ddth1, ddth2, _) = dsdt torque s
      dth1' = s.aDth1 + Dt * ddth1
      dth2' = s.aDth2 + Dt * ddth2
      th1'  = s.aTh1 + Dt * dth1'
      th2'  = s.aTh2 + Dt * dth2'
  in MkA th1' th2' dth1' dth2'

||| One physics step. Action 0 = -1 torque, 1 = 0 torque, 2 = +1 torque.
export
aStep : AState -> Nat -> (Double, AState, Outcome, Info)
aStep s action =
  let torque = cast {to=Double} (cast {to=Integer} action) - 1.0
      s1 = eulerStep torque s
      s2 = eulerStep torque s1
      s3 = eulerStep torque s2
      s4 = eulerStep torque s3
      th1 = wrapAngle s4.aTh1
      th2 = wrapAngle s4.aTh2
      dth1 = clamp (negate MaxVel1) MaxVel1 s4.aDth1
      dth2 = clamp (negate MaxVel2) MaxVel2 s4.aDth2
      s' = MkA th1 th2 dth1 dth2
      terminated = negate (prim__doubleCos th1)
                 - prim__doubleCos (th2 + th1) > 1.0
      reward = if terminated then 0.0 else -1.0
  in (reward, s',
      if terminated then Terminated else Continue,
      [])

||| Observation: [cos(theta1), sin(theta1), cos(theta2), sin(theta2), dtheta1, dtheta2].
export
aObserve : AState -> Vect 6 Double
aObserve s =
  [ prim__doubleCos s.aTh1
  , prim__doubleSin s.aTh1
  , prim__doubleCos s.aTh2
  , prim__doubleSin s.aTh2
  , s.aDth1
  , s.aDth2
  ]

public export
Env AState Nat (Vect 6 Double) where
  reset = MkA 0.0 0.0 0.0 0.0
  step = aStep
  observe = aObserve
  actionSpace = Discrete 3
  obsSpace = Box [-1.0, -1.0, -1.0, -1.0, negate MaxVel1, negate MaxVel2]
                 [ 1.0,  1.0,  1.0,  1.0, MaxVel1, MaxVel2]
  defaultTimeLimit = Just 500
