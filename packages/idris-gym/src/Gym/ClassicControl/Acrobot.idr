module Gym.ClassicControl.Acrobot

import Data.Vect
import Gym.Env


----------------------------------------------------------------------
-- Acrobot-v1 (Gymnasium-compatible)
--
-- Single-step RK4 integration with dt = 0.2, matching the canonical
-- `gymnasium.envs.classic_control.acrobot.rk4` reference. Each `step`
-- advances state by one dt=0.2 RK4 evaluation using four `dsdt` calls.
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
Dt : Double; Dt = 0.2

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

-- Build a candidate state for an RK4 intermediate evaluation:
-- s + scale * (derivative tuple).
shiftBy : Double -> AState -> (Double, Double, Double, Double, Double) -> AState
shiftBy scale s (dth1, dth2, ddth1, ddth2, _) =
  MkA (s.aTh1  + scale * dth1)
      (s.aTh2  + scale * dth2)
      (s.aDth1 + scale * ddth1)
      (s.aDth2 + scale * ddth2)

-- One RK4 step of size Dt = 0.2, matching gymnasium's
-- `rk4(dsdt, state, [0, dt])` call shape:
--   k1 = f(s)
--   k2 = f(s + (Dt/2) * k1)
--   k3 = f(s + (Dt/2) * k2)
--   k4 = f(s + Dt * k3)
--   s' = s + (Dt/6) * (k1 + 2*k2 + 2*k3 + k4)
rk4Step : Double -> AState -> AState
rk4Step torque s =
  let halfDt = Dt / 2.0
      k1 = dsdt torque s
      k2 = dsdt torque (shiftBy halfDt s k1)
      k3 = dsdt torque (shiftBy halfDt s k2)
      k4 = dsdt torque (shiftBy Dt s k3)
      (dth1a, dth2a, ddth1a, ddth2a, _) = k1
      (dth1b, dth2b, ddth1b, ddth2b, _) = k2
      (dth1c, dth2c, ddth1c, ddth2c, _) = k3
      (dth1d, dth2d, ddth1d, ddth2d, _) = k4
      sixthDt = Dt / 6.0
      th1' = s.aTh1  + sixthDt * (dth1a  + 2.0 * dth1b  + 2.0 * dth1c  + dth1d)
      th2' = s.aTh2  + sixthDt * (dth2a  + 2.0 * dth2b  + 2.0 * dth2c  + dth2d)
      dth1' = s.aDth1 + sixthDt * (ddth1a + 2.0 * ddth1b + 2.0 * ddth1c + ddth1d)
      dth2' = s.aDth2 + sixthDt * (ddth2a + 2.0 * ddth2b + 2.0 * ddth2c + ddth2d)
  in MkA th1' th2' dth1' dth2'

||| One physics step. Action 0 = -1 torque, 1 = 0 torque, 2 = +1 torque.
export
aStep : AState -> Nat -> (Double, AState, Outcome, Info)
aStep s action =
  let torque = cast {to=Double} (cast {to=Integer} action) - 1.0
      sRk = rk4Step torque s
      th1 = wrapAngle sRk.aTh1
      th2 = wrapAngle sRk.aTh2
      dth1 = clamp (negate MaxVel1) MaxVel1 sRk.aDth1
      dth2 = clamp (negate MaxVel2) MaxVel2 sRk.aDth2
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
