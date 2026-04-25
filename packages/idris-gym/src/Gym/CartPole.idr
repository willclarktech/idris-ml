module Gym.CartPole

import Data.Vect
import Gym.Env


----------------------------------------------------------------------
-- CartPole-v1 (Gymnasium-compatible constants)
----------------------------------------------------------------------

Gravity : Double;       Gravity = 9.8
MassCart : Double;      MassCart = 1.0
MassPole : Double;      MassPole = 0.1
TotalMass : Double;     TotalMass = MassCart + MassPole
HalfPoleLen : Double;   HalfPoleLen = 0.5
PoleMassLen : Double;   PoleMassLen = MassPole * HalfPoleLen
ForceMag : Double;      ForceMag = 10.0
Tau : Double;           Tau = 0.02
ThetaThresh : Double;   ThetaThresh = 12.0 * 2.0 * 3.141592653589793 / 360.0
XThresh : Double;       XThresh = 2.4

||| CartPole internal state: position, velocity, angle, angular velocity.
public export
record CPState where
  constructor MkCP
  cpX, cpXDot, cpTheta, cpThetaDot : Double

||| One physics step of CartPole. Action 0 = left, 1 = right.
export
cpStep : CPState -> Nat -> (Double, CPState, Bool)
cpStep s action =
  let force = if action == 1 then ForceMag else negate ForceMag
      cosT = prim__doubleCos s.cpTheta
      sinT = prim__doubleSin s.cpTheta
      temp = (force + PoleMassLen * s.cpThetaDot * s.cpThetaDot * sinT) / TotalMass
      tAcc = (Gravity * sinT - cosT * temp) /
             (HalfPoleLen * (4.0 / 3.0 - MassPole * cosT * cosT / TotalMass))
      xAcc = temp - PoleMassLen * tAcc * cosT / TotalMass
      s' = MkCP (s.cpX + Tau * s.cpXDot) (s.cpXDot + Tau * xAcc)
                (s.cpTheta + Tau * s.cpThetaDot) (s.cpThetaDot + Tau * tAcc)
  in (1.0, s', abs s'.cpX > XThresh || abs s'.cpTheta > ThetaThresh)

||| Extract 4-element observation vector from CartPole state.
export
cpObserve : CPState -> Vect 4 Double
cpObserve s = [s.cpX, s.cpXDot, s.cpTheta, s.cpThetaDot]

||| Maximum steps per CartPole episode (matches Gymnasium CartPole-v1).
public export
cartPoleMaxSteps : Nat
cartPoleMaxSteps = 200

public export
Env CPState Nat (Vect 4 Double) where
  reset = MkCP 0 0 0 0
  step = cpStep
  observe = cpObserve
  maxSteps = cartPoleMaxSteps
