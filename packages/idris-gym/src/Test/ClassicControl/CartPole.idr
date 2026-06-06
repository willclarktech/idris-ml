module Test.ClassicControl.CartPole

import Data.Vect
import Test.Harness
import Gym.Env
import Gym.ClassicControl.CartPole


tol : Double
tol = 1.0e-9


-- Extract fields from the 4-tuple for readability.
rewardOf : (Double, CPState, Outcome, Info) -> Double
rewardOf (r, _, _, _) = r

outcomeOf : (Double, CPState, Outcome, Info) -> Outcome
outcomeOf (_, _, o, _) = o

stateOf : (Double, CPState, Outcome, Info) -> CPState
stateOf (_, s, _, _) = s


cpZero : CPState
cpZero = MkCP 0 0 0 0


export
tests : List (IO Bool)
tests =
  [ check "reset components within Gymnasium U(-0.05, 0.05)" $
      let (r, _) = reset {state=CPState} {action=Nat} {obs=Vect 4 Double} 42
      in abs r.cpX <= 0.05 && abs r.cpXDot <= 0.05
         && abs r.cpTheta <= 0.05 && abs r.cpThetaDot <= 0.05

  , check "reset is deterministic for a given seed" $
      let (a, _) = reset {state=CPState} {action=Nat} {obs=Vect 4 Double} 42
          (b, _) = reset {state=CPState} {action=Nat} {obs=Vect 4 Double} 42
      in a.cpX == b.cpX && a.cpXDot == b.cpXDot
         && a.cpTheta == b.cpTheta && a.cpThetaDot == b.cpThetaDot

  , check "reset differs across seeds" $
      let (a, _) = reset {state=CPState} {action=Nat} {obs=Vect 4 Double} 0
          (b, _) = reset {state=CPState} {action=Nat} {obs=Vect 4 Double} 1
      in a.cpX /= b.cpX || a.cpXDot /= b.cpXDot
         || a.cpTheta /= b.cpTheta || a.cpThetaDot /= b.cpThetaDot

  , check "reset advances the seed" $
      let (_, s') = reset {state=CPState} {action=Nat} {obs=Vect 4 Double} 42
      in s' /= 42

  , check "observe length 4" $
      length (cpObserve (cpZero)) == 4

  , check "step reward is 1" $
      rewardOf (cpStep (cpZero) 1) == 1.0

  , check "step does not terminate at zero" $
      outcomeOf (cpStep (cpZero) 1) == Continue

  , check "step advances x" $
      (stateOf (cpStep (cpZero) 1)).cpX /= 0.0
      || (stateOf (cpStep (cpZero) 1)).cpXDot /= 0.0

  , check "extreme x terminates" $
      let s  = MkCP 3.0 0.0 0.0 0.0       -- already past XThresh
          r  = cpStep s 1
      in outcomeOf r == Terminated

  , check "info is empty" $
      case cpStep (cpZero) 0 of
        (_, _, _, info) => info == []

  , check "actionSpace is Discrete 2" $
      case actionSpace {state=CPState} {action=Nat} {obs=Vect 4 Double} of
        Discrete 2 => True
        _ => False

  , check "defaultTimeLimit is 200" $
      defaultTimeLimit {state=CPState} {action=Nat} {obs=Vect 4 Double} == Just 200
  ]
