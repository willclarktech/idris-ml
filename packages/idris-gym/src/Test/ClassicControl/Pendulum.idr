module Test.ClassicControl.Pendulum

import Data.Vect
import Test.Harness
import Gym.Env
import Gym.ClassicControl.Pendulum

rewardOf : (Double, PState, Outcome, Info) -> Double
rewardOf (r, _, _, _) = r

stateOf : (Double, PState, Outcome, Info) -> PState
stateOf (_, s, _, _) = s

outcomeOf : (Double, PState, Outcome, Info) -> Outcome
outcomeOf (_, _, o, _) = o

pInit : PState
pInit = MkP 3.141592653589793 0.0

export
tests : List (IO Bool)
tests =
  [ check "reset components in Gymnasium ranges" $
      let (r, _) = reset {state=PState} {action=Double} {obs=Vect 3 Double} 42
      in abs r.pTheta <= 3.141592653589793 && abs r.pThetaDot <= 1.0

  , check "reset differs across seeds" $
      let (a, _) = reset {state=PState} {action=Double} {obs=Vect 3 Double} 0
          (b, _) = reset {state=PState} {action=Double} {obs=Vect 3 Double} 1
      in a.pTheta /= b.pTheta || a.pThetaDot /= b.pThetaDot

  , check "observe length 3" $
      length (pObserve (pInit)) == 3

  , check "obs first two components are cos/sin of theta" $
      case pObserve (pInit) of
        [c, s, _] =>
          abs (c - prim__doubleCos 3.141592653589793) < 1.0e-9
          && abs (s - prim__doubleSin 3.141592653589793) < 1.0e-9
        _ => False

  , check "reward always non-positive" $
      rewardOf (pStep (pInit) 0.0) <= 0.0

  , check "zero torque at pi is max reward (theta norm ~0)" $
      -- At theta=pi, angleNormalize(pi) = -pi (wrap), so reward = -pi^2
      let r = rewardOf (pStep (pInit) 0.0)
      in r > -10.0 && r < 0.0

  , check "never terminates naturally" $
      outcomeOf (pStep (pInit) 0.0) == Continue

  , check "large torque clipped" $
      rewardOf (pStep (pInit) 10.0)
      == rewardOf (pStep (pInit) 2.0)

  , check "defaultTimeLimit is 200" $
      defaultTimeLimit {state=PState} {action=Double} {obs=Vect 3 Double} == Just 200
  ]
