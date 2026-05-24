module Test.ClassicControl.Acrobot

import Data.Vect
import Test.Harness
import Gym.Env
import Gym.ClassicControl.Acrobot


rewardOf : (Double, AState, Outcome, Info) -> Double
rewardOf (r, _, _, _) = r

stateOf : (Double, AState, Outcome, Info) -> AState
stateOf (_, s, _, _) = s

outcomeOf : (Double, AState, Outcome, Info) -> Outcome
outcomeOf (_, _, o, _) = o


aInit : AState
aInit = MkA 0.0 0.0 0.0 0.0


export
tests : List (IO Bool)
tests =
  [ check "reset all zero" $
      let r : AState
          r = reset {state=AState} {action=Nat} {obs=Vect 6 Double}
      in r.aTh1 == 0.0 && r.aTh2 == 0.0
         && r.aDth1 == 0.0 && r.aDth2 == 0.0

  , check "observe length 6" $
      length (aObserve (aInit)) == 6

  , check "reward is -1 on non-goal" $
      rewardOf (aStep (aInit) 2) == -1.0

  , check "at reset (hanging) does not terminate" $
      outcomeOf (aStep (aInit) 1) == Continue

  , check "obs has cos/sin structure" $
      case aObserve (MkA 0.0 0.0 0.0 0.0) of
        [c1, s1, c2, s2, _, _] =>
          abs (c1 - 1.0) < 1.0e-9 && abs s1 < 1.0e-9
          && abs (c2 - 1.0) < 1.0e-9 && abs s2 < 1.0e-9
        _ => False

  , check "defaultTimeLimit is 500" $
      defaultTimeLimit {state=AState} {action=Nat} {obs=Vect 6 Double} == Just 500
  ]
