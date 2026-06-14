module Test.ClassicControl.MountainCar

import Data.Vect
import Test.Harness
import Gym.Env
import Gym.ClassicControl.MountainCar

rewardOf : (Double, MCState, Outcome, Info) -> Double
rewardOf (r, _, _, _) = r

stateOf : (Double, MCState, Outcome, Info) -> MCState
stateOf (_, s, _, _) = s

outcomeOf : (Double, MCState, Outcome, Info) -> Outcome
outcomeOf (_, _, o, _) = o

mcInit : MCState
mcInit = MkMC (-0.5) 0.0

export
tests : List (IO Bool)
tests =
  [ check "reset pos in Gymnasium U(-0.6, -0.4), vel=0" $
      let (r, _) = reset {state=MCState} {action=Nat} {obs=Vect 2 Double} 42
      in r.mcPos >= -0.6 && r.mcPos <= -0.4 && r.mcVel == 0.0

  , check "reset differs across seeds" $
      let (a, _) = reset {state=MCState} {action=Nat} {obs=Vect 2 Double} 0
          (b, _) = reset {state=MCState} {action=Nat} {obs=Vect 2 Double} 1
      in a.mcPos /= b.mcPos

  , check "step reward -1" $
      rewardOf (mcStep (mcInit) 2) == -1.0

  , check "observe length 2" $
      length (mcObserve (mcInit)) == 2

  , check "push right advances velocity" $
      (stateOf (mcStep (mcInit) 2)).mcVel /= 0.0

  , check "at goal terminates" $
      -- pos well past goal and zero velocity: physics tick leaves pos >= 0.5
      let s  = MkMC 0.55 0.0
          r  = mcStep s 1
      in outcomeOf r == Terminated

  , check "position bounds respected" $
      let s  = MkMC (-1.2) (-0.07)
          s' = stateOf (mcStep s 0)
      in s'.mcPos >= -1.2 && s'.mcPos <= 0.6

  , check "defaultTimeLimit is 200" $
      defaultTimeLimit {state=MCState} {action=Nat} {obs=Vect 2 Double} == Just 200
  ]
