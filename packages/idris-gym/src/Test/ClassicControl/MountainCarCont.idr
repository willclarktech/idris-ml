module Test.ClassicControl.MountainCarCont

import Data.Vect

import Gym.ClassicControl.MountainCarCont
import Gym.Env
import Test.Harness

rewardOf : (Double, MCCState, Outcome, Info) -> Double
rewardOf (r, _, _, _) = r

stateOf : (Double, MCCState, Outcome, Info) -> MCCState
stateOf (_, s, _, _) = s

outcomeOf : (Double, MCCState, Outcome, Info) -> Outcome
outcomeOf (_, _, o, _) = o

mccInit : MCCState
mccInit = MkMCC (-0.5) 0.0

export
tests : List (IO Bool)
tests =
  [ check "reset pos in Gymnasium U(-0.6, -0.4), vel=0" $
      let (r, _) = reset {state=MCCState} {action=Double} {obs=Vect 2 Double} (Seeded 42)
      in r.mccPos >= -0.6 && r.mccPos <= -0.4 && r.mccVel == 0.0

  , check "reset differs across seeds" $
      let (a, _) = reset {state=MCCState} {action=Double} {obs=Vect 2 Double} (Seeded 0)
          (b, _) = reset {state=MCCState} {action=Double} {obs=Vect 2 Double} (Seeded 1)
      in a.mccPos /= b.mccPos

  , check "observe length 2" $
      length (mccObserve (mccInit)) == 2

  , check "reward is negative for non-zero action" $
      rewardOf (mccStep (mccInit) 0.5) < 0.0

  , check "zero action gives zero quadratic cost" $
      rewardOf (mccStep (mccInit) 0.0) == 0.0

  , check "at goal with positive velocity terminates" $
      let s  = MkMCC 0.45 0.01
          r  = mccStep s 0.0
      in outcomeOf r == Terminated

  , check "at goal with positive velocity terminates (>=)" $
      -- small push keeps velocity >= 0 and position >= goal after step
      let s  = MkMCC 0.5 0.001
          r  = mccStep s 0.5
      in outcomeOf r == Terminated

  , check "action clipped to bounds" $
      -- large positive action should behave like action=1.0
      rewardOf (mccStep (mccInit) 10.0)
      == rewardOf (mccStep (mccInit) 1.0)

  , check "defaultTimeLimit is 999" $
      defaultTimeLimit {state=MCCState} {action=Double} {obs=Vect 2 Double} == Just 999
  ]
