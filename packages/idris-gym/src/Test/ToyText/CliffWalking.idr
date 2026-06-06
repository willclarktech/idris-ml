module Test.ToyText.CliffWalking

import Test.Harness
import Gym.Env
import Gym.ToyText.CliffWalking


rewardOf : (Double, CWState, Outcome, Info) -> Double
rewardOf (r, _, _, _) = r

stateOf : (Double, CWState, Outcome, Info) -> CWState
stateOf (_, s, _, _) = s

outcomeOf : (Double, CWState, Outcome, Info) -> Outcome
outcomeOf (_, _, o, _) = o


cwInit : CWState
cwInit = MkCW 3 0


export
tests : List (IO Bool)
tests =
  [ check "reset at (3,0) for any seed" $
      let (r,  _) = reset {state=CWState} {action=Nat} {obs=Nat} 42
          (r2, _) = reset {state=CWState} {action=Nat} {obs=Nat} 99
      in r.cwRow == 3 && r.cwCol == 0
         && r2.cwRow == 3 && r2.cwCol == 0

  , check "CliffWalking reset passes seed through unchanged" $
      let (_, s') = reset {state=CWState} {action=Nat} {obs=Nat} 42
      in s' == 42

  , check "observe encodes row*12+col" $
      cwObserve (MkCW 2 3) == 27

  , check "step into cliff resets to start, -100" $
      let s  = MkCW 3 0             -- start
          r  = cwStep s 1           -- right into (3,1) cliff
      in rewardOf r == -100.0
         && (stateOf r).cwRow == 3
         && (stateOf r).cwCol == 0
         && outcomeOf r == Continue

  , check "step to goal terminates, -1" $
      let s  = MkCW 2 11            -- above goal
          r  = cwStep s 2           -- down into (3,11) goal
      in rewardOf r == -1.0
         && outcomeOf r == Terminated

  , check "step bumping wall stays put" $
      let s  = MkCW 0 0
          r  = cwStep s 3           -- left into wall
      in (stateOf r).cwCol == 0

  , check "normal step -1" $
      rewardOf (cwStep (MkCW 0 5) 2) == -1.0

  , check "actionSpace is Discrete 4" $
      case actionSpace {state=CWState} {action=Nat} {obs=Nat} of
        Discrete 4 => True
        _ => False
  ]
