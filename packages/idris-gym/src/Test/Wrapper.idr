module Test.Wrapper

import Data.Vect
import Test.Harness
import Gym.Env
import Gym.Space
import Gym.Wrapper
import Gym.ClassicControl.CartPole

-- Run the TimeLimited CartPole with action 1 until an outcome fires,
-- returning (numSteps, finalOutcome).
runTl : Nat -> TimeLimited CPState -> (Nat, Outcome)
runTl Z     _  = (Z, Continue)
runTl (S k) tl =
  let (_, tl', out, _) = timeLimitedStep {state=CPState} {action=Nat}
                                          {obs=Vect 4 Double} tl 1
  in case out of
       Continue => let (n, o) = runTl k tl' in (S n, o)
       _        => (S Z, out)

-- Explicit CPState zero state — avoids Env instance disambiguation.
cpZero : CPState
cpZero = MkCP 0 0 0 0

export
tests : List (IO Bool)
tests =
  [ check "TimeLimit truncates after N steps" $
      let tl = timeLimited 5 cpZero
          (n, out) = runTl 100 tl
      in n == 5 && out == Truncated

  , check "Recorded step 1 accumulates reward 1" $
      let init = recorded cpZero
      in case recordedStep {state=CPState} {action=Nat}
                           {obs=Vect 4 Double} init 1 of
           (_, s', _, _) => s'.totalReward == 1.0 && s'.epLength == 1

  , check "clipScalarAction within bounds" $
      clipScalarAction (Box [-2.0] [2.0]) 0.5 == 0.5

  , check "clipScalarAction exceeds" $
      clipScalarAction (Box [-2.0] [2.0]) 10.0 == 2.0

  , check "rescaleScalarAction maps [-1,1] to [0,10]" $
      rescaleScalarAction (Box [-1.0] [1.0]) (Box [0.0] [10.0]) 0.0 == 5.0
  ]
