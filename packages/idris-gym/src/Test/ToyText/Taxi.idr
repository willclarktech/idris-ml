module Test.ToyText.Taxi

import Test.Harness
import Gym.Env
import Gym.ToyText.Taxi


rewardOf : (Double, TState, Outcome, Info) -> Double
rewardOf (r, _, _, _) = r

stateOf : (Double, TState, Outcome, Info) -> TState
stateOf (_, s, _, _) = s

outcomeOf : (Double, TState, Outcome, Info) -> Outcome
outcomeOf (_, _, o, _) = o


tInit : TState
tInit = MkT 2 2 0 3


export
tests : List (IO Bool)
tests =
  [ check "default start at (2,2)" $
      let r : TState
          r = reset {state=TState} {action=Nat} {obs=Nat}
      in r.tRow == 2 && r.tCol == 2

  , check "encoding is 0..499" $
      let obs = tObserve (MkT 4 4 4 3)
      in obs < 500

  , check "move down" $
      let r = tStep (MkT 2 2 0 3) 0
      in (stateOf r).tRow == 3

  , check "normal move -1" $
      rewardOf (tStep (MkT 2 2 0 3) 0) == -1.0

  , check "illegal pickup -10" $
      -- Taxi at (2,2), passenger at R(0,0), not at same square
      rewardOf (tStep (MkT 2 2 0 3) 4) == -10.0

  , check "legal pickup -1" $
      -- Taxi at R(0,0), passenger at R(0,0)
      rewardOf (tStep (MkT 0 0 0 3) 4) == -1.0

  , check "pickup moves passenger into taxi" $
      let r = tStep (MkT 0 0 0 3) 4
      in (stateOf r).tPass == 4

  , check "legal dropoff +20" $
      -- Taxi at dest B(4,3), passenger in taxi
      rewardOf (tStep (MkT 4 3 4 3) 5) == 20.0

  , check "legal dropoff terminates" $
      outcomeOf (tStep (MkT 4 3 4 3) 5) == Terminated

  , check "illegal dropoff -10" $
      -- Passenger in taxi but not at dest
      rewardOf (tStep (MkT 2 2 4 3) 5) == -10.0

  , check "actionSpace is Discrete 6" $
      case actionSpace {state=TState} {action=Nat} {obs=Nat} of
        Discrete 6 => True
        _ => False
  ]
