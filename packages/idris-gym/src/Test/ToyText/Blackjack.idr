module Test.ToyText.Blackjack

import Data.Vect
import Test.Harness
import Gym.Env
import Gym.Rng
import Gym.ToyText.Blackjack

rewardOf : (Double, BJState, Outcome, Info) -> Double
rewardOf (r, _, _, _) = r

stateOf : (Double, BJState, Outcome, Info) -> BJState
stateOf (_, s, _, _) = s

outcomeOf : (Double, BJState, Outcome, Info) -> Outcome
outcomeOf (_, _, o, _) = o

export
tests : List (IO Bool)
tests =
  [ check "init gives player 2 cards" $
      let (s, _) = initBJ 42
      in length s.bjPlayer == 2

  , check "init gives dealer 2 cards" $
      let (s, _) = initBJ 42
      in length s.bjDealer == 2

  , check "init is not done" $
      let (s, _) = initBJ 42
      in s.bjDone == False

  , check "observe length 3" $
      length (bjObserve (fst (initBJ 42))) == 3

  , check "hit adds a card" $
      let (s, _) = initBJ 42
          r = bjStep s 1
      in length (stateOf r).bjPlayer == 3

  , check "stick terminates" $
      let (s, _) = initBJ 42
          r = bjStep s 0
      in outcomeOf r == Terminated

  , check "reward is in {-1, 0, 1}" $
      let (s, _) = initBJ 42
          r  = bjStep s 0
          rw = rewardOf r
      in rw == -1.0 || rw == 0.0 || rw == 1.0

  , check "seed is deterministic" $
      let (s1, _) = initBJ 42
          (s2, _) = initBJ 42
      in s1.bjPlayer == s2.bjPlayer && s1.bjDealer == s2.bjDealer

  , check "actionSpace Discrete 2" $
      case actionSpace {state=BJState} {action=Nat} {obs=Vect 3 Double} of
        Discrete 2 => True
        _          => False
  ]
