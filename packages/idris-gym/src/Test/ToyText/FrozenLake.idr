module Test.ToyText.FrozenLake

import Gym.Env
import Gym.Rng
import Gym.ToyText.FrozenLake
import Test.Harness

rewardOf : (Double, FLState, Outcome, Info) -> Double
rewardOf (r, _, _, _) = r

stateOf : (Double, FLState, Outcome, Info) -> FLState
stateOf (_, s, _, _) = s

outcomeOf : (Double, FLState, Outcome, Info) -> Outcome
outcomeOf (_, _, o, _) = o

-- Count samples where the stepped outcome goes "right" (pos = start + 1) vs other.
-- Start at position 0. Action = right (2). With slip, 1/3 chance each of
-- intended/left-perpendicular/right-perpendicular.
countRights : Seed -> Nat -> Nat -> Nat
countRights _ Z acc        = acc
countRights seed (S k) acc =
  let (st, _) = initFL True seed
      r    = flStep st 2
      s'   = stateOf r
      acc' = if s'.flPos == 1 then S acc else acc
      -- Advance seed to get independent trials
      (_, seed') = nextDouble seed
  in countRights seed' k acc'

export
tests : List (IO Bool)
tests =
  [ check "non-slippery step right" $
      let (st, _) = initFL False 42
          r  = flStep st 2
      in (stateOf r).flPos == 1

  , check "non-slippery left wall stays put" $
      let (st, _) = initFL False 42
          r  = flStep st 0
      in (stateOf r).flPos == 0

  , check "step into hole terminates, 0 reward" $
      -- Hole at (1,1) = position 5. From (1,0) = position 4, move right.
      let st = { flPos := 4 } (fst (initFL False 42))
          r  = flStep st 2
      in outcomeOf r == Terminated && rewardOf r == 0.0

  , check "step into goal terminates, reward 1" $
      -- Goal at (3,3) = position 15. From (2,3) = 11, down.
      let st = { flPos := 11 } (fst (initFL False 42))
          r  = flStep st 1
      in outcomeOf r == Terminated && rewardOf r == 1.0

  , check "slippery has ~1/3 rightward from start" $
      -- Run 300 trials, count how many go right from start(0) when action=right.
      let rights = countRights 7 300 0
          frac = cast {to=Double} (cast {to=Integer} rights) / 300.0
      in abs (frac - 0.333333) < 0.15

  , check "obsSpace Discrete 16" $
      case obsSpace {state=FLState} {action=Nat} {obs=Nat} of
        Discrete 16 => True
        _           => False
  ]
