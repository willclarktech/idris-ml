module Gym.ToyText.CliffWalking

import Gym.Env
import Gym.Rng

----------------------------------------------------------------------
-- CliffWalking-v0 (Gymnasium-compatible)
--
-- 4x12 grid. Start at (3,0), goal at (3,11). Row 3, cols 1..10 is a
-- cliff: stepping there gives reward -100 and teleports to start
-- (episode continues). Reward -1 per step. Episode ends at goal.
----------------------------------------------------------------------

NumRows  : Nat; NumRows = 4
NumCols  : Nat; NumCols = 12
StartRow : Nat; StartRow = 3
StartCol : Nat; StartCol = 0
GoalRow  : Nat; GoalRow = 3
GoalCol  : Nat; GoalCol = 11

||| State = single integer encoding row * 12 + col.
public export
record CWState where
  constructor MkCW
  cwRow, cwCol : Nat

startState : CWState
startState = MkCW StartRow StartCol

encode : CWState -> Nat
encode s = s.cwRow * NumCols + s.cwCol

-- Is (r, c) the cliff strip?
onCliff : Nat -> Nat -> Bool
onCliff r c = r == 3 && c >= 1 && c < 11

-- Clamp position to grid bounds.
clampRow : Integer -> Nat
clampRow r =
  if r < 0 then Z
  else if r >= cast NumRows then cast NumRows `minus` 1
  else cast r

clampCol : Integer -> Nat
clampCol c =
  if c < 0 then Z
  else if c >= cast NumCols then cast NumCols `minus` 1
  else cast c

||| Action 0=up, 1=right, 2=down, 3=left.
export
cwStep : CWState -> Nat -> (Double, CWState, Outcome, Info)
cwStep s action =
  let rI = cast {to=Integer} s.cwRow
      cI         = cast {to=Integer} s.cwCol
      (rI', cI') = case action of
                     0 => (rI - 1, cI)       -- up
                     1 => (rI,     cI + 1)   -- right
                     2 => (rI + 1, cI)       -- down
                     _ => (rI,     cI - 1)   -- left (default)
      rn                    = clampRow rI'
      cn                    = clampCol cI'
      tentative             = MkCW rn cn
      (s', reward, outcome) =
        if onCliff rn cn
          then (startState, -100.0, Continue)
        else if rn == GoalRow && cn == GoalCol
          then (tentative, -1.0, Terminated)
        else (tentative, -1.0, Continue)
  in (reward, s', outcome, [])

||| Observation: single integer row*12+col.
export
cwObserve : CWState -> Nat
cwObserve = encode

public export
Env CWState Nat Nat where
  reset            = \s => (startState, s)
  step             = cwStep
  observe          = cwObserve
  actionSpace      = Discrete 4
  obsSpace         = Discrete 48   -- 4 * 12
  defaultTimeLimit = Nothing
