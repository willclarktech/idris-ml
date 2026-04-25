module Gym.ToyText.Taxi

import Data.Vect
import Gym.Env


----------------------------------------------------------------------
-- Taxi-v3 (Gymnasium-compatible)
--
-- 5x5 grid with 4 designated locations R(0,0), G(0,4), Y(4,0), B(4,3).
-- Taxi must pick up passenger from their location and drop them at
-- the destination location.
--
-- State encoding: ((row*5 + col)*5 + passIdx)*4 + destIdx
--   passIdx in {0,1,2,3} = location, 4 = in taxi
--   destIdx in {0,1,2,3}
-- Total: 5*5*5*4 = 500 states.
--
-- Actions: 0=down, 1=up, 2=right, 3=left, 4=pickup, 5=dropoff
-- Rewards: -1/step, +20 dropoff at correct dest, -10 illegal pickup/dropoff
----------------------------------------------------------------------

NumRows : Nat; NumRows = 5
NumCols : Nat; NumCols = 5

-- Locations in (row, col): R, G, Y, B
locRow : Nat -> Nat
locRow 0 = 0   -- R
locRow 1 = 0   -- G
locRow 2 = 4   -- Y
locRow _ = 4   -- B

locCol : Nat -> Nat
locCol 0 = 0   -- R
locCol 1 = 4   -- G
locCol 2 = 0   -- Y
locCol _ = 3   -- B

||| Taxi state: taxi position, passenger location, destination.
public export
record TState where
  constructor MkT
  tRow, tCol, tPass, tDest : Nat

||| Default initial state: taxi at (2,2), passenger at R (0), dest at B (3).
export
defaultStart : TState
defaultStart = MkT 2 2 0 3

encode : TState -> Nat
encode s = ((s.tRow * NumCols + s.tCol) * 5 + s.tPass) * 4 + s.tDest

-- Walls in the 5x5 taxi grid: cannot move between (row, col_a)-(row, col_b)
-- Gymnasium's walls: between cols 1-2 in rows 0,1 and between cols 2-3 in rows 3,4.
blocked : Nat -> Nat -> Nat -> Nat -> Bool
blocked r c r' c' =
  if r /= r' then False
  else
    let (lo, hi) = if c < c' then (c, c') else (c', c)
    in if (r == 0 || r == 1) && lo == 1 && hi == 2 then True
       else if (r == 3 || r == 4) && lo == 2 && hi == 3 then True
       else False

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

||| One step.
export
tStep : TState -> Nat -> (Double, TState, Outcome, Info)
tStep s action =
  case action of
    0 => moveTo (cast {to=Integer} s.tRow + 1) (cast {to=Integer} s.tCol)  -- down
    1 => moveTo (cast {to=Integer} s.tRow - 1) (cast {to=Integer} s.tCol)  -- up
    2 => moveTo (cast {to=Integer} s.tRow) (cast {to=Integer} s.tCol + 1)  -- right
    3 => moveTo (cast {to=Integer} s.tRow) (cast {to=Integer} s.tCol - 1)  -- left
    4 => pickup                                                             -- pickup
    _ => dropoff                                                            -- dropoff
  where
    moveTo : Integer -> Integer -> (Double, TState, Outcome, Info)
    moveTo rI cI =
      let rn = clampRow rI
          cn = clampCol cI
          blockedMove = blocked s.tRow s.tCol rn cn
          s' = if blockedMove then s else { tRow := rn, tCol := cn } s
      in (-1.0, s', Continue, [])

    pickup : (Double, TState, Outcome, Info)
    pickup =
      if s.tPass < 4
         && s.tRow == locRow s.tPass && s.tCol == locCol s.tPass
        then (-1.0, { tPass := 4 } s, Continue, [])
      else (-10.0, s, Continue, [])

    dropoff : (Double, TState, Outcome, Info)
    dropoff =
      if s.tPass == 4
         && s.tRow == locRow s.tDest && s.tCol == locCol s.tDest
        then (20.0, { tPass := s.tDest } s, Terminated, [])
      else (-10.0, s, Continue, [])

||| Observation: single-integer state encoding (0..499).
export
tObserve : TState -> Nat
tObserve = encode

public export
Env TState Nat Nat where
  reset = defaultStart
  step = tStep
  observe = tObserve
  actionSpace = Discrete 6
  obsSpace = Discrete 500
  defaultTimeLimit = Just 200
