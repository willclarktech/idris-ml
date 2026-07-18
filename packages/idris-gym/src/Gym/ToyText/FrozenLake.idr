module Gym.ToyText.FrozenLake

import Data.Vect

import Gym.Env
import Gym.Rng

----------------------------------------------------------------------
-- FrozenLake-v1 (Gymnasium-compatible, 4x4 default map)
--
-- Map layout (Gymnasium 4x4):
--   S F F F
--   F H F H
--   F F F H
--   H F F G
-- Where S=start, F=frozen, H=hole (terminal), G=goal (terminal, +1).
--
-- Actions: 0=left, 1=down, 2=right, 3=up.
-- With isSlippery=True, the chosen direction has probability 1/3 and
-- each perpendicular direction has probability 1/3 (no-op replaced by
-- slip). Gymnasium's exact slip distribution.
--
-- Reward: +1 at goal, 0 elsewhere. No reward shaping.
----------------------------------------------------------------------

public export
data Tile = Start | Frozen | Hole | Goal

NumRows : Nat; NumRows = 4
NumCols : Nat; NumCols = 4

defaultMap : Vect 16 Tile
defaultMap =
  [ Start,  Frozen, Frozen, Frozen
  , Frozen, Hole,   Frozen, Hole
  , Frozen, Frozen, Frozen, Hole
  , Hole,   Frozen, Frozen, Goal
  ]

||| FrozenLake state. The seed is advanced on each stochastic step.
public export
record FLState where
  constructor MkFL
  flPos      : Nat
  flMap      : Vect 16 Tile
  flSeed     : Seed
  flSlippery : Bool

||| Reset from a caller-provided seed and slipperiness flag.
||| The input Seed seeds the internal `flSeed` used by slip; the returned
||| caller-side Seed is advanced one step so repeated resets diverge.
export
initFL : Bool -> Seed -> (FLState, Seed)
initFL slip seed =
  let (_, seed') = splitMix64 seed
  in (MkFL 0 defaultMap seed slip, seed')

encodeRC : Nat -> Nat -> Nat
encodeRC r c = r * NumCols + c

decodeR : Nat -> Nat
decodeR p = p `div` NumCols

decodeC : Nat -> Nat
decodeC p = p `mod` NumCols

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

-- Apply an action deterministically to a position.
moveDet : Nat -> Nat -> (Nat, Nat)
moveDet pos action =
  let r = cast {to=Integer} (decodeR pos)
      c        = cast {to=Integer} (decodeC pos)
      (r', c') = case action of
                   0 => (r,     c - 1)  -- left
                   1 => (r + 1, c    )  -- down
                   2 => (r,     c + 1)  -- right
                   _ => (r - 1, c    )  -- up
  in (clampRow r', clampCol c')

-- For a slippery step, resolve the actual direction taken.
slipAction : Seed -> Nat -> (Nat, Seed)
slipAction seed intended =
  let (choice, seed') = nextNat seed 3
      -- 0 -> intended; 1 -> left perpendicular; 2 -> right perpendicular
      actual : Nat
      actual = case choice of
                 0 => intended
                 1 => case intended of
                        0 => 3   -- left -> up
                        1 => 0   -- down -> left
                        2 => 1   -- right -> down
                        _ => 2   -- up -> right
                 _ => case intended of
                        0 => 1   -- left -> down
                        1 => 2   -- down -> right
                        2 => 3   -- right -> up
                        _ => 0   -- up -> left
  in (actual, seed')

tileAt : Vect 16 Tile -> Nat -> Tile
tileAt m p =
  case natToFin p 16 of
    Just ix => index ix m
    Nothing => Frozen   -- unreachable when positions stay in bounds

||| One step.
export
flStep : FLState -> Nat -> (Double, FLState, Outcome, Info)
flStep s action =
  let (act, seed') = if s.flSlippery
                        then slipAction s.flSeed action
                        else (action, s.flSeed)
      (rn, cn) = moveDet s.flPos act
      pos'     = encodeRC rn cn
      s'       = { flPos := pos', flSeed := seed' } s
  in case tileAt s.flMap pos' of
       Goal => (1.0, s', Terminated, [])
       Hole => (0.0, s', Terminated, [])
       _    => (0.0, s', Continue, [])

||| Observation: current position (0..15).
export
flObserve : FLState -> Nat
flObserve s = s.flPos

public export
Env FLState Nat Nat where
  reset            = initFL True
  step             = flStep
  observe          = flObserve
  actionSpace      = Discrete 4
  obsSpace         = Discrete 16
  defaultTimeLimit = Nothing
