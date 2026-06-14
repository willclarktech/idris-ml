module Bench

import Data.List
import Data.Vect
import System
import System.Clock

import Gym.ClassicControl.Acrobot
import Gym.ClassicControl.Pendulum
import Gym.Env
import Gym.Rng
import Gym.ToyText.Blackjack
import Gym.ToyText.CliffWalking
import Gym.ToyText.Taxi

%default partial

----------------------------------------------------------------------
-- Timing helper
----------------------------------------------------------------------

elapsedMs : Clock Monotonic -> Clock Monotonic -> Double
elapsedMs t0 t1 =
  let s  = cast {to=Double} (seconds t1 - seconds t0)
      ns = cast {to=Double} (nanoseconds t1 - nanoseconds t0)
  in s * 1000.0 + ns / 1000000.0

bench : String -> Nat -> (Nat -> IO a) -> IO ()
bench name iters body = do
  t0 <- clockTime Monotonic
  _  <- body iters
  t1 <- clockTime Monotonic
  let ms = elapsedMs t0 t1
  let nsPerCall = ms * 1000000.0 / cast {to=Double} (cast {to=Integer} iters)
  putStrLn (name ++ ":  " ++ show iters ++ " iters in "
                 ++ show ms ++ " ms  ("
                 ++ show nsPerCall ++ " ns/call)")

----------------------------------------------------------------------
-- 1. Rng.nextDouble (candidate #3)
----------------------------------------------------------------------

-- Drive nextDouble iters times, threading the seed; accumulate result so
-- the compiler can't eliminate the call.
loopNextDouble : Nat -> Seed -> Double -> Double
loopNextDouble Z      _ acc = acc
loopNextDouble (S k)  s acc =
  let (d, s') = nextDouble s
  in loopNextDouble k s' (acc + d)

benchRngNextDouble : Nat -> IO Double
benchRngNextDouble n = pure (loopNextDouble n 0x12345 0.0)

----------------------------------------------------------------------
-- 2. Blackjack handSum + usableAce (candidate #5)
----------------------------------------------------------------------

-- Three precomputed BJ states with different ace patterns. Cycling between
-- them per-iteration defeats hoisting without allocating fresh lists in the
-- loop body. The states are top-level CAFs — built once.
bj0 : BJState
bj0 = MkBJ [7, 3, 1]     [10, 6] False 0x12345  -- one ace, low total

bj1 : BJState
bj1 = MkBJ [5, 1, 1, 1]  [10, 6] False 0x12345  -- multiple aces

bj2 : BJState
bj2 = MkBJ [10, 8]       [10, 6] False 0x12345  -- no aces

loopBJObserve : Nat -> Bits64 -> Double -> Double
loopBJObserve Z     _   acc = acc
loopBJObserve (S k) i   acc =
  -- Bits64 modulo for the cycling index avoids any Peano-Nat gotchas.
  let s = case prim__and_Bits64 i 3 of
            0 => bj0
            1 => bj1
            _ => bj2
      v = bjObserve s
      x = case v of [a, b, c] => a + b + c
  in loopBJObserve k (i + 1) (acc + x)

benchBlackjackObserve : Nat -> IO Double
benchBlackjackObserve n = pure (loopBJObserve n 0 0.0)

----------------------------------------------------------------------
-- 3. Acrobot step+observe (candidate #1)
----------------------------------------------------------------------

acrobotInit : AState
acrobotInit = MkA 0.1 0.2 0.0 0.0

-- One step (action=0 i.e. -1 torque) + observation. Take the sum of obs
-- to keep dependencies live.
loopAcrobot : Nat -> AState -> Double -> Double
loopAcrobot Z     _ acc = acc
loopAcrobot (S k) s acc =
  let (_, s', _, _) = aStep s 0
      obs           = aObserve s'
      [c1, s1, c2, s2, d1, d2] = obs
      x             = c1 + s1 + c2 + s2 + d1 + d2
  in loopAcrobot k s' (acc + x)

benchAcrobot : Nat -> IO Double
benchAcrobot n = pure (loopAcrobot n acrobotInit 0.0)

----------------------------------------------------------------------
-- 4. Pendulum step+observe (candidate #2 — sanity check)
----------------------------------------------------------------------

pendulumInit : PState
pendulumInit = MkP 3.14 0.0

loopPendulum : Nat -> PState -> Double -> Double
loopPendulum Z     _ acc = acc
loopPendulum (S k) s acc =
  let (_, s', _, _) = pStep s 0.5
      obs           = pObserve s'
      [c, sn, dd]   = obs
      x             = c + sn + dd
  in loopPendulum k s' (acc + x)

benchPendulum : Nat -> IO Double
benchPendulum n = pure (loopPendulum n pendulumInit 0.0)

----------------------------------------------------------------------
-- 5. Taxi step (candidate #4)
----------------------------------------------------------------------

taxiInit : TState
taxiInit = MkT 2 2 0 3

-- Cycle through actions to exercise the moveTo paths.
loopTaxi : Nat -> TState -> Bits64 -> Double -> Double
loopTaxi Z     _ _ acc = acc
loopTaxi (S k) s aIdx acc =
  let (r, s', _, _) = tStep s (cast {to=Nat} (prim__and_Bits64 aIdx 3))
  in loopTaxi k s' (aIdx + 1) (acc + r)

benchTaxi : Nat -> IO Double
benchTaxi n = pure (loopTaxi n taxiInit 0 0.0)

----------------------------------------------------------------------
-- 6. CliffWalking step (candidate #4 sibling)
----------------------------------------------------------------------

cwInit : CWState
cwInit = MkCW 0 0

loopCW : Nat -> CWState -> Bits64 -> Double -> Double
loopCW Z     _ _ acc = acc
loopCW (S k) s aIdx acc =
  let (r, s', _, _) = cwStep s (cast {to=Nat} (prim__and_Bits64 aIdx 3))
  in loopCW k s' (aIdx + 1) (acc + r)

benchCliffWalking : Nat -> IO Double
benchCliffWalking n = pure (loopCW n cwInit 0 0.0)

----------------------------------------------------------------------
-- Driver
----------------------------------------------------------------------

main : IO ()
main = do
  -- Warm up
  _ <- benchRngNextDouble 100000

  -- Iteration counts tuned for ~1-3 second runs. Allow per-bench
  -- selection via CLI so a single bench can be re-run during dev.
  args <- getArgs
  let selected = case args of
        _ :: rest@(_ :: _) => rest  -- everything past prog name
        _                  => ["all"]
  let want : String -> Bool
      want s = elem "all" selected || elem s selected
  when (want "rng")          $ bench "Rng.nextDouble       " 5_000_000 benchRngNextDouble
  when (want "blackjack")    $ bench "Blackjack.bjObserve  "   500_000 benchBlackjackObserve
  when (want "pendulum")     $ bench "Pendulum step+observe" 1_000_000 benchPendulum
  when (want "acrobot")      $ bench "Acrobot step+observe "   500_000 benchAcrobot
  when (want "taxi")         $ bench "Taxi step            "   500_000 benchTaxi
  when (want "cliffwalking") $ bench "CliffWalking step    "   500_000 benchCliffWalking
