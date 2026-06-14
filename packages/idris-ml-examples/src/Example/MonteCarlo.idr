module Example.MonteCarlo

import Data.Fin
import Data.List
import Data.Maybe
import Data.Vect
import System

import Array
import BuildConfig
import Compat.Random
import DataStream
import Executor
import Fit
import Gym.Env
import Gym.Rng
import Gym.ToyText.Blackjack
import Math
import Train

----------------------------------------------------------------------
-- Env dimensions
--
-- State encoding: (player_sum in 4..21, dealer_show in 1..10, usable in {0,1}).
-- idx = (ps - 4) * 20 + (ds - 1) * 2 + ua   →  0..359
-- We use NumStates = 400 as a safety margin.
-- Max trajectory length per hand is small (<= ~10 hits before bust), but
-- we allocate noise for 10 actions (= 20 uniforms) per hand.
----------------------------------------------------------------------

NumStates  : Nat; NumStates = 400
NumActions : Nat; NumActions = 2
MaxSteps   : Nat; MaxSteps = 10

----------------------------------------------------------------------
-- Q-table + visit counts (both as Tensors)
----------------------------------------------------------------------

QTable : Type
QTable = Array [NumStates, NumActions] Double

CountTable : Type
CountTable = Array [NumStates, NumActions] Double

MCModel : Type
MCModel = (QTable, CountTable)

zeroModel : MCModel
zeroModel =
  (replicate {dims = [NumStates, NumActions]} 0.0,
   replicate {dims = [NumStates, NumActions]} 0.0)

tGet : Fin NumStates -> Fin NumActions -> Array [NumStates, NumActions] Double -> Double
tGet i j t =
  let SArray v = index j (index i t)
  in v

tRow : Fin NumStates -> Array [NumStates, NumActions] Double -> Vector NumActions Double
tRow i t = index i t

tSet : Fin NumStates -> Fin NumActions -> Double ->
       Array [NumStates, NumActions] Double -> Array [NumStates, NumActions] Double
tSet i j v (VArray rows) =
  let (VArray cells) = Data.Vect.index i rows
      newRow = VArray (Data.Vect.replaceAt j (SArray v) cells)
  in VArray (Data.Vect.replaceAt i newRow rows)

----------------------------------------------------------------------
-- State encoding
----------------------------------------------------------------------

clampRange : Integer -> Integer -> Integer -> Integer
clampRange lo hi x = if x < lo then lo else if x > hi then hi else x

encodeBJ : BJState -> Fin NumStates
encodeBJ s =
  case bjObserve s of
    [p, d, u] =>
      let psI = clampRange 4 21 (cast {to=Integer} p)
          dsI = clampRange 1 10 (cast {to=Integer} d)
          uaI = if cast {to=Integer} u >= 1 then 1 else 0
          psN : Nat
          psN = integerToNat (psI - 4)
          dsN : Nat
          dsN = integerToNat (dsI - 1)
          uaN : Nat
          uaN = integerToNat uaI
          idx = psN * 20 + dsN * 2 + uaN
      in case natToFin idx NumStates of
           Just f  => f
           Nothing => FZ

----------------------------------------------------------------------
-- Epsilon-greedy
----------------------------------------------------------------------

epsGreedy : Double -> Vector NumActions Double -> Double -> Double -> Fin NumActions
epsGreedy eps qr u1 u2 =
  if u1 < eps
    then let idx : Nat
             idx = integerToNat (cast (u2 * cast NumActions))
         in case natToFin idx NumActions of
              Just f  => f
              Nothing => FZ
    else argmax qr

----------------------------------------------------------------------
-- Episode rollout (trajectory only; rewards pooled at terminal)
----------------------------------------------------------------------

-- Returns (trajectory (s, a) in play order, terminalReward).
playHand : Double -> QTable -> BJState -> Nat -> List Double ->
           List (Fin NumStates, Fin NumActions) ->
           (List (Fin NumStates, Fin NumActions), Double)
playHand _ _ _ Z _ acc                             = (reverse acc, 0.0)
playHand _ _ _ _ [] acc                            = (reverse acc, 0.0)
playHand _ _ _ _ [_] acc                           = (reverse acc, 0.0)
playHand eps q st (S steps) (u1 :: u2 :: rest) acc =
  let sIdx = encodeBJ st
      aFin = epsGreedy eps (tRow sIdx q) u1 u2
      aNat = finToNat aFin
  in case bjStep st aNat of
       (reward, st', outcome, _) =>
         let acc' = (sIdx, aFin) :: acc
         in case outcome of
              Continue => playHand eps q st' steps rest acc'
              _        => (reverse acc', reward)

----------------------------------------------------------------------
-- First-visit MC update
----------------------------------------------------------------------

-- Apply G to each distinct (s, a) pair visited (first occurrence only).
-- Already-seen pairs are skipped via the `seen` accumulator.
applyVisits : List (Fin NumStates, Fin NumActions) ->
              List (Fin NumStates, Fin NumActions) ->
              Double -> MCModel -> MCModel
applyVisits [] _ _ m                       = m
applyVisits ((s, a) :: rest) seen g (q, n) =
  if any (\(s', a') => finToNat s == finToNat s' && finToNat a == finToNat a') seen
    then applyVisits rest seen g (q, n)
    else
      let oldN = tGet s a n
          newN = oldN + 1.0
          n'   = tSet s a newN n
          oldQ = tGet s a q
          newQ = oldQ + (g - oldQ) / newN
          q'   = tSet s a newQ q
      in applyVisits rest ((s, a) :: seen) g (q', n')

----------------------------------------------------------------------
-- Config + epoch
----------------------------------------------------------------------

record Config where
  constructor MkConfig
  epsilon : Double
  epochs  : Nat
  seed    : Bits64

defaultConfig : Config
defaultConfig = MkConfig 0.1 50000 42

specs : List (ArgSpec Config)
specs = [ Arg "--epsilon" (\v, c => { epsilon := cast v } c)
        , Arg "--epochs" (\v, c => { epochs := castNat v } c)
        , Arg "--seed" (\v, c => { seed := castBits64 v } c)
        ]

record EpochInput where
  constructor MkEI
  envSeed : Bits64
  noise   : List Double

epochMC : Config -> MCModel -> EpochInput -> (MCModel, Double)
epochMC cfg (q, n) (MkEI envSeed noise) =
  let (st0, _) = initBJ envSeed
      (traj, reward) = playHand cfg.epsilon q st0 MaxSteps noise []
      (q', n')       = applyVisits traj [] reward (q, n)
  in ((q', n'), negate reward)

----------------------------------------------------------------------
-- Input generation
----------------------------------------------------------------------

genNoise : Nat -> IO (List Double)
genNoise Z     = pure []
genNoise (S k) = do
  u <- randomRIO (the Double 0.0, 1.0)
  rest <- genNoise k
  pure (u :: rest)

||| Derive a Bits64 seed from the IO PRNG (via a Double in [0,1)).
genSeed : IO Bits64
genSeed = do
  u <- randomRIO (the Double 0.0, 1.0)
  pure (cast {to=Bits64} (cast {to=Integer} (u * 9.223372036854776e18)))

genInput : IO EpochInput
genInput = do
  s <- genSeed
  noise <- genNoise (MaxSteps * 2)
  pure (MkEI s noise)

----------------------------------------------------------------------
-- Greedy evaluation (plays against the env using argmax on Q)
----------------------------------------------------------------------

evalHand : QTable -> BJState -> Nat -> Double
evalHand _ _ Z      = 0.0
evalHand q st (S k) =
  let sIdx = encodeBJ st
      aNat = finToNat (argmax (tRow sIdx q))
  in case bjStep st aNat of
       (reward, st', outcome, _) =>
         case outcome of
           Continue => evalHand q st' k
           _        => reward

evalN : QTable -> Nat -> Bits64 -> Double -> Double -> IO Double
evalN _ Z _ wins played           = pure (wins / played)
evalN q (S k) envSeed wins played = do
  s <- genSeed
  let r = evalHand q (fst (initBJ s)) MaxSteps
      wins' = if r > 0.0 then wins + 1.0 else wins
  evalN q k envSeed wins' (played + 1.0)

----------------------------------------------------------------------
-- Main
----------------------------------------------------------------------

main : IO ()
main = do
  requireMachine {m = ChosenMachine}
  args <- getArgs
  let cfg = parseArgs defaultConfig specs (drop 1 args)
  srand cfg.seed

  putStrLn "=== First-visit MC on Blackjack ==="
  putStrLn $ "Config: epsilon=" ++ show cfg.epsilon
           ++ " epochs=" ++ show cfg.epochs
           ++ " seed=" ++ show cfg.seed
  putStrLn ""

  let trainCfg : TrainConfig MCModel
      trainCfg = mkTrainConfig cfg.epochs 5000 NoEarlyStop (const (pure [])) (\_ => pure ())
  (trained, epochsDone, _) <- fitCustom {ex=ExampleExecutor}
    (\m, d => pure (epochMC cfg m d))
    (generate genInput)
    trainCfg zeroModel

  let (q, _) = trained

  putStrLn ""
  winRate <- evalN q 5000 0 0.0 0.0
  putStrLn $ "Eval (5000 hands, greedy): win_rate=" ++ show winRate
  putStrLn ""
  putStrLn $ formatResult [("win_rate", show winRate),
                            ("epochs", show epochsDone),
                            ("seed", show cfg.seed)]
