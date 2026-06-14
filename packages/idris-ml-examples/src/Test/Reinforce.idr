module Test.Reinforce

import Control.Linear.LIO
import Data.Linear.Notation
import Data.List
import Data.Vect
import System

import Compat.Random
import Example.Reinforce
import Gym.ClassicControl.CartPole
import Gym.Vector
import ML.Simple
import Test.Harness

-- Short step budget so test runs are tape-friendly. 200 steps × N=2
-- × 2 paths blew up under whatever backend the test linked against.
testMaxSteps : Nat
testMaxSteps = 20

||| Build a fresh deterministic REINFORCE policy for use in both rollout
||| paths. Built ONCE per test and reused across the sequential and
||| batched rollouts, so the two paths see an identical model — that's
||| what makes the parity assertion meaningful (the init values
||| themselves don't matter). Mirrors the example's `Policy` MLP
||| (4 -> 128 -> tanh -> 2) on the `Nn`/`runInit` surface.
mkModel : IO Policy
mkModel = do
  srand 12345  -- deterministic init for parity reproducibility
  runInit $ do
    l1 <- linear {i=4} {o=128}
    l2 <- linear {i=128} {o=2}
    pure (l1 ~~> tanhA ~~> l2 ~~> Nil)

||| A deterministic pseudo-RNG sequence to drive `categoricalSample`.
||| Pre-computed so both rollouts see identical randomness. Idris is
||| strict, so we can't use a naive infinite `cycle` — build the list
||| of length n directly by indexing into a fixed table.
fakeRandomness : Nat -> List Double
fakeRandomness n = go n 0
  where
    table : Vect 8 Double
    table = [0.13, 0.81, 0.27, 0.55, 0.09, 0.92, 0.41, 0.68]

    pick : Nat -> Double
    pick i = let m : Fin 8 = restrict 7 (cast i) in index m table

    go : Nat -> Nat -> List Double
    go Z      _ = []
    go (S k)  i = pick i :: go k (S i)

||| Initial CartPole state. `MkCP 0 0 0 0` is the standard reset.
initState : CPState
initState = MkCP 0 0 0 0

||| Stage 1 parity: batched with N=1 must produce identical
||| per-episode total reward to a single sequential rollout, given
||| matched RNG and initial state.
testParityN1 : IO Bool
testParityN1 = do
  model <- mkModel
  let rs : List Double = fakeRandomness testMaxSteps
      states : VecEnv 1 CPState     = MkVecEnv [initState]
      rss    : Vect 1 (List Double) = [rs]

  -- Thread the (linear) policy through both rollout paths; rollouts are
  -- read-only on params, so the two paths see identical weights → parity.
  (seqSteps, batchSteps) <- Control.Linear.LIO.run (do
     (MkBang ss # p1) <- rolloutEpL model initState rs testMaxSteps []
     (MkBang bs # p2) <- rolloutEpBatchedL p1 states rss testMaxSteps
     discard p2
     pure (ss, bs))
  let seqReward = sumRewards seqSteps
      batchReward = sumRewards (head batchSteps)

  checkClose "N=1 parity (total reward)" seqReward batchReward 1.0e-9

||| Stage 2 parity: batched with N=2 over (env1, rs1) and (env2, rs2)
||| produces per-env total rewards matching individual sequential
||| rollouts. Catches isolation bugs between envs in the batched code.
testParityN2 : IO Bool
testParityN2 = do
  model <- mkModel
  let rs1 = fakeRandomness testMaxSteps
      rs2                           = drop 3 (fakeRandomness (testMaxSteps + 3))
      states : VecEnv 2 CPState     = MkVecEnv [initState, initState]
      rss    : Vect 2 (List Double) = [rs1, rs2]

  (seq1, seq2, batch) <- Control.Linear.LIO.run (do
     (MkBang s1 # p1) <- rolloutEpL model initState rs1 testMaxSteps []
     (MkBang s2 # p2) <- rolloutEpL p1 initState rs2 testMaxSteps []
     (MkBang b # p3) <- rolloutEpBatchedL p2 states rss testMaxSteps
     discard p3
     pure (s1, s2, b))
  let batch1 = index 0 batch
      batch2 = index 1 batch

  ok1 <- checkClose "N=2 env0 reward parity" (sumRewards seq1) (sumRewards batch1) 1.0e-9
  ok2 <- checkClose "N=2 env1 reward parity" (sumRewards seq2) (sumRewards batch2) 1.0e-9
  pure (ok1 && ok2)

export
tests : List (IO Bool)
tests =
  [ testParityN1
  , testParityN2
  ]
