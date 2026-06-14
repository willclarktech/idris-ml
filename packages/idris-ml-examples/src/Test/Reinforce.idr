module Test.Reinforce

import Data.List
import Data.Vect
import System

import BuildConfig
import Compat.Random
import Example.Reinforce
import Executor
import Gym.ClassicControl.CartPole
import Gym.Vector
import Layer.Activation
import Layer.Core
import Layer.Linear
import Tensor
import Test.Harness

-- Short step budget so test runs are tape-friendly. 200 steps × N=2
-- × 2 paths blew up under whatever backend the test linked against.
testMaxSteps : Nat
testMaxSteps = 20

||| Build a fresh deterministic REINFORCE policy for use in both rollout
||| paths. Initialization reads from C-side RNG state, so we build it
||| ONCE per test and reuse the same Network for sequential and batched
||| rollouts.
mkModel : IO (Network 4 [16, 16] 2 ExampleExecutor ExampleDType WithGrad)
mkModel = do
  srand 12345  -- deterministic init for parity reproducibility
  ll1 <- linearLayerAny {i=4} {o=16} "test_ll1"
  ll2 <- linearLayerAny {i=16} {o=2} "test_ll2"
  pure (ll1 ~~> tanhLayerAny ~~> OutputLayer ll2)

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
      states : VecEnv 1 CPState = MkVecEnv [initState]
      rss : Vect 1 (List Double) = [rs]

  seqSteps <- rolloutEp model initState rs testMaxSteps []
  batchSteps <- rolloutEpBatched model states rss testMaxSteps
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
      rs2 = drop 3 (fakeRandomness (testMaxSteps + 3))
      states : VecEnv 2 CPState = MkVecEnv [initState, initState]
      rss : Vect 2 (List Double) = [rs1, rs2]

  seq1 <- rolloutEp model initState rs1 testMaxSteps []
  seq2 <- rolloutEp model initState rs2 testMaxSteps []
  batch <- rolloutEpBatched model states rss testMaxSteps
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
