module Test.Reinforce

import Control.Linear.LIO
import Data.Linear.Notation
import Data.List
import Data.Vect
import System

import Example.Reinforce
import Gym.ClassicControl.CartPole
import Gym.Vector
import Ml.Compat.Random
import Ml.Rng
import Ml.Simple
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

||| A deterministic action sequence, replayed through `Ml.Rng.replayRng`'s
||| decision channel — `rng.choice` returns these outright, so both rollout
||| paths take identical actions and their trajectories are comparable.
||| `offset` desynchronizes two envs' sequences.
fakeDecisions : Nat -> (offset : Nat) -> List Nat
fakeDecisions n offset = go n offset
  where
    table : Vect 8 Nat
    table = [0, 1, 0, 1, 1, 0, 0, 1]

    pick : Nat -> Nat
    pick i = let m : Fin 8 = restrict 7 (cast i) in index m table

    go : Nat -> Nat -> List Nat
    go Z     _ = []
    go (S k) i = pick i :: go k (S i)

||| The batched rollout draws step-major over the envs still active at that
||| step (frozen envs skip their draw), so its decision stream is the two
||| per-env streams woven together while both run, then the survivor alone.
weave : List Nat -> List Nat -> List Nat
weave []        ys        = ys
weave xs        []        = xs
weave (x :: xs) (y :: ys) = x :: y :: weave xs ys

||| Fixed CartPole start state for the parity comparison. Not a reset draw
||| (`cpReset` randomizes per Gymnasium): the two rollout paths have to begin
||| from the same state for their returns to be comparable at all.
initState : CPState
initState = MkCP 0 0 0 0

||| Stage 1 parity: batched with N=1 must produce identical
||| per-episode total reward to a single sequential rollout, given
||| matched decisions and initial state.
testParityN1 : IO Bool
testParityN1 = do
  model <- mkModel
  rngSeq <- replayRng [] [] (fakeDecisions testMaxSteps 0)
  rngBat <- replayRng [] [] (fakeDecisions testMaxSteps 0)
  let states : VecEnv 1 CPState = MkVecEnv [initState]

  -- Thread the (linear) policy through both rollout paths; rollouts are
  -- read-only on params, so the two paths see identical weights → parity.
  (seqSteps, batchSteps) <- Control.Linear.LIO.run (do
     (MkBang ss # p1) <- rolloutEpL rngSeq model initState testMaxSteps []
     (MkBang bs # p2) <- rolloutEpBatchedL rngBat p1 states testMaxSteps
     discard p2
     pure (ss, bs))
  let seqReward = sumRewards seqSteps
      batchReward = sumRewards (head batchSteps)

  checkClose "N=1 parity (total reward)" seqReward batchReward 1.0e-9

||| Stage 2 parity: batched with N=2 over two desynchronized decision
||| sequences produces per-env total rewards matching individual sequential
||| rollouts. Catches isolation bugs between envs in the batched code.
testParityN2 : IO Bool
testParityN2 = do
  model <- mkModel
  let ds1 = fakeDecisions testMaxSteps 0
      ds2                       = fakeDecisions testMaxSteps 3
      states : VecEnv 2 CPState = MkVecEnv [initState, initState]
  rng1 <- replayRng [] [] ds1
  rng2 <- replayRng [] [] ds2

  (seq1, seq2, batch) <- Control.Linear.LIO.run (do
     (MkBang s1 # p1) <- rolloutEpL rng1 model initState testMaxSteps []
     (MkBang s2 # p2) <- rolloutEpL rng2 p1 initState testMaxSteps []
     -- The sequential runs pin each env's episode length, which is what
     -- the batched path's interleaved stream depends on.
     rngBat <- liftIO1 (replayRng [] []
                 (weave (take (length s1) ds1) (take (length s2) ds2)))
     (MkBang b # p3) <- rolloutEpBatchedL rngBat p2 states testMaxSteps
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
