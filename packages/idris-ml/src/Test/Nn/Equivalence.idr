module Test.Nn.Equivalence

import Control.Linear.LIO
import Data.Linear.Notation
import Data.List
import Data.Vect

import Executor
import Nn.Activation
import Nn.Linear
import Nn.Module
import Nn.Seq
import Optimizer
import Tensor
import Test.Config
import Test.Harness

-- The equivalence oracle (Phase 2 keystone). An MLP [3]→relu[4]→[2] is
-- expressed two ways and trained identically:
--
--   * NEW: an `Nn.Seq` (Linear :: relu :: Linear :: Nil), driven by
--     `forwardSeq`.
--   * REFERENCE: the bare C op chain `tlinear2d ∘ trelu ∘ tlinear2d` — the
--     exact ops the legacy `Network.applyVarBatch` chain calls. (We compare
--     against the op chain rather than importing the old `Layer/` surface
--     to dodge the `~~>`/`Nil`/`::` operator collision and the duplicate
--     registry entries pulling in both surfaces would create.)
--
-- Both sides start from identical Const params and see identical data, so —
-- same ops, same order — every per-step loss must match bitwise. Grad
-- gating (zero_grad → backward → step; off-graph params keep grad 0) keeps
-- the two param sets from cross-updating in the shared registry.

constG : {r : Nat} -> {dims : Vect r Nat} -> Double -> IO (Tensor dims TestExecutor TestDType WithGrad)
constG v = do
  t <- tensor {ex=TestExecutor} {dt=TestDType} {dims} (Const v)
  pure (retypeGrad t)

-- Shape-polymorphic sum-of-squared-error → scalar (tmseLoss is 1-D-only;
-- this is the same prim sequence over a [b,o] activation).
mse2d : {0 a, b : Nat} -> Tensor [a, b] TestExecutor TestDType WithGrad ->
        Tensor [a, b] TestExecutor TestDType WithGrad -> IO (Tensor [] TestExecutor TestDType WithGrad)
mse2d p t = ioRerun (\_ =>
  let diff = primSub {ex=TestExecutor} p.tensorPtr t.tensorPtr
      sq   = primMul {ex=TestExecutor} diff diff
  in MkTensor (primSum {ex=TestExecutor} sq) Nothing)

-- Fresh input/target each step (graph leaves are not reused across
-- backward passes — matches a real training loop's per-batch inputs).
-- Linear: consumes the net, threads it back beside the (banged) loss.
stepSeq : Optimizer TestExecutor -> (1 _ : Seq 3 2 TestExecutor TestDType WithGrad) ->
          L IO {use=1} (LPair (!* Double) (Seq 3 2 TestExecutor TestDType WithGrad))
stepSeq opt net = do
  x    <- liftIO1 (constG {dims=[2,3]} 1.0)
  tgt  <- liftIO1 (constG {dims=[2,2]} 0.5)
  (MkBang out # net') <- forwardSeq {b=2} net x
  loss <- liftIO1 (mse2d out tgt)
  d    <- liftIO1 (nativeTrainStep opt loss)
  pure1 (MkBang d # net')

-- Thread the linear net through `n` steps, accumulating losses (reversed).
loopSeq : Optimizer TestExecutor -> Nat -> List Double ->
          (1 _ : Seq 3 2 TestExecutor TestDType WithGrad) ->
          L IO {use=1} (LPair (!* (List Double)) (Seq 3 2 TestExecutor TestDType WithGrad))
loopSeq _   Z     acc net = pure1 (MkBang (reverse acc) # net)
loopSeq opt (S k) acc net = do
  (MkBang d # net') <- stepSeq opt net
  loopSeq opt k (d :: acc) net'

stepRef : Optimizer TestExecutor ->
          Tensor [4,3] TestExecutor TestDType WithGrad -> Tensor [4] TestExecutor TestDType WithGrad ->
          Tensor [2,4] TestExecutor TestDType WithGrad -> Tensor [2] TestExecutor TestDType WithGrad ->
          IO Double
stepRef opt w1 b1 w2 b2 = do
  x    <- constG {dims=[2,3]} 1.0
  tgt  <- constG {dims=[2,2]} 0.5
  h0   <- tlinear2d w1 x b1
  h1   <- trelu h0
  out  <- tlinear2d w2 h1 b2
  loss <- mse2d out tgt
  nativeTrainStep opt loss

loopN : Nat -> (Int -> IO Double) -> IO (List Double)
loopN n f = traverse f [0 .. cast {to=Int} n - 1]

trainsIdentically : IO Bool
trainsIdentically = do
  -- NEW Seq, params "seq.*"
  sw1 <- param {ex=TestExecutor} {dt=TestDType} {dims=[4, 3]} "seq.w1" (Const 0.1)
  sb1 <- param {ex=TestExecutor} {dt=TestDType} {dims=[4]}    "seq.b1" (Const 0.0)
  sw2 <- param {ex=TestExecutor} {dt=TestDType} {dims=[2, 4]} "seq.w2" (Const 0.2)
  sb2 <- param {ex=TestExecutor} {dt=TestDType} {dims=[2]}    "seq.b2" (Const 0.0)
  let net = the (Seq 3 2 TestExecutor TestDType WithGrad)
              (  the (Linear 3 4 TestExecutor TestDType WithGrad) (MkLinear sw1 sb1)
              :: the (Activation 4 4 TestExecutor TestDType WithGrad) reluA
              :: the (Linear 4 2 TestExecutor TestDType WithGrad) (MkLinear sw2 sb2)
              :: Nil )
  optS <- sgd {ex=TestExecutor} 0.05 defaultOpts
  lossSeq <- Control.Linear.LIO.run (do
    (MkBang ls # net') <- loopSeq optS 8 [] net
    discardSeq net'
    pure ls)
  -- REFERENCE op chain, params "ref.*" (identical Const init)
  rw1 <- param {ex=TestExecutor} {dt=TestDType} {dims=[4, 3]} "ref.w1" (Const 0.1)
  rb1 <- param {ex=TestExecutor} {dt=TestDType} {dims=[4]}    "ref.b1" (Const 0.0)
  rw2 <- param {ex=TestExecutor} {dt=TestDType} {dims=[2, 4]} "ref.w2" (Const 0.2)
  rb2 <- param {ex=TestExecutor} {dt=TestDType} {dims=[2]}    "ref.b2" (Const 0.0)
  optR <- sgd {ex=TestExecutor} 0.05 defaultOpts
  lossRef <- loopN 8 (\_ => stepRef optR rw1 rb1 rw2 rb2)
  let maxDiff   = foldl max 0.0 (zipWith (\a, b => abs (a - b)) lossSeq lossRef)
  let decreased = case (lossSeq, reverse lossSeq) of
                    (f :: _, l :: _) => l < f
                    _                => False
  check ("Seq MLP loss == op-chain reference bitwise + decreases (maxDiff="
         ++ show maxDiff ++ ", losses=" ++ show lossSeq ++ ")")
        (maxDiff < 1.0e-9 && decreased)

export
tests : List (IO Bool)
tests = [trainsIdentically]
