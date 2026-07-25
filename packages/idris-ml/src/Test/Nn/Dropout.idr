module Test.Nn.Dropout

import Control.Linear.LIO
import Data.Linear.Notation
import Data.Vect

import Ml.Executor
import Ml.Nn.Dropout
import Ml.Nn.Module
import Ml.Nn.Seq
import Ml.Rng
import Ml.Tensor
import Test.Harness

import Test.Config

read4 : Tensor [2, 2] TestExecutor TestDType g -> List Double
read4 t = [ primItem2d {ex=TestExecutor} t.tensorPtr i j
          | (i, j) <- the (List (Int, Int)) [(0,0),(0,1),(1,0),(1,1)] ]

-- Eval mode is identity (no masking).
evalIsIdentity : IO Bool
evalIsIdentity = do
  x   <- tensor {ex=TestExecutor} {dt=TestDType} {dims=[2, 2]} (FromVect [1.0, 2.0, 3.0, 4.0])
  out <- Control.Linear.LIO.run (do
           (MkBang o # m') <- forward {b=2} (the (Dropout 2 2 TestExecutor TestDType NoGrad) (setTraining False (dropout 0.5))) x
           discard m'
           pure o)
  check ("eval-mode dropout is identity (got " ++ show (read4 out) ++ ")")
        (read4 out == [1.0, 2.0, 3.0, 4.0])

-- p=0 training mode keeps everything (scale 1/(1-0) = 1).
zeroProbKeepsAll : IO Bool
zeroProbKeepsAll = do
  x   <- tensor {ex=TestExecutor} {dt=TestDType} {dims=[2, 2]} (FromVect [1.0, 2.0, 3.0, 4.0])
  out <- Control.Linear.LIO.run (do
           (MkBang o # m') <- forward {b=2} (the (Dropout 2 2 TestExecutor TestDType NoGrad) (dropout 0.0)) x
           discard m'
           pure o)
  check ("p=0 training dropout keeps all (got " ++ show (read4 out) ++ ")")
        (read4 out == [1.0, 2.0, 3.0, 4.0])

-- `eval` must put Dropout into inference mode, the way PyTorch's `.eval()`
-- recurses into submodules. Without it inference keeps the training-time mask
-- and the 1/(1-p) survivor scaling, so a trained model scores well below what
-- it learned.
evalDisablesDropout : IO Bool
evalDisablesDropout = do
  x   <- tensor {ex=TestExecutor} {dt=TestDType} {dims=[2, 2]} (FromVect [1.0, 2.0, 3.0, 4.0])
  out <- Control.Linear.LIO.run (do
           infer <- eval (the (Dropout 2 2 TestExecutor TestDType WithGrad) (dropout 0.5))
           (MkBang o # m') <- forward {b=2} infer x
           discard m'
           pure o)
  check ("eval puts dropout in inference mode (got " ++ show (read4 out) ++ ")")
        (read4 out == [1.0, 2.0, 3.0, 4.0])

-- The same through a chain: `eval` on a `Seq` has to reach the Dropout inside
-- it, which is how every example holds one.
evalDisablesDropoutInSeq : IO Bool
evalDisablesDropoutInSeq = do
  x   <- tensor {ex=TestExecutor} {dt=TestDType} {dims=[2, 2]} (FromVect [1.0, 2.0, 3.0, 4.0])
  out <- Control.Linear.LIO.run (do
           infer <- eval (the (Seq 2 2 TestExecutor TestDType WithGrad) (dropout 0.5 ~~> Nil))
           (MkBang o # m') <- forwardSeq {b=2} infer x
           discard m'
           pure o)
  check ("eval reaches dropout inside a Seq (got " ++ show (read4 out) ++ ")")
        (read4 out == [1.0, 2.0, 3.0, 4.0])

-- A recorded mask applies exactly: kept elements carry 1/(1-p), dropped
-- elements are zero. p=0.5 on [1,2,3,4] with keep-bits [1,0,0,1] is
-- [2,0,0,8] — any masking or scaling error shows in the values.
givenBitsForward : IO Bool
givenBitsForward = do
  x    <- tensor {ex=TestExecutor} {dt=TestDType} {dims=[2, 2]} (FromVect [1.0, 2.0, 3.0, 4.0])
  msrc <- recordedMasks [[True, False, False, True]]
  out  <- Control.Linear.LIO.run (do
            (MkBang o # m') <- forward {b=2} (the (Dropout 2 2 TestExecutor TestDType NoGrad) (dropoutWith msrc 0.5)) x
            discard m'
            pure o)
  check ("recorded mask applies as 0 / 1/(1-p) (got " ++ show (read4 out) ++ ")")
        (read4 out == [2.0, 0.0, 0.0, 8.0])

-- Gradient flows through the recorded mask like through dropout: with
-- loss = sum(rowsums(out)^2), dL/dparam = 2 * rowsum * maskScale on kept
-- elements and exactly 0 on dropped ones.
givenBitsBackward : IO Bool
givenBitsBackward = do
  countBefore <- getParamCount {ex=TestExecutor}
  p    <- param {ex=TestExecutor} {dt=TestDType} {dims=[2, 2]} "test_dropout_mask_grad" (FromVect [1.0, 2.0, 3.0, 4.0])
  ones <- tensor {ex=TestExecutor} {dt=TestDType} {dims=[2]} (Const 1.0)
  zs   <- tensor {ex=TestExecutor} {dt=TestDType} {dims=[2]} Zeros
  msrc <- recordedMasks [[True, False, False, True]]
  out  <- Control.Linear.LIO.run (do
            (MkBang o # m') <- forward {b=2} (the (Dropout 2 2 TestExecutor TestDType WithGrad) (dropoutWith msrc 0.5)) p
            discard m'
            pure o)
  rows <- tmv out (retypeGrad ones)
  loss <- tmseLoss rows (retypeGrad zs)
  runBackward loss
  gs <- traverse (getParamGradAt {ex=TestExecutor} countBefore) (the (List Int) [0, 1, 2, 3])
  check ("gradient through the recorded mask (got " ++ show gs ++ ")")
        (gs == the (List Double) [8.0, 0.0, 0.0, 32.0])

export
tests : List (IO Bool)
tests = [ evalIsIdentity, zeroProbKeepsAll, evalDisablesDropout
        , evalDisablesDropoutInSeq, givenBitsForward, givenBitsBackward ]
