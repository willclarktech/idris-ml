module Test.Nn.Dropout

import Control.Linear.LIO
import Data.Linear.Notation
import Data.Vect

import Ml.Executor
import Ml.Nn.Dropout
import Ml.Nn.Module
import Ml.Nn.Seq
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

export
tests : List (IO Bool)
tests = [evalIsIdentity, zeroProbKeepsAll, evalDisablesDropout, evalDisablesDropoutInSeq]
