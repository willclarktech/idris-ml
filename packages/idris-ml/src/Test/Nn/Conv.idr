module Test.Nn.Conv

import Control.Linear.LIO
import Data.Linear.Notation
import Data.List
import Data.Vect

import Ml.Executor
import Ml.Nn.Conv
import Ml.Nn.Init
import Ml.Nn.Module
import Ml.Tensor
import Test.Harness

import Test.Config

-- inC=1, outC=1, 3x3 input, 2x2 all-ones kernel, no pad, stride 1.
-- ConvOutDim 3 2 0 = 2, so output is 2x2 = 4 values; each = sum of a 2x2
-- window of ones = 4.0. Batched b=1: [1,9] -> [1,4] all 4.0.
forwardComputes : IO Bool
forwardComputes = do
  ker <- param {ex=TestExecutor} {dt=TestDType} {dims=[1, 1, 2, 2]} "cv.k" (Const 1.0)
  bia <- param {ex=TestExecutor} {dt=TestDType} {dims=[1]}          "cv.b" (Const 0.0)
  let cv = the (Conv2D 1 1 3 3 2 2 0 0 9 4 TestExecutor TestDType WithGrad) (MkConv2D ker bia)
  x   <- tensor {ex=TestExecutor} {dt=TestDType} {dims=[1, 9]} (Const 1.0)
  out <- Control.Linear.LIO.run (do
           (MkBang o # m') <- forward {b=1} cv (retypeGrad x)
           discard m'
           pure o)
  let vs = [ primItem2d {ex=TestExecutor} out.tensorPtr 0 j | j <- the (List Int) [0,1,2,3] ]
  check ("Conv2D 2x2-ones over 3x3-ones (got " ++ show vs ++ ")")
        (vs == [4.0, 4.0, 4.0, 4.0])

paramsExposed : IO Bool
paramsExposed = do
  ker <- param {ex=TestExecutor} {dt=TestDType} {dims=[1, 1, 2, 2]} "cp.k" (Const 1.0)
  bia <- param {ex=TestExecutor} {dt=TestDType} {dims=[1]}          "cp.b" (Const 0.0)
  let cv = the (Conv2D 1 1 3 3 2 2 0 0 9 4 TestExecutor TestDType WithGrad) (MkConv2D ker bia)
  check ("Params (Conv2D) = kernel,bias (got " ++ show (mapMaybe paramName (params cv)) ++ ")")
        (mapMaybe paramName (params cv) == ["cp.k", "cp.b"])

smartCtorNames : IO Bool
smartCtorNames = do
  _ <- runInit $ scoped "cnn"
         (conv2d {ex=TestExecutor} {dt=TestDType} {g=WithGrad} {inC=1} {outC=2} {h=4} {w=4} {kH=3} {kW=3} {padH=0} {padW=0})
  cnt <- getParamCount {ex=TestExecutor}
  names <- traverse (\i => getParamName {ex=TestExecutor} i) [0 .. cnt - 1]
  check "conv2d registers cnn.conv2d_0.weight + .bias"
        (("cnn.conv2d_0.weight" `elem` names) && ("cnn.conv2d_0.bias" `elem` names))

-- Conv1D, inC=1 outC=1 len=3 kL=2 pad=0, kernel [1,2], bias 0, input
-- [1,2,3]. Cross-correlation (PyTorch conv1d): out[0]=1·1+2·2=5,
-- out[1]=2·1+3·2=8. ConvOutDim 3 2 0 = 2. Batched b=1: [1,3] -> [1,2].
forward1dComputes : IO Bool
forward1dComputes = do
  ker <- param {ex=TestExecutor} {dt=TestDType} {dims=[1, 1, 2]} "c1.k" (FromVect [1.0, 2.0])
  bia <- param {ex=TestExecutor} {dt=TestDType} {dims=[1]}       "c1.b" (Const 0.0)
  let cv = the (Conv1D 1 1 3 2 0 3 2 TestExecutor TestDType WithGrad) (MkConv1D ker bia)
  x   <- tensor {ex=TestExecutor} {dt=TestDType} {dims=[1, 3]} (FromVect [1.0, 2.0, 3.0])
  out <- Control.Linear.LIO.run (do
           (MkBang o # m') <- forward {b=1} cv (retypeGrad x)
           discard m'
           pure o)
  let vs = [ primItem2d {ex=TestExecutor} out.tensorPtr 0 j | j <- the (List Int) [0,1] ]
  check ("Conv1D [1,2] over [1,2,3] (got " ++ show vs ++ ")")
        (vs == [5.0, 8.0])

smartCtor1dNames : IO Bool
smartCtor1dNames = do
  _ <- runInit $ scoped "cnn1"
         (conv1d {ex=TestExecutor} {dt=TestDType} {g=WithGrad} {inC=1} {outC=2} {len=8} {kL=3} {pad=0})
  cnt <- getParamCount {ex=TestExecutor}
  names <- traverse (\i => getParamName {ex=TestExecutor} i) [0 .. cnt - 1]
  check "conv1d registers cnn1.conv1d_0.weight + .bias"
        (("cnn1.conv1d_0.weight" `elem` names) && ("cnn1.conv1d_0.bias" `elem` names))

export
tests : List (IO Bool)
tests = [forwardComputes, paramsExposed, smartCtorNames, forward1dComputes, smartCtor1dNames]
