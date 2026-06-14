module Test.Nn.Freeze

import Control.Linear.LIO
import Data.Linear.Notation
import Data.Vect

import Executor
import Nn.Init
import Nn.Module
import Tensor
import Test.Config
import Test.Harness

-- A toy single-param layer (param-bearing, so freeze has a handle to flip).
data Lin : Nat -> Nat -> (0 _ : Executor) -> (0 _ : DType) -> (0 _ : GradMode) -> Type where
  MkLin : Tensor [2] ex dt g -> Lin i o ex dt g

Params Lin where
  params (MkLin w)   = [toParam w]
  reflect (MkLin w)  = MkBang [toParam w] # MkLin w
  castGrad (MkLin w) = MkLin (retypeGrad w)
  discard (MkLin _)  = pure ()

lin : {0 ex : Executor} -> Backend ex dt => String -> Init (Lin 2 2 ex dt WithGrad)
lin kind = do
  name <- freshChild kind
  w    <- liftIO $ param {ex} {dt} {dims=[2]} (name ++ ".weight") (Const 1.0)
  pure (MkLin w)

-- Read requires_grad through ioRerun: primRequiresGrad is pure, so reading
-- it three times around effectful freeze/unfreeze would otherwise be CSE'd
-- to a single hoisted call (the FFI-purity trap).
readRG : AnyPtr -> IO Int
readRG ptr = ioRerun (\_ => primRequiresGrad {ex=TestExecutor} ptr)

-- freeze flips the C-side requires_grad off for every param; unfreeze
-- flips it back on. Round-trips through the Frozen wrapper.
freezeRoundTrip : IO Bool
freezeRoundTrip = do
  l <- runInit (lin {ex=TestExecutor} {dt=TestDType} "fz")
  case params l of
    [w] =>
      Control.Linear.LIO.run (do
        before   <- liftIO1 (readRG w.paramPtr)
        fr       <- freeze {ex=TestExecutor} l
        after    <- liftIO1 (readRG w.paramPtr)
        m        <- unfreeze {ex=TestExecutor} fr
        restored <- liftIO1 (readRG w.paramPtr)
        discard m
        liftIO1 (check ("freeze/unfreeze flip requires_grad 1->0->1 (got "
               ++ show before ++ "/" ++ show after ++ "/" ++ show restored ++ ")")
              (before == 1 && after == 0 && restored == 1)))
    _ => check "toy layer should expose exactly one param" False

export
tests : List (IO Bool)
tests = [freezeRoundTrip]
