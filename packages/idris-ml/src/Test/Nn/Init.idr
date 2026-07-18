module Test.Nn.Init

import Data.Vect

import Ml.Executor
import Ml.Nn.Init
import Ml.Tensor
import Test.Harness

import Test.Config

-- scoped + freshChild auto-number siblings per (scope, kind); a nested
-- scope restarts numbering under its own path.
namesAutoNumber : IO Bool
namesAutoNumber = do
  ns <- runInit $ scoped "actor" $ do
    a <- freshChild "linear"
    b <- freshChild "linear"
    c <- scoped "head" (freshChild "linear")
    pure (the (List String) [a, b, c])
  check ("scoped/freshChild auto-number (got " ++ show ns ++ ")")
        (ns == ["actor.linear_0", "actor.linear_1", "actor.head.linear_0"])

-- `named` pins the next child verbatim; numbering resumes afterwards.
namedPins : IO Bool
namedPins = do
  ns <- runInit $ scoped "model" $ do
    e <- named "embed" (freshChild "linear")
    f <- freshChild "linear"
    pure (the (List String) [e, f])
  check ("named pins, then numbering resumes (got " ++ show ns ++ ")")
        (ns == ["model.embed", "model.linear_0"])

-- The derived name flows through the unchanged C registry: register a real
-- param under an Init-derived name and read it back via getParamName.
registryRoundTrip : IO Bool
registryRoundTrip = do
  nm <- runInit $ scoped "rt" (freshChild "linear")
  _  <- param {ex=TestExecutor} {dt=TestDType} {dims=[2, 2]} (nm ++ ".weight") (Const 1.0)
  cnt <- getParamCount {ex=TestExecutor}
  names <- traverse (\i => getParamName {ex=TestExecutor} i) [0 .. cnt - 1]
  check ("registry holds the derived name (looking for rt.linear_0.weight)")
        ("rt.linear_0.weight" `elem` names)

-- scopedChild numbers a *container* and nests its children; a second
-- container of the same kind gets the next index.
nestsComposites : IO Bool
nestsComposites = do
  ns <- runInit $ do
    a <- scopedChild "block" (do
           x <- freshChild "linear"
           y <- freshChild "linear"
           pure (the (List String) [x, y]))
    b <- scopedChild "block" (freshChild "linear")
    pure (a ++ [b])
  check ("scopedChild numbers + nests composites (got " ++ show ns ++ ")")
        (ns == ["block_0.linear_0", "block_0.linear_1", "block_1.linear_0"])

export
tests : List (IO Bool)
tests = [namesAutoNumber, namedPins, registryRoundTrip, nestsComposites]
