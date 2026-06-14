module Test.Nn.Init

import Data.Vect

import Test.Harness
import Executor
import Tensor
import Nn.Init
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

export
tests : List (IO Bool)
tests = [namesAutoNumber, namedPins, registryRoundTrip]
