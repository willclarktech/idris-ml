module Test.Nn.Group

import Control.Linear.LIO
import Data.Linear.Notation
import Data.List
import Data.Vect

import Ml.Executor
import Ml.Nn.Group
import Ml.Nn.Init
import Ml.Nn.Module
import Ml.Tensor
import Test.Harness

import Test.Config

-- A toy single-param layer so groupOf has something to enumerate.
data Lin : Nat -> Nat -> (0 _ : Executor) -> (0 _ : DType) -> (0 _ : GradMode) -> Type where
  MkLin : Tensor [2] ex dt g -> Lin i o ex dt g

Params Lin where
  params (MkLin w)   = [toParam w]
  reflect (MkLin w)  = MkBang [toParam w] # MkLin w
  castGrad (MkLin w) = MkLin (retypeGrad w)
  discard (MkLin _)  = pure ()

-- Smart constructor: registers one param under the Init-derived name.
lin : {0 ex : Executor} -> Backend ex dt => String -> Init (Lin 2 2 ex dt WithGrad)
lin kind = do
  name <- freshChild kind
  w    <- liftIO $ param {ex} {dt} {dims=[2]} (name ++ ".weight") (Const 1.0)
  pure (MkLin w)

-- Two submodels whose scope names overlap as substrings ("actor" is a
-- prefix of "actorX"): the exact-set groupOf must keep them disjoint where
-- a substring match would leak "actorX"'s param into "actor"'s group.
buildPair : IO (Lin 2 2 TestExecutor TestDType WithGrad, Lin 2 2 TestExecutor TestDType WithGrad)
buildPair = runInit $ do
  a <- scoped "actor"  (lin "linear")
  b <- scoped "actorX" (lin "linear")
  pure (a, b)

groupsAreExact : IO Bool
groupsAreExact = do
  (a, b) <- buildPair
  let ga = groupOf a
  check ("groupOf actor is exactly its own param (got " ++ show ga ++ ")")
        (ga == ["actor.linear_0.weight"])

groupsDontLeak : IO Bool
groupsDontLeak = do
  (a, b) <- buildPair
  let ga = groupOf a
  let gb = groupOf b
  check "groupOf actor excludes actorX's param (no substring leak)"
        (not ("actorX.linear_0.weight" `elem` ga) && all (\n => not (n `elem` ga)) gb)

-- The linear twin: reflectNames consumes a single-owner (linear) model and
-- returns its exact registry names beside the threaded-back model — the source
-- that feeds `restrictTo` for optimizer ownership where the model is a `1`
-- resource. Same exact-set guarantee as groupOf, on the linear path.
reflectNamesIsExact : IO Bool
reflectNamesIsExact = Control.Linear.LIO.run $ do
  a <- runInitL (scoped "rn" (lin {ex=TestExecutor} {dt=TestDType} "linear"))
  let (MkBang names # a') = reflectNames a
  discard a'
  liftIO1 (check ("reflectNames returns the model's exact names (got " ++ show names ++ ")")
                 (names == ["rn.linear_0.weight"]))

export
tests : List (IO Bool)
tests = [groupsAreExact, groupsDontLeak, reflectNamesIsExact]
