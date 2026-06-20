module Test.Nn.Derive

import Data.List
import Data.Vect
import Language.Reflection
import Language.Reflection.Util

import Executor
import Nn.Derive
import Nn.Linear
import Nn.Module
import Tensor
import Test.Config
import Test.Harness

%language ElabReflection

-- Call-site-local rule wrapper (an imported rule passed by value to
-- `derive` leaves a stuck elaborator script; a current-package wrapper
-- calling the imported pure `GCastImpl` reduces fine).
gcast : List Name -> ParamTypeInfo -> Res (List TopLevel)
gcast nms p = GCastImpl nms p

-- A nested sub-record (one leaf layer). Derive its GCast first so the
-- composite below can recurse into it via instance resolution.
record Sub (i, o : Nat) (0 ex : Executor) (0 dt : DType) (0 g : GradMode) where
  constructor MkSub
  inner : Linear i o ex dt g

%runElab derive `{Sub} [gcast]

-- A composite exercising every field shape the deriver must classify:
-- a bare `Tensor`, a plain leaf layer, a `Vect` of leaf layers, and a
-- nested derived record.
record Composite (i, o, n : Nat) (0 ex : Executor) (0 dt : DType) (0 g : GradMode) where
  constructor MkComposite
  bias  : Tensor [o] ex dt g
  head  : Linear i o ex dt g
  stack : Vect n (Linear o o ex dt g)
  sub   : Sub i o ex dt g

%runElab derive `{Composite} [gcast]

mkLin : {0 ex : Executor} -> Backend ex dt => {i, o : Nat} -> String -> IO (Linear i o ex dt WithGrad)
mkLin pfx = do
  w <- param {ex} {dt} {dims = [o, i]} (pfx ++ ".weight") (Const 1.0)
  b <- param {ex} {dt} {dims = [o]}    (pfx ++ ".bias")   (Const 0.0)
  pure (MkLinear w b)

mkComposite : IO (Composite 2 3 2 TestExecutor TestDType WithGrad)
mkComposite = do
  bias <- param {ex = TestExecutor} {dt = TestDType} {dims = [3]} "bias" (Const 0.0)
  hd   <- mkLin {i = 2} {o = 3} "head"
  s0   <- mkLin {i = 3} {o = 3} "stk0"
  s1   <- mkLin {i = 3} {o = 3} "stk1"
  inr  <- mkLin {i = 2} {o = 3} "sub.inner"
  pure (MkComposite bias hd [s0, s1] (MkSub inr))

-- Every leaf param, in declaration order, flattened.
expectedNames : List String
expectedNames = sort
  [ "bias"
  , "head.weight", "head.bias"
  , "stk0.weight", "stk0.bias"
  , "stk1.weight", "stk1.bias"
  , "sub.inner.weight", "sub.inner.bias"
  ]

namesOf : List SomeParam -> List String
namesOf = sort . mapMaybe paramName

-- The derived `gparams` must collect exactly the leaf params across all
-- four field shapes (bare Tensor, plain layer, Vect-of-layers, nested
-- record) — set membership, not order.
derivedGparamsExact : IO Bool
derivedGparamsExact = do
  comp <- mkComposite
  let got = namesOf (gparams comp)
  check ("derived gparams = all leaf params (got " ++ show got ++ ")")
        (got == expectedNames)

-- The derived `gcastGrad` is a pure phantom retype: round-tripping
-- WithGrad -> NoGrad -> WithGrad must leave the param set untouched and
-- typecheck at every shape.
derivedGcastRoundtrips : IO Bool
derivedGcastRoundtrips = do
  comp <- mkComposite
  let inf  = gcastGrad {g' = NoGrad} comp
  let back = gcastGrad {g' = WithGrad} inf
  check "derived gcastGrad preserves the param set across WithGrad->NoGrad->WithGrad"
        (namesOf (gparams inf) == expectedNames && namesOf (gparams back) == expectedNames)

export
tests : List (IO Bool)
tests = [derivedGparamsExact, derivedGcastRoundtrips]
