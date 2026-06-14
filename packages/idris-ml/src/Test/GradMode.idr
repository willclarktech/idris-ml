module Test.GradMode

import Data.Vect

import Test.Harness
import Executor
import Tensor
import Layer
import Test.Config

-- weakenGrad round-trip: build a tensor with requires_grad=1, flip
-- it to 0 via weakenGrad, confirm the C-side flag agrees.
--
-- Deliberately avoids any code path that consumes RNG state
-- (e.g. `linearLayerAny`'s `xavier uniform`) so downstream
-- PRNG-seeded tests in this suite remain reproducible.

weakenGradFlipsRequiresGrad : IO Bool
weakenGradFlipsRequiresGrad = do
  let ptr = primCreateScalar {ex=TestExecutor} 1.0 1  -- rg=1 at construction
  let t = the (Tensor (the (Vect 0 Nat) []) TestExecutor TestDType WithGrad) (MkTensor ptr Nothing)
  let before = primRequiresGrad {ex=TestExecutor} t.tensorPtr
  t' <- weakenGrad t
  let after = primRequiresGrad {ex=TestExecutor} t'.tensorPtr
  check "weakenGrad: rg 1 -> 0" (before == 1 && after == 0)

-- freezeNetwork + unfreezeNetwork round-trip on a parameter-free
-- network (single tanh activation). Walks the layer chain, calls each
-- layer's freezeLayer / unfreezeLayer, and confirms the types compose
-- end-to-end. Avoids RNG-using layer constructors so downstream
-- PRNG-seeded tests stay reproducible.

freezeUnfreezeRoundTrip : IO Bool
freezeUnfreezeRoundTrip = do
  let net : Network 4 [] 4 TestExecutor TestDType WithGrad
      net = OutputLayer (the (AnyLayer 4 4 TestExecutor TestDType WithGrad) tanhLayerAny)
  frozen <- freezeNetwork net
  -- frozen : Network 4 [] 4 TestExecutor TestDType NoGrad — compile-checked
  _ <- unfreezeNetwork frozen
  -- unfrozen : Network 4 [] 4 TestExecutor TestDType WithGrad — compile-checked
  check "freezeNetwork / unfreezeNetwork round-trip typechecks" True

export
tests : List (IO Bool)
tests =
  [ weakenGradFlipsRequiresGrad
  , freezeUnfreezeRoundTrip
  ]
