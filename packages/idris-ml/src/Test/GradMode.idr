module Test.GradMode

import Data.Vect

import Executor
import Tensor
import Test.Config
import Test.Harness

-- weakenGrad round-trip: build a tensor with requires_grad=1, flip
-- it to 0 via weakenGrad, confirm the C-side flag agrees.
--
-- Deliberately avoids any code path that consumes RNG state so
-- downstream PRNG-seeded tests in this suite remain reproducible.
-- (Model-level freeze/unfreeze on the Nn `Frozen` surface is covered
-- by Test.Nn.Freeze.)

weakenGradFlipsRequiresGrad : IO Bool
weakenGradFlipsRequiresGrad = do
  let ptr    = primCreateScalar {ex=TestExecutor} 1.0 1  -- rg=1 at construction
  let t      = the (Tensor (the (Vect 0 Nat) []) TestExecutor TestDType WithGrad) (MkTensor ptr Nothing)
  let before = primRequiresGrad {ex=TestExecutor} t.tensorPtr
  t' <- weakenGrad t
  let after = primRequiresGrad {ex=TestExecutor} t'.tensorPtr
  check "weakenGrad: rg 1 -> 0" (before == 1 && after == 0)

export
tests : List (IO Bool)
tests =
  [ weakenGradFlipsRequiresGrad
  ]
