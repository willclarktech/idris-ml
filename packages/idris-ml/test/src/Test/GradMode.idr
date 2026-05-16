module Test.GradMode

import Data.Vect

import Harness
import Device
import Tensor
import Layer


-- weakenGrad round-trip: build a tensor with requires_grad=1, flip
-- it to 0 via weakenGrad, confirm the C-side flag agrees.
--
-- Deliberately avoids any code path that consumes RNG state
-- (e.g. `linearLayerAny`'s `xavier uniform`) so downstream
-- PRNG-seeded tests in this suite remain reproducible.

weakenGradFlipsRequiresGrad : IO Bool
weakenGradFlipsRequiresGrad = do
  let ptr = prim__createScalar 1.0 1  -- rg=1 at construction
  let t = the (Tensor (the (Vect 0 Nat) []) CPU WithGrad) (MkTensor ptr Nothing)
  let before = prim__requiresGrad t.tensorPtr
  t' <- weakenGrad t
  let after = prim__requiresGrad t'.tensorPtr
  check "weakenGrad: rg 1 -> 0" (before == 1 && after == 0)

-- freezeNetwork is purely type-level (no FFI). The compile-time
-- promise is what matters: the result type has `NoGrad`. We can't
-- exercise it cheaply at runtime (constructing a real Network would
-- pull in RNG-using layer initializers); the type signature being
-- present and the believe_me-cast compiling is the guarantee. This
-- test exists as a one-line lock to catch a future accidental
-- signature change.

freezeNetworkSignatureExists : IO Bool
freezeNetworkSignatureExists =
  let _ : (Network 2 [] 3 CPU WithGrad -> Network 2 [] 3 CPU NoGrad)
       := freezeNetwork
  in check "freezeNetwork: signature compiles" True


export
tests : List (IO Bool)
tests =
  [ weakenGradFlipsRequiresGrad
  , freezeNetworkSignatureExists
  ]
