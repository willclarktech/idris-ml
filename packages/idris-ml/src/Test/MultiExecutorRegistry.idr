||| Acceptance test for the per-device parameter registry.
|||
||| Each backend keeps its own `static param_registry` in its
||| translation unit. `tparamScalar {ex}` registers through
||| `primParamRegister {ex}` (→ `param_register_return_<b>`) and
||| `getParamCount {ex}` queries `param_count_<b>`, so a param
||| registered on one device must be invisible to another device's
||| registry.
|||
||| This is the regression test for the latent bug the unified-name
||| alias machinery masked: before the fix, registering a
||| `(TorchExecutor TCpu)` param routed through the unified
||| `param_register_return` symbol — link-aliased to the *primary*
||| backend — so it landed in tape's registry instead of torch's.
|||
||| Requires the multi-backend build (tape + torch + mlx all linked):
|||
|||     make BACKEND=torch,tape,mlx test-multi
|||
||| The deltas (not absolute counts) are asserted so the test is
||| robust to params other suites register earlier in the run.
module Test.MultiExecutorRegistry

import Data.Vect

import Executor
import Tensor
import Test.Harness

||| Register one param on `TorchExecutor TCpu` and assert torch's registry
||| count grows by exactly one while tape's count is unchanged —
||| proving the two registries are independent and that registration
||| dispatches on the type-level device, not the link-time primary.
registryIsolation : IO Bool
registryIsolation = do
  tapeBefore  <- getParamCount {ex=TapeExecutor}
  torchBefore <- getParamCount {ex=TorchExecutor TCpu}
  _ <- tparamScalar {ex=TorchExecutor TCpu} {dt = F64} "mdr_torch_only" 0.5
  tapeAfter   <- getParamCount {ex=TapeExecutor}
  torchAfter  <- getParamCount {ex=TorchExecutor TCpu}
  ok1 <- check "torch registry grew by 1" (torchAfter == torchBefore + 1)
  ok2 <- check "tape registry unaffected" (tapeAfter == tapeBefore)
  pure (ok1 && ok2)

||| The mirror direction: a tape param must not appear in torch's
||| registry either.
registryIsolationTape : IO Bool
registryIsolationTape = do
  tapeBefore  <- getParamCount {ex=TapeExecutor}
  torchBefore <- getParamCount {ex=TorchExecutor TCpu}
  _ <- tparamScalar {ex=TapeExecutor} {dt = F64} "mdr_tape_only" 0.5
  tapeAfter   <- getParamCount {ex=TapeExecutor}
  torchAfter  <- getParamCount {ex=TorchExecutor TCpu}
  ok1 <- check "tape registry grew by 1" (tapeAfter == tapeBefore + 1)
  ok2 <- check "torch registry unaffected" (torchAfter == torchBefore)
  pure (ok1 && ok2)

export
tests : List (IO Bool)
tests =
  [ registryIsolation
  , registryIsolationTape
  ]
