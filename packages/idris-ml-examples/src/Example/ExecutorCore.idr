||| Phase 2.1a smoke: exercises the production `UserExecutorCore`
||| interface against `TapeExecutor`. Confirms the interface + per-backend
||| `%foreign` forwarding wired up in Phase 2.1a actually resolves at
||| runtime to the suffixed tape symbols emitted by Phase 1's rename
||| headers.
|||
||| Delete after Phase 2.1c converts the live `Tensor.idr` ops to
||| these same interface methods (at which point every example
||| exercises the interface).
module Example.ExecutorCore

import BuildConfig
import Ml.Executor.Core
import Ml.Executor.Tape

||| Build two scalars, add them, read back. Forces the typechecker
||| to resolve `UserExecutorCore TapeExecutor` and the runtime to actually
||| call `_tensor_add_tape` via the interface dispatch.
addViaInterface :
  (0 ex : Executor) -> UserExecutorCore ex =>
  Double -> Double -> Double
addViaInterface ex a b =
  primItem {ex}
    (primAdd {ex} (primCreateScalar {ex} a 0) (primCreateScalar {ex} b 0))

main : IO ()
main = do
  requireMachine {m = ChosenMachine}
  putStrLn ("device: " ++ deviceName {ex=TapeExecutor})
  putStrLn ("3 + 4         = " ++ show (addViaInterface TapeExecutor 3.0 4.0))
  putStrLn ("(2 + 3) * 5   = "
    ++ show
        (primItem {ex=TapeExecutor}
          (primMul {ex=TapeExecutor}
            (primAdd {ex=TapeExecutor}
              (primCreateScalar {ex=TapeExecutor} 2.0 0)
              (primCreateScalar {ex=TapeExecutor} 3.0 0))
            (primCreateScalar {ex=TapeExecutor} 5.0 0))))
  putStrLn ("clampMin 1.5 of -2.0 = "
    ++ show
        (primItem {ex=TapeExecutor}
          (primClampMin {ex=TapeExecutor}
            (primCreateScalar {ex=TapeExecutor} (-2.0) 0)
            1.5)))
