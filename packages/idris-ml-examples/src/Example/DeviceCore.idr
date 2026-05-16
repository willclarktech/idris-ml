||| Phase 2.1a smoke: exercises the production `UserDeviceCore`
||| interface against `TapeDev`. Confirms the interface + per-backend
||| `%foreign` forwarding wired up in Phase 2.1a actually resolves at
||| runtime to the suffixed tape symbols emitted by Phase 1's rename
||| headers.
|||
||| Delete after Phase 2.1c converts the live `Tensor.idr` ops to
||| these same interface methods (at which point every example
||| exercises the interface).
module Example.DeviceCore

import Device.Core
import Device.Tape


||| Build two scalars, add them, read back. Forces the typechecker
||| to resolve `UserDeviceCore TapeDev` and the runtime to actually
||| call `_tensor_add_tape` via the interface dispatch.
addViaInterface :
  (0 d : Type) -> UserDeviceCore d =>
  Double -> Double -> Double
addViaInterface d a b =
  primItem {d}
    (primAdd {d} (primCreateScalar {d} a 0) (primCreateScalar {d} b 0))

main : IO ()
main = do
  putStrLn ("device: " ++ deviceName {d = TapeDev})
  putStrLn ("3 + 4         = " ++ show (addViaInterface TapeDev 3.0 4.0))
  putStrLn ("(2 + 3) * 5   = "
    ++ show
        (primItem {d = TapeDev}
          (primMul {d = TapeDev}
            (primAdd {d = TapeDev}
              (primCreateScalar {d = TapeDev} 2.0 0)
              (primCreateScalar {d = TapeDev} 3.0 0))
            (primCreateScalar {d = TapeDev} 5.0 0))))
  putStrLn ("clampMin 1.5 of -2.0 = "
    ++ show
        (primItem {d = TapeDev}
          (primClampMin {d = TapeDev}
            (primCreateScalar {d = TapeDev} (-2.0) 0)
            1.5)))
