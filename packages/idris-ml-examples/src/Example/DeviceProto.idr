||| Phase 0a smoke runner — calls `Device.Proto.demoRunOnTape` to
||| confirm the `UserDeviceCore` interface + `TapeDev` instance compiles
||| AND executes correctly (interface methods forward to the live tape
||| C primitives at runtime).
|||
||| Not a real example; delete after Phase 2.x folds the interface
||| pattern into `Tensor.idr` proper.
module Example.DeviceProto

import Device.Proto

main : IO ()
main = demoRunOnTape
