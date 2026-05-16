||| Pluggable-Device interface. Phase 2.1 of the refactor; see
||| `docs/develop/design-decisions.md` "Pluggable Device via sliced
||| `UserDevice` interfaces" for the design.
|||
||| Phase 2.1 ships the **lifecycle + arithmetic** slice (~20 ops).
||| Later slices (2.2-2.5) extend with linear / NN / conv / tape ops.
|||
||| Users implementing their own backend declare an empty type and
||| an instance:
|||
|||   data MyDev : Type where MD : MyDev
|||
|||   UserDeviceCore MyDev where
|||     primAdd = prim__addMine
|||     ...
|||
||| The built-in `TapeDev` / `TorchDev` / `MlxDev` (in `Device.Tape`,
||| `Device.Torch`, `Device.Mlx`) forward to the per-backend C
||| symbols emitted by Phase 1's rename headers
||| (`tensor_add_tape` / `tensor_add_torch` / `tensor_add_mlx`).
module Device.Core


----------------------------------------------------------------------
-- `Device` kind alias
--
-- `Device` is a 0-quantity alias for `Type`. Tensor's `d` phantom is
-- declared as `(0 d : Device)`, which is exactly `(0 d : Type)`
-- underneath but reads as "d is a device tag" at every kind-binder
-- site. No type-system enforcement: nothing stops a caller writing
-- `Tensor [4] Bool`. But construction (`primCreate*`) and operations
-- (`tadd` etc.) both require `UserDeviceCore d =>`, so non-device
-- `d`s can be declared but never inhabited or operated on.
--
-- See `docs/develop/design-decisions.md` "Open `d` kind: why
-- `Device = Type` instead of a real sub-kind" for the alternatives
-- considered and why we kept it open.
----------------------------------------------------------------------

public export
0 Device : Type
Device = Type


----------------------------------------------------------------------
-- UserDeviceCore — lifecycle + arithmetic slice
----------------------------------------------------------------------

||| Phase 2.1 interface: the ~20 ops needed for tensor lifecycle and
||| elementwise arithmetic. Later phases add `UserDeviceLinear`,
||| `UserDeviceNN`, `UserDeviceConv`, `UserDeviceTape` slices.
public export
interface UserDeviceCore (0 d : Device) where
  ||| Human-readable device tag: "tape", "torch", "mlx", "mybackend".
  ||| Used in logs and `Show Device`-style stringification.
  deviceName : String

  -- Lifecycle ---------------------------------------------------------
  ||| Allocate a 0-rank tensor with the given value and grad flag.
  ||| `requires_grad` is 0 or 1.
  primCreateScalar : Double -> Int -> AnyPtr

  ||| Allocate a multi-dimensional tensor. Args: data buffer, shape
  ||| buffer, rank, requires_grad.
  primCreate : AnyPtr -> AnyPtr -> Int -> Int -> AnyPtr

  ||| Release a tensor handle. No-op on backends that GC.
  primFree : AnyPtr -> ()

  ||| Read a 0-rank tensor's value.
  primItem : AnyPtr -> Double

  ||| Deep-copy a tensor (new handle, same shape and values, fresh
  ||| autograd node).
  primClone : AnyPtr -> AnyPtr

  -- Elementwise arithmetic --------------------------------------------
  primAdd     : AnyPtr -> AnyPtr -> AnyPtr
  primSub     : AnyPtr -> AnyPtr -> AnyPtr
  primMul     : AnyPtr -> AnyPtr -> AnyPtr
  primDiv     : AnyPtr -> AnyPtr -> AnyPtr
  primNeg     : AnyPtr -> AnyPtr
  primAbs     : AnyPtr -> AnyPtr
  primExp     : AnyPtr -> AnyPtr
  primLog     : AnyPtr -> AnyPtr
  primSqrt    : AnyPtr -> AnyPtr
  primPow     : AnyPtr -> AnyPtr -> AnyPtr
  primSigmoid : AnyPtr -> AnyPtr
  primTanh    : AnyPtr -> AnyPtr

  -- Scalar arithmetic -------------------------------------------------
  primAddScalar : AnyPtr -> Double -> AnyPtr
  primMulScalar : AnyPtr -> Double -> AnyPtr
  primClampMin  : AnyPtr -> Double -> AnyPtr
