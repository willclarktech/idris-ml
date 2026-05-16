||| Phase 0a prototype for the pluggable-Device refactor.
|||
||| See `docs/develop/design-decisions.md` and the project plan:
||| TODO row "Pluggable / dependent `Device` for user-supplied backends".
|||
||| This module is a SIDE FILE — it exists to prove the interface +
||| forwarding pattern works against today's FFI surface without
||| touching the live `Tensor` record. Phase 2.x will fold this into
||| `Tensor.idr` for real; for now it answers the Phase 0a question
||| "does Idris-2 resolve `UserDeviceCore d`-constrained ops against
||| an instance that forwards to existing tape primitives?".
module Device.Proto

import Data.Vect


----------------------------------------------------------------------
-- Smallest viable interface slice (~10 ops, lifecycle + arithmetic)
----------------------------------------------------------------------

||| Core lifecycle + elementwise primitives every backend must provide.
||| Phase 0a slice: just enough surface to exercise interface
||| resolution + forwarding. Real refactor (Phase 2.1+) will widen
||| this into UserDeviceCore / UserDeviceArith / UserDeviceLinear /
||| ... sub-interfaces; this is the smoke version.
public export
interface UserDeviceCore (0 d : Type) where
  ||| Human-readable tag, e.g. "tape", "torch", "mlx", "mybackend".
  ||| Used for logs + the `Show` instance once `Tensor` re-routes
  ||| through `UserDevice`.
  deviceName  : String

  ||| Scalar-handle constructor. `requiresGrad` arg matches the
  ||| existing `tensor_create_scalar` C signature (0 / 1).
  primScalar  : Double -> Int -> AnyPtr

  ||| Pure-Double readback. Mirrors `prim__item`.
  primItem    : AnyPtr -> Double

  ||| Pure-arithmetic ops; all consume + produce backend-side handles.
  primAdd     : AnyPtr -> AnyPtr -> AnyPtr
  primSub     : AnyPtr -> AnyPtr -> AnyPtr
  primMul     : AnyPtr -> AnyPtr -> AnyPtr
  primDiv     : AnyPtr -> AnyPtr -> AnyPtr
  primNeg     : AnyPtr -> AnyPtr
  primExp     : AnyPtr -> AnyPtr
  primLog     : AnyPtr -> AnyPtr


----------------------------------------------------------------------
-- Built-in instance: TapeDev (forwards to existing libidrisml symbols)
----------------------------------------------------------------------
--
-- These %foreign declarations re-bind today's tape-backend C symbols
-- under prototype names. Phase 1 will rename the C symbols themselves
-- (`tensor_add` -> `tensor_add_tape`) so all three backends can be
-- linked simultaneously; for Phase 0a we just reuse the existing
-- single-dylib symlink — the goal is to prove the interface plumbing,
-- not to rebuild the link model.

%foreign "C:tensor_create_scalar,libidrisml"
prim__scalar_tape : Double -> Int -> AnyPtr

%foreign "C:tensor_item,libidrisml"
prim__item_tape : AnyPtr -> Double

%foreign "C:tensor_add,libidrisml"
prim__add_tape : AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:tensor_sub,libidrisml"
prim__sub_tape : AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:tensor_mul,libidrisml"
prim__mul_tape : AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:tensor_div,libidrisml"
prim__div_tape : AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:tensor_neg,libidrisml"
prim__neg_tape : AnyPtr -> AnyPtr

%foreign "C:tensor_exp,libidrisml"
prim__exp_tape : AnyPtr -> AnyPtr

%foreign "C:tensor_log,libidrisml"
prim__log_tape : AnyPtr -> AnyPtr


||| Empty type — the "instance head" for the tape backend. A user
||| backend would declare its own equivalent (`data MyDev = MD`) and
||| supply its own `UserDeviceCore` instance.
public export
data TapeDev : Type where MkTapeDev : TapeDev

public export
UserDeviceCore TapeDev where
  deviceName  = "tape"
  primScalar  = prim__scalar_tape
  primItem    = prim__item_tape
  primAdd     = prim__add_tape
  primSub     = prim__sub_tape
  primMul     = prim__mul_tape
  primDiv     = prim__div_tape
  primNeg     = prim__neg_tape
  primExp     = prim__exp_tape
  primLog     = prim__log_tape


----------------------------------------------------------------------
-- Polymorphic-over-device demo functions
----------------------------------------------------------------------

||| Generic-in-d: build a scalar handle via the interface, read it back.
||| Phase 0a smoke: confirms `UserDeviceCore d => ...` resolves cleanly
||| when called with an explicit instance.
public export
roundTrip : (0 d : Type) -> UserDeviceCore d => Double -> Double
roundTrip d x = primItem {d} (primScalar {d} x 0)

||| `a + b` via the interface, in/out via primScalar / primItem.
public export
addViaInterface :
  (0 d : Type) -> UserDeviceCore d =>
  Double -> Double -> Double
addViaInterface d a b =
  primItem {d}
    (primAdd {d} (primScalar {d} a 0) (primScalar {d} b 0))

||| `(a + b) * c` — composes three ops; demonstrates interface
||| methods chain without per-call dictionary noise at the call site.
public export
fmaViaInterface :
  (0 d : Type) -> UserDeviceCore d =>
  Double -> Double -> Double -> Double
fmaViaInterface d a b c =
  primItem {d}
    (primMul {d}
      (primAdd {d} (primScalar {d} a 0) (primScalar {d} b 0))
      (primScalar {d} c 0))

||| Same composition, written generically — what real Tensor ops will
||| look like post-Phase-2: `addViaInterface` etc. are not specialized
||| to TapeDev; user-backend instances drop in.
public export
demoRunOnTape : IO ()
demoRunOnTape = do
  putStrLn ("device: " ++ deviceName {d = TapeDev})
  putStrLn ("roundTrip 7.0  = "       ++ show (roundTrip TapeDev 7.0))
  putStrLn ("3 + 4          = "       ++ show (addViaInterface TapeDev 3.0 4.0))
  putStrLn ("(2 + 3) * 5    = "       ++ show (fmaViaInterface TapeDev 2.0 3.0 5.0))
