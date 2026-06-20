||| The `Tensor` autograd-handle record + aliases, RuntimeDType
||| instances, executor migration, grad-mode retype, and `ioRerun`.
module Tensor.Core

import Data.Vect

import Array
import DType.Core
import Executor
import GradMode
import Tensor.Handle
import Tensor.Internal

----------------------------------------------------------------------
-- Per-dtype creation primitives + RuntimeDType F32 / F64 instances
--
-- Same scheme-wrapper template as the unsuffixed primitives above,
-- but bound to the per-dtype C symbols (tensor_create_scalar_f32 vs
-- _f64, etc.). The RuntimeDType F32 / F64 instances at the bottom
-- bind the typeclass methods to these primitives so smart constructors
-- with `RuntimeDType dt =>` dispatch statically based on the type-level
-- dtype.
--
-- Layout: scalar / create / 1d / 2d / param_{1,2,3,4}d / state_{1,2}d
-- — each group has _f32 then _f64.
----------------------------------------------------------------------

-- tensor_create_scalar

-- tensor_create

-- tensor_create_2d

-- Per-dtype cast primitives. Backend support mirrors the create
-- primitives: mlx/torch implement both; tape implements _f64 (no-op
-- alias today, since the only valid source dtype is F64) and aborts
-- on _f32. Source dtype is read from the handle on the C side.

-- RuntimeDType instances — the runtime dtype tag passed across the
-- Idris↔C FFI boundary. Kind-major, precision-minor layout: each kind
-- (U/I/F/BF/TF/...) gets 4 lanes for 8/16/32/64-bit variants;
-- bit_width = 8 << (tag & 3) for numeric families. Tag 0 is reserved
-- as invalid so a zero-initialized dtag traps at the dispatch's
-- `default:` arm instead of silently meaning F32. Sub-byte families
-- (24-31) are reserved for future quantization dtypes (U4/I4/NF4/
-- ternary/MX) — those don't fit the `8 << lane` formula and live in
-- families with named lanes.
--
-- Layout (defined slots; reserved slots omitted):
--   0  reserved (invalid; zero-init traps)
--   1  Bool
--   4  U8                          (family 1: U, lane 0 = 8-bit)
--   8  I8     9  I16  10 I32  11 I64    (family 2: I)
--   13 F16   14  F32  15 F64           (family 3: F; lane 0 / F8 reserved)
--   17 BF16                            (family 4: BF; lanes for BF8/32/64 reserved)
--   (family 5: TF — TF8/16/32/64 — all reserved)
--   24 Binary, 25 Ternary              (family 6: sub-byte quant — named
--                                       lanes, not arithmetic; B1 of #411)
--   (family 7: 4-bit-and-up quant — NF4/MX/FP4 — all reserved)

public export
RuntimeDType Bool where
  dtypeTag = 1

public export
RuntimeDType U8 where
  dtypeTag = 4

public export
RuntimeDType I8 where
  dtypeTag = 8

public export
RuntimeDType I16 where
  dtypeTag = 9

public export
RuntimeDType I32 where
  dtypeTag = 10

public export
RuntimeDType I64 where
  dtypeTag = 11

public export
RuntimeDType F16 where
  dtypeTag = 13

public export
RuntimeDType F32 where
  dtypeTag = 14

public export
RuntimeDType F64 where
  dtypeTag = 15

public export
RuntimeDType BF16 where
  dtypeTag = 17

-- Sub-byte quantization dtypes (#411 BitNet). The C-side dispatch
-- table only routes these through pack/unpack and (eventually)
-- BitLinear forward; ordinary create/cast paths abort if reached
-- with these tags until B3's per-backend kernels land.
public export
RuntimeDType Binary where
  dtypeTag = 24

public export
RuntimeDType Ternary where
  dtypeTag = 25

||| Construct a flat 1-D image tensor (length `flatLen` = rows * cols)
||| for image `idx` in the IDX dataset `ds`. The C side allocates and
||| memcpys a fresh `double[flatLen]` (ownership transfers — the
||| streamed creator path free()s its input buffer); this routes that
||| pointer through the generic dtype-streamed creator (`dtCreate1d`)
||| so the result honestly matches the caller's chosen `t`. The
||| intermediate 3-D shape the legacy `mnist_get_image` produced was
||| always reshape-flattened at the call site, so the 1-D shape here
||| saves an FFI hop without changing user code's downstream view.
public export
idxImage : {0 ex : Executor} -> {0 t : Type} -> Backend ex t =>
           AnyPtr -> Int -> Int -> AnyPtr
idxImage ds idx flatLen =
  dtCreate1d {ex} {t} flatLen (prim__idxImageDoubles ds idx) 0 (deviceStreamTag {ex})

----------------------------------------------------------------------
-- Path C P3-1 spike: rank-aware Tensor
----------------------------------------------------------------------
--
-- Today's `Tensor d` is shape-erased and packed into the outer
-- `Array dims (Tensor d)` via Vect-of-Vect, scalarising at every
-- op. `Tensor dims d` lifts shape onto the Tensor itself: one tensor
-- handle per typed shape, no per-element packing.
--
-- `Tensor []` is the scalar — distinguished from `Tensor [n]` etc. by
-- type. Loss naturally types as `Tensor [] ex`.
--
-- Keep `paramId`: the C-side optimizer registry is keyed on it.
-- Drop the cached `value : Double` — read at the boundary via
-- `tensorItem`.
--
-- Spike-only; lives in a parallel layer/example axis.

||| The autograd handle. Under the wrapped-handle ABI, `tensorPtr` is
||| not a raw pointer but a Chez vector `#(tensor-handle raw_ptr)`
||| produced by the creating FFI's Scheme glue and registered with the
||| `idris-tensor-guardian`. The vector IS the Tensor's runtime
||| identity — Idris-Chez codegen can't elide it without eliding the
||| Tensor value itself. C FFIs internally `vector-ref` to extract the
||| raw pointer, so this layer is invisible above the FFI boundary.
||| See docs/develop/tensor-lifecycle-plan.md.
public export
record Tensor (dims : Vect rank Nat) (0 ex : Executor) (0 dt : DType) (0 g : GradMode) where
  constructor MkTensor
  tensorPtr : AnyPtr
  paramId   : Maybe String

||| A live Tensor handle: retain/release its C-side refcount so a
||| generation-scoped free (e.g. `withNoGrad` exit) spares it.
public export
KeepAlive (Tensor dims ex dt g) where
  keepAliveRetain  t = retainHandle t.tensorPtr
  keepAliveRelease t = releaseHandle t.tensorPtr

||| Transfer a tensor to a different device. The one place where
||| device types intentionally change.
|||
||| Dispatches on `backendTag` equality:
|||
||| * Source and dest share a backend → fast intra-backend path. The
|||   backend's `primIntraMigrate` swaps the hardware variant on the
|||   *same* C handle (`tensor_to_device_<b>(handle, "mps"|"cuda:n")`
|||   on torch; a stream-tag flip on mlx; a no-op on tape). Preserves
|||   param-registry membership.
|||
|||  * Different backends → host round-trip. Source's `primToHost`
|||   copies the tensor into a CPU double buffer (F32 storage is
|||   promoted losslessly); the shape gets marshalled into a CPU int
|||   buffer; dest's `primCreateFromHost` reconstructs the tensor on
|||   the target backend with storage matching the type-level `dt`
|||   (the `RuntimeDType dt` tag is threaded through, so an F32
|||   tensor that hops backends stays F32 storage instead of
|||   silently landing in the destination's default dtype). Every
|||   step is primIO-sequenced and both temporary buffers are freed
|||   after the create. The destination tensor is a *fresh* C handle
|||   on the dest backend — registry membership does NOT follow; users
|||   transferring parameters across backends re-register on the dest
|||   side.
|||
||| The `Compatible d2 dt` constraint makes inadmissible hops
||| unrepresentable — e.g. moving an F64 tensor onto Metal
||| (`MlxExecutor MGpu` / `TorchExecutor TMps`) fails to typecheck
||| instead of aborting in the backend.
|||
||| `paramId` is preserved on the Idris-side `Tensor` record either
||| way; only the C-side registry differs.
export
toExecutor : {0 d1 : Type} -> (0 d2 : Type) ->
           UserExecutorTransfer d1 => UserExecutorTransfer d2 =>
           RuntimeDType dt         => Compatible d2 dt =>
           {rank : Nat} -> {dims : Vect rank Nat} ->
           Tensor dims d1 dt WithGrad -> IO (Tensor dims d2 dt WithGrad)
toExecutor d2 src =
  if backendTag {ex=d1} == backendTag {ex=d2}
    then pure (MkTensor
                (primIntraMigrate {ex=d2}
                  src.tensorPtr (deviceName {ex=d2}))
                src.paramId)
    else do
      let nI    = cast {to=Int} (product dims)
      let rankI = cast {to=Int} (length dims)
      -- Every FFI step goes through primIO so Idris-2's Chez codegen
      -- sequences them with %World instead of let-laziness — the
      -- same pattern as Test.Transfer.makeVec4 (see gotchas.md
      -- "Pure-typed FFI helpers reorder across sibling lets").
      dataBuf  <- primIO (\w => MkIORes (primAllocHost {ex=d1} nI) w)
      dataBuf' <- primIO (\w =>
        MkIORes (primToHost {ex=d1} src.tensorPtr dataBuf) w)
      shapeBuf <- primIO (\w => MkIORes (primAllocIntHost {ex=d2} rankI) w)
      shapeBuf' <- primIO (\w => MkIORes (writeShape shapeBuf 0 dims) w)
      destPtr <- primIO (\w =>
        MkIORes (primCreateFromHost {ex=d2} dataBuf' shapeBuf' rankI 0
                  (dtypeTag {t=dt})) w)
      -- Backend-side `tensor_create_<b>` copies the buffer into its
      -- own arena/storage, so both host buffers are dead here. An
      -- earlier version leaked them after chained hops crashed at
      -- the third hop "in unclear ways" — consistent with the
      -- pure-typed-let reorder/elision class this function used to
      -- be written in, which the primIO sequencing above excludes
      -- structurally. A C-level 3-hop probe (tape→torch→mlx→tape,
      -- optimize-level 3) with immediate frees runs clean, and so
      -- does Example.PrecisionDemo's 4-hop Part 3.
      primIO (primFreeHost {ex=d1} dataBuf')
      primIO (primFreeIntHost {ex=d2} shapeBuf')
      pure (MkTensor destPtr src.paramId)
  where
    writeShape : AnyPtr -> Int -> Vect r Nat -> AnyPtr
    writeShape buf _ []          = buf
    writeShape buf off (x :: xs) =
      let buf' = primSetIntHost {ex=d2} buf off (cast {to=Int} x)
      in writeShape buf' (off + 1) xs

-- EAFP availability gate (runtime hardware-presence half) ------------
--
-- See docs/develop/device-availability-gating.md. The compile-time
-- `Linked` gate (Executor.Core) settles "is this backend compiled in";
-- this settles the genuinely-runtime question "is this *linked* device
-- backed by real hardware right now" (e.g. cuda:1 on a 1-GPU box, MPS
-- on a non-Apple host). We answer it the easier-to-ask-forgiveness way:
-- attempt the construction; the backend's C shim wraps the alloc in
-- try/catch and returns a NULL handle on its own exception; we lift
-- NULL -> Left. One source of truth (the real allocation), no TOCTOU,
-- no separate is_available probe to drift. Backends whose construction
-- never fails (tape; mlx stream switch) simply never report Left.

||| Why a device-pinned construction failed. Carries the device's
||| human name (`deviceName {ex}`) for diagnostics; the caller decides
||| whether to skip (tests) or hard-fail with a clear message.
public export
data ExecutorError : Type where
  DeviceUnavailable : (device : String) -> ExecutorError

public export
Show ExecutorError where
  show (DeviceUnavailable d) =
    "device unavailable: \"" ++ d ++ "\" is linked but not backed by "
      ++ "usable hardware on this host"

||| Run a device-pinned construction action under EAFP semantics. If
||| the backend's shim returned a NULL handle (it caught its own
||| allocation/transfer exception), surface `Left (DeviceUnavailable
||| (deviceName {ex}))`; otherwise `Right` the tensor. This is the one
||| primitive every checked constructor builds on — it composes with
||| *any* existing `IO (Tensor ...)` producer (`tconstScalar`,
||| `tparam2d`, `toExecutor`, …) rather than duplicating each.
export
attemptOn : {0 ex : Executor} -> UserExecutorCore ex =>
            IO (Tensor dims ex dt g) -> IO (Either ExecutorError (Tensor dims ex dt g))
attemptOn act = do
  t <- act
  pure $ if prim__handleIsNull t.tensorPtr == 1
           then Left (DeviceUnavailable (deviceName {ex}))
           else Right t

||| `toExecutor` under the EAFP gate: a move to an absent destination
||| device surfaces as `Left ExecutorError` instead of aborting deep in
||| the backend. Wired to the same null-handle primitive as `attemptOn`.
||| The destination construction (`primIntraMigrate` /
||| `primCreateFromHost`) routes through the backend's guarded shim.
export
toExecutorChecked : {0 d1 : Type} -> (0 d2 : Type) ->
                  UserExecutorTransfer d1 => UserExecutorTransfer d2 =>
                  RuntimeDType dt         => Compatible d2 dt =>
                  {rank : Nat} -> {dims : Vect rank Nat} ->
                  Tensor dims d1 dt WithGrad ->
                  IO (Either ExecutorError (Tensor dims d2 dt WithGrad))
toExecutorChecked d2 src = attemptOn {ex=d2} (toExecutor d2 src)

||| A discovered device, reduced to the facts discovery + reporting
||| need: its human name, its physical `HardwareClass` (for grouping
||| devices that share silicon), and a pre-baked `probe` that attempts
||| a 1-element allocation under the EAFP gate. The concrete `(d, dt)`
||| is captured at `someExecutor` construction (where a compatible dtype
||| is known), so this descriptor is dtype-agnostic and existential-
||| free — you can't mint more tensors from it, which is exactly what
||| discovery wants (use sites name the concrete device themselves).
public export
record SomeExecutor where
  constructor MkSomeExecutor
  deviceLabel : String
  hwClass     : HardwareClass
  probe       : IO Bool

||| EAFP device discovery: keep the candidates whose probe succeeds.
||| The candidate list is caller-supplied — built-ins compose a list
||| from their `Linked`-witnessed tags, BYO backends append their own
||| `someExecutor` descriptors. The decision always comes from a real
||| allocation, never a standalone `is_available` probe.
export
availableExecutors : List SomeExecutor -> IO (List SomeExecutor)
availableExecutors []           = pure []
availableExecutors (sd :: rest) = do
  ok    <- sd.probe
  rest' <- availableExecutors rest
  pure $ if ok then sd :: rest' else rest'

||| Mark a tensor as no-grad: flips the C-side `requires_grad` flag to
||| false and retypes the handle as `NoGrad`. After this, downstream
||| ops on the tensor build no tape entries (per-backend semantics:
||| tape sets the field, torch calls `set_requires_grad_(false)`, mlx
||| sets the bool). For parameter tensors this effectively freezes
||| them — gradients no longer flow back to update their value. For
||| activation tensors it's harmless (they aren't graph leaves).
||| Mirrors PyTorch's `tensor.requires_grad_(False)`.
|||
||| Linear in its input: consumes the original tensor reference at
||| compile time, so a caller can't accidentally use the pre-weaken
||| variable afterwards (the runtime state has changed under it).
||| Closes the "freeze then keep using the original WithGrad type"
||| aliasing footgun.
export
weakenGrad : UserExecutorTraining ex => (1 _ : Tensor dims ex dt g) -> IO (Tensor dims ex dt NoGrad)
weakenGrad (MkTensor ptr pid) = do
  primIO (primSetRequiresGrad {ex} ptr 0)
  pure (MkTensor ptr pid)

||| Pure type-level cast between grad-modes. `g` is 0-quantity in
||| `Tensor`'s declaration, so `MkTensor` is polymorphic in `g` and
||| destructure-reconstruct flips the type tag with no runtime work
||| and no type-system bypass. Used by `unfreezeLayer` impls to
||| retype tensor fields after flipping the C-side `requires_grad`
||| flag. Not a control surface for users — to *change* the runtime
||| flag, use `weakenGrad`.
export
retypeGrad : Tensor dims ex dt g1 -> Tensor dims ex dt g2
retypeGrad (MkTensor ptr pid) = MkTensor ptr pid

||| Type-level aliases for common Tensor shapes. Aliases route shape
||| arithmetic (e.g. `4 * o`) through a Nat-argument slot rather than
||| inlining inside a Vect literal — the latter triggers an Idris 2
||| type-checker hang on multiplicative Nat expressions.
||| (`Tensor [4 * o, i] ex` hangs; `TMat (4 * o) i d` works.)
public export
0 TVec : Nat -> Executor -> DType -> GradMode -> Type
TVec n ex dt g = Tensor [n] ex dt g

public export
0 TMat : Nat -> Nat -> Executor -> DType -> GradMode -> Type
TMat m n ex dt g = Tensor [m, n] ex dt g

-- Smart constructors --------------------------------------------------

||| Lift a pure expression into an IO action whose body is RE-EVALUATED
||| on every sequencing (NOT memoized like `Lazy a`). The correct
||| primitive for "FFI side effect deferred until IO is run". Every
||| Tensor smart constructor below uses this — their bodies are pure
||| expressions whose evaluation triggers FFI side effects, so wrapping
||| in `ioRerun` lets IO sequencing control when those side effects fire
||| (specifically: makes `withNoGrad (do ...)` properly bracket them).
export %inline
ioRerun : (() -> a) -> IO a
ioRerun f = primIO (\w => MkIORes (f ()) w)

||| Scalar Tensor from a Double. Takes the value as a runtime argument
||| so Idris/Chez does NOT memoise the FFI result as a module-level
||| constant — same defence as `freshZeroLossT`. Non-grad: the C
||| backend creates a non-persistent scalar that is freed by the next
||| `tape_reset` (i.e. fine to call inside an epoch's loss builder).
export
||| Note: keeps the unified `prim__createScalar` (Phase 1 alias to
||| the primary backend) rather than dispatching via
||| `UserExecutorCore.primCreateScalar`. The op has no Tensor input, so
||| `d` would need to be inferred from the result's use-site and Idris
||| 2's bidirectional inference doesn't reliably push the instance
||| constraint through every call site that just lets-binds the
||| result. For built-in devices this matches the previous behavior
||| (alias to primary); for user-supplied devices, users should
||| construct scalars via their own `UserExecutorCore.primCreateScalar`
||| directly. Same compromise applies to `tparamScalar` and
||| `freshZeroLossT`.
tconstScalar : {0 ex : Executor} -> Backend ex dt => Double -> IO (Tensor [] ex dt WithGrad)
tconstScalar v = ioRerun (\_ => MkTensor (dtCreateScalar {ex} {t=dt} v 0 (deviceStreamTag {ex})) Nothing)

||| Build a `SomeExecutor` candidate for a concrete `(device, dtype)`.
||| The probe attempts a tiny scalar allocation through `attemptOn`, so
||| a linked-but-absent device (e.g. `cuda:1` on a 1-GPU box) reports
||| `False`; a backend whose construction never fails reports `True`.
export
someExecutor : {0 ex : Executor} -> {0 dt : DType} ->
             Backend ex dt => HardwareClassed ex => SomeExecutor
someExecutor =
  MkSomeExecutor (deviceName {ex}) (hardwareClass {ex})
    (do r <- attemptOn {ex} (tconstScalar {ex} {dt} 0.0)
        pure $ case r of
                 Right _ => True
                 Left _  => False)
