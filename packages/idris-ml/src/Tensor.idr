module Tensor

import Data.List
import Data.Maybe
import Data.SortedMap
import Data.Vect
import Compat.Random

import DataPoint
import Device
import public DType.Core
import public GradMode
import Floating
import Array
import Util


----------------------------------------------------------------------
-- Backend FFI (libtorch via libidrisml)
----------------------------------------------------------------------

-- ----------------------------------------------------------------------
-- Managed-handle plumbing (Chez guardian + foreign-procedure drain).
-- See docs/develop/tensor-lifecycle.md for the design.
--
-- Every Tensor-returning FFI's Scheme wrapper wraps the C return in a
-- 3-slot Chez vector and registers it with a top-level guardian:
--   (vector 'tensor-handle-v2 "TAG" raw_ptr)
-- slot 0: the literal sentinel symbol `'tensor-handle-v2` (the `-v2`
--         marks the current layout; a stale consumer reading slot 1
--         expecting a raw pointer would get a string and crash
--         obviously, rather than corrupt silently).
-- slot 1: backend tag string — one of "tape" / "torch" / "mlx" (set
--         by per-backend wraps in Device/*.idr) or "primary" (set by
--         the unsuffixed wraps in this file; they call C symbols
--         aliased at link time to whichever backend is primary, so
--         "primary" routes back through the unified
--         `tensor_release_handle` alias).
-- slot 2: the raw C tensor pointer (what foreign-procedure calls
--         actually consume).
--
-- The vector IS the Tensor's runtime identity in Chez — the
-- Idris-Chez compiler can't elide it without eliding the value
-- itself. When the wrap becomes GC-unreachable,
-- prim__drainManagedHandlesC pops it, reads the tag at slot 1, and
-- calls `tensor_release_handle_<tag>` on the raw pointer at slot 2.
-- Per-backend dispatch is what makes multi-backend builds correct:
-- before v2 the drain always called the link-time-aliased unified
-- symbol (typically the primary's no-op), so mlx-allocated tensors
-- leaked their refcount and triggered SIGSEGV during exit-time mlx
-- static destructor teardown.
--
-- Idris does not distinguish "raw AnyPtr" from "wrapped AnyPtr" at the
-- type level — both are AnyPtr. Every %foreign "scheme:..." wrapper
-- internally `(vector-ref a<i> 2)`s its Tensor args to extract the
-- raw pointer before calling the C function. The wrap layer is
-- invisible to Idris.
-- ----------------------------------------------------------------------

-- Self-init: creates the guardian and ensures libidrisml is loaded with
-- RTLD_GLOBAL so subsequent `foreign-procedure` lookups for C symbols
-- (tensor_retain_handle, tensor_release_handle, mlx_set_gc_drain_callback)
-- succeed. The %foreign "C:..." declarations elsewhere in this file
-- also trigger libidrisml load on first call, but the wrapped-handle ABI
-- shifts many FFIs to %foreign "scheme:..." with embedded
-- foreign-procedure calls; loading the lib here removes the ordering
-- dependency on a stray %foreign "C:..." call firing first.
%foreign "scheme:(lambda (dummy) (if (top-level-bound? 'idris-tensor-guardian) 0 (begin (load-shared-object \"libidrisml.dylib\") (set-top-level-value! 'idris-tensor-guardian (make-guardian)) 1)))"
prim__initGuardianC : Int -> PrimIO Int

-- Install `idris-drain-once`: a top-level Scheme procedure that pops
-- one dead wrap from the guardian (or returns #f if none), reads the
-- backend tag at slot 1 and the raw pointer at slot 2, and calls the
-- matching `tensor_release_handle_<tag>` (or unified
-- `tensor_release_handle` for "primary"). Caches the foreign-procedure
-- per tag in `idris-release-cache` so tight eval loops don't re-resolve.
-- Idempotent — running again replaces the same top-level binding.
%foreign "scheme:(lambda (dummy) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t)))))) 1)"
prim__installDrainHelperC : Int -> PrimIO Int

-- Drain the guardian: repeatedly pop + release until empty. Returns the
-- count drained. Self-initializing — yields 0 if the guardian doesn't
-- exist yet (no managed-handle wraps have happened).
%foreign "scheme:(lambda (dummy) (if (not (top-level-bound? 'idris-drain-once)) 0 (let loop ((n 0)) (if ((top-level-value 'idris-drain-once)) (loop (+ n 1)) n))))"
prim__drainManagedHandlesC : Int -> PrimIO Int

-- Force a Chez major GC. Use sparingly — only at known-safe drain points.
%foreign "scheme:(lambda (dummy) (collect 4) 0)"
prim__forceMajorGcC : Int -> PrimIO Int

||| Initialize the managed-handle guardian. Idempotent. Returns 1 the
||| first time it runs, 0 thereafter. Call once at backend init / first
||| tensor creation — wrapHandle assumes the guardian exists.
export
initManagedHandles : IO Int
initManagedHandles = do
  r <- primIO (prim__initGuardianC 0)
  _ <- primIO (prim__installDrainHelperC 0)
  pure r

||| Drain the guardian. Pops dead wrappers and calls
||| tensor_release_handle on each. Returns the number drained.
export
drainManagedHandles : IO Int
drainManagedHandles = primIO (prim__drainManagedHandlesC 0)

||| Force a Chez major GC. Combined with drainManagedHandles, this is the
||| reclamation mechanism for eval-phase tight loops. Expensive — only
||| call at boundaries like no_grad_end or every Nth FFI in heavy code.
export
forceMajorGc : IO ()
forceMajorGc = do
  _ <- primIO (prim__forceMajorGcC 0)
  pure ()

-- Retain / release a managed handle by its wrap (tag-dispatched, mirrors
-- the drain). The `KeepAlive` interface uses these to rescue tensors that
-- escape a generation-scoped free: retain (rc++) before the no_grad_end
-- sweep, release (rc--) after, so the sweep's "free block-local rc==1"
-- step spares them. Tape/torch retains are no-op stubs, so this is free
-- on those backends. The wrap layout (`(vector 'tensor-handle-v2 tag raw)`)
-- is uniform across backends.
%foreign "scheme:(lambda (wr) (let ((tag (vector-ref wr 1)) (raw (vector-ref wr 2))) (let ((sym (if (string=? tag \"primary\") \"tensor_retain_handle\" (string-append \"tensor_retain_handle_\" tag)))) ((foreign-procedure sym (void*) void) raw))) 0)"
prim__retainHandleC : AnyPtr -> PrimIO Int

%foreign "scheme:(lambda (wr) (let ((tag (vector-ref wr 1)) (raw (vector-ref wr 2))) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) ((foreign-procedure sym (void*) void) raw))) 0)"
prim__releaseHandleC : AnyPtr -> PrimIO Int

-- EAFP availability gate: a device-pinned construction whose backend
-- shim caught its own allocation/transfer exception returns a NULL raw
-- handle (slot 2 of the wrap-v2 vector is the fixnum 0). This predicate
-- reads slot 2 and reports null as 1, live as 0. The arg is threaded
-- through so the Idris-Chez compiler can't CSE the call to a constant.
%foreign "scheme:(lambda (wr) (if (eqv? (vector-ref wr 2) 0) 1 0))"
prim__handleIsNull : AnyPtr -> Int

||| Bump the C-side refcount of a managed handle (by its wrap).
export
retainHandle : AnyPtr -> IO ()
retainHandle wr = ignore $ primIO (prim__retainHandleC wr)

||| Drop the C-side refcount of a managed handle (by its wrap).
export
releaseHandle : AnyPtr -> IO ()
releaseHandle wr = ignore $ primIO (prim__releaseHandleC wr)

-- Lifecycle
--
-- Wrapped-handle ABI (mlx): every Tensor-returning FFI's Scheme wrapper
-- wraps the C return in a Chez vector + registers it with the
-- idris-tensor-guardian + retains via tensor_retain_handle. Every
-- Tensor-consuming FFI's Scheme wrapper extracts the raw pointer via
-- vector-ref. The Idris-level value (AnyPtr) is the wrap; the wrap is
-- the Tensor's identity in the Chez runtime — the Idris-Chez compiler
-- can't elide it without eliding the value itself.
--
-- See docs/develop/tensor-lifecycle-plan.md.


-- (Phase 7: `prim__toDevice` and `prim__tensorDevice` were unified-
-- name FFI bindings used by the old `toDevice`. They're now unused
-- — `toDevice` lives on top of `UserDeviceTransfer`'s per-backend
-- methods. The unified C symbols still exist (renamed at link time
-- to `_<primary>` suffixes via the alias machinery) but no Idris
-- code consumes them.)

-- Arithmetic (all return new tensors — libtorch builds autograd graph)


-- Linear algebra

-- Fused 1D linear: y = W @ x + bias. Eliminates the per-call FFI
-- overhead of separate prim__mv + prim__add.


-- Activation


-- Loss

-- Reduction


-- Array creation/accessors


-- NTM


-- Cross-attention: Q @ K^T * scale [+ mask] -> softmax -> @ V


-- Embedding

-- Batch Norm

-- Dropout

-- Shape / info queries


-- Gather / Scatter


-- Sort / Scan


-- Average Pooling


-- Conv1D / MaxPool1D


-- Conv2D / MaxPool2D


-- MNIST data loading
%foreign "C:mnist_load,libidrisml"
export
prim__mnistLoad : String -> String -> AnyPtr

%foreign "C:mnist_count,libidrisml"
export
prim__mnistCount : AnyPtr -> Int


%foreign "C:mnist_get_label,libidrisml"
export
prim__mnistGetLabel : AnyPtr -> Int -> Int

-- Parameter registry: `primParamRegister {d}` (UserDeviceTape) is
-- the sole entry. Each backend's instance wraps the returned handle
-- with its OWN tag ("tape"/"torch"/"mlx") and retains via the
-- suffixed `tensor_retain_handle_<b>`. The former unified-name
-- `prim__paramRegister` here hardcoded the wrap tag to "primary" and
-- bound `param_register_return` / `tensor_retain_handle` unaliased —
-- a latent bug for non-primary backends in a multi-link build.

-- In-place scalar subtract on a tensor (under no_grad). Returns tensor for threading.

-- Array-level parameter creation


-- State tensors (non-learnable, non-grad). Both init-time permanent state
-- (NTM mask, BatchNorm running stats, transformer PE, DNC mask) AND
-- per-sequence transient state (Ntm/Dnc zeroState) flow through this
-- single path. mlx: is_state=1, refcount-driven; the Idris-side wrap is
-- the only stable holder, so the Tensor lives as long as the holder does.
-- tape/torch: the backend's own arena/shared_ptr handles freeing.


-- Fused LSTM gates: takes combined [4*o] tensor + prev_cell [o], returns pair handle


||| Tensors reachable from a value that must survive a generation-scoped
||| free — specifically the *result* of a `withNoGrad` block. At the
||| block's `primNoGradEnd`, every wrap-only (rc==1) handle created inside
||| the block is deleted (that's what bounds the live-handle count). The
||| result, if it holds tensors created inside the block, would be caught
||| by that sweep; `keepAliveRetain` bumps their refcount first so they're
||| spared, and `keepAliveRelease` drops it again afterwards.
|||
||| Scalars / strings are no-ops (zero overhead on the common scalar-
||| returning eval loops). Containers recurse. Provide an instance for any
||| custom type returned from `withNoGrad` that carries live tensors
||| (e.g. an RL rollout buffer); otherwise its tensors get freed and a
||| later use dangles.
public export
interface KeepAlive a where
  keepAliveRetain  : a -> IO ()
  keepAliveRelease : a -> IO ()

public export
KeepAlive () where
  keepAliveRetain _ = pure (); keepAliveRelease _ = pure ()
public export
KeepAlive Double where
  keepAliveRetain _ = pure (); keepAliveRelease _ = pure ()
public export
KeepAlive Int where
  keepAliveRetain _ = pure (); keepAliveRelease _ = pure ()
public export
KeepAlive Integer where
  keepAliveRetain _ = pure (); keepAliveRelease _ = pure ()
public export
KeepAlive Nat where
  keepAliveRetain _ = pure (); keepAliveRelease _ = pure ()
public export
KeepAlive Bool where
  keepAliveRetain _ = pure (); keepAliveRelease _ = pure ()
public export
KeepAlive String where
  keepAliveRetain _ = pure (); keepAliveRelease _ = pure ()
public export
(KeepAlive a, KeepAlive b) => KeepAlive (a, b) where
  keepAliveRetain (x, y)  = do keepAliveRetain x; keepAliveRetain y
  keepAliveRelease (x, y) = do keepAliveRelease x; keepAliveRelease y
public export
KeepAlive a => KeepAlive (List a) where
  keepAliveRetain  = traverse_ keepAliveRetain
  keepAliveRelease = traverse_ keepAliveRelease
public export
KeepAlive a => KeepAlive (Maybe a) where
  keepAliveRetain  = maybe (pure ()) keepAliveRetain
  keepAliveRelease = maybe (pure ()) keepAliveRelease
public export
{n : Nat} -> KeepAlive a => KeepAlive (Vect n a) where
  keepAliveRetain  = traverse_ keepAliveRetain
  keepAliveRelease = traverse_ keepAliveRelease

||| Run an `IO` action with autograd disabled. Inside the action,
||| every tensor op skips tape/autograd graph construction, so the
||| results have no path to backward. Standard for RL rollouts and
||| any inference pass. Mirrors PyTorch's `with torch.no_grad():`.
||| Nested calls are stacked: only the outermost begin/end pair
||| toggles tracking, so library code can call this without checking
||| whether the caller already disabled grad.
|||
||| On exit the backend deletes every wrap-only handle created inside
||| the block (generation-scoped free), which is what keeps the live
||| handle / Metal-buffer count bounded across long eval loops.
|||
||| **Contract:** the block's *result* must hold no live tensors created
||| inside the block — extract them to scalars/host data (`tensorItem`,
||| `tvecToVector`) before returning, as every eval/rollout here does. If
||| you must return a live `Tensor` (or a struct containing one), use
||| `withNoGradKeep`, which rescues result tensors from the sweep via
||| `KeepAlive`. Returning a live tensor from plain `withNoGrad` will
||| free it at the bracket exit and dangle on next use.
|||
||| The no-grad scope is per-backend (tape/mlx push a counter,
||| torch arms a `NoGradGuard`), so it dispatches via
||| `primNoGradBegin`/`primNoGradEnd` from the in-scope
||| `UserDeviceTape d`. `d` doesn't appear in the action type, so
||| callers pin it explicitly (`withNoGrad {d=ExampleDevice} ...`).
export
withNoGrad : {0 d : Device} -> UserDeviceTape d => IO a -> IO a
withNoGrad act = do
  primIO (primNoGradBegin {d})
  result <- act
  forceMajorGc
  _ <- drainManagedHandles
  primIO (primNoGradEnd {d})
  pure result

||| `withNoGrad` for blocks whose result carries live tensors created
||| inside the block (e.g. a cached embedding, an inference output kept
||| for later use). `keepAliveRetain` bumps their refcount so the block-
||| exit generation-scoped free spares them, then releases afterwards.
||| The common scalar-returning eval/rollout loops use plain `withNoGrad`.
export
withNoGradKeep : {0 d : Device} -> UserDeviceTape d => KeepAlive a => IO a -> IO a
withNoGradKeep act = do
  primIO (primNoGradBegin {d})
  result <- act
  keepAliveRetain result
  forceMajorGc
  _ <- drainManagedHandles
  primIO (primNoGradEnd {d})
  keepAliveRelease result
  pure result

||| Run a *grad-mode* IO action inside a generation bracket, freeing the
||| wrap-only tensors it created on exit. Unlike `withNoGrad` this keeps
||| autograd ON — for heavy training inner loops (a DQN replay step, a PPO
||| rollout step) whose per-step grad intermediates would otherwise pile up
||| within a single epoch past the mlx buffer ceiling. The per-epoch bracket
||| in `runTrainingIO` is the outer frame; this nests inside it. The free is
||| create_id-based (no GC needed); registry params (rc>1) and result
||| tensors (retained via `KeepAlive`) are spared. Most callers pass a
||| scalar/`()` result, so `KeepAlive` is a no-op.
export
withGenFree : {0 d : Device} -> UserDeviceTape d => KeepAlive a => IO a -> IO a
withGenFree act = do
  primIO (primEpochBegin {d})
  result <- act
  keepAliveRetain result
  primIO (primEpochEnd {d})
  keepAliveRelease result
  pure result

----------------------------------------------------------------------
-- Sequencing helper
----------------------------------------------------------------------

-- Force evaluation of first arg, return second.
-- Must use concrete AnyPtr types (not polymorphic) to avoid
-- argument count issues at the FFI boundary.


----------------------------------------------------------------------
-- C-side allocation + bulk-load helpers
----------------------------------------------------------------------

%foreign "C:tensor_alloc_doubles,libidrisml"
export prim__allocDoubles : Int -> AnyPtr


-- Wrapper that returns the buffer pointer for threading through let chains
%foreign "C:tensor_write_double_return,libidrisml"
export
prim__setDouble : AnyPtr -> Int -> Double -> AnyPtr


-- 3D batched attention ops


||| Tile a 2D tensor: `[m, n] -> [m*rep0, n*rep1]`. Element `(i, j)` in the
||| output equals element `(i mod m, j mod n)` in the input.


-- Array pointer array: stack scalar Tensor tensorPtrs to create
-- a 1D/2D tensor that preserves the autograd graph.

-- Returns the array for threading


-- N-ary cat: caller retains ownership of the handle array.
-- See tensor_cat in backend.h.

-- Batch [...] tensors into [count, ...]. Equivalent to stack at dim=0.


%foreign "C:tensor_alloc_ints,libidrisml"
export
prim__allocInts : Int -> AnyPtr

%foreign "C:tensor_write_int_return,libidrisml"
export
prim__setInt : AnyPtr -> Int -> Int -> AnyPtr


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


-- tensor_create_1d


-- tensor_create_2d


-- tensor_create_param_1d


-- tensor_create_param_2d


-- tensor_create_param_3d


-- tensor_create_param_4d


-- tensor_create_state_1d


-- tensor_create_state_2d


-- Per-dtype cast primitives. Backend support mirrors the create
-- primitives: mlx/torch implement both; tape implements _f64 (no-op
-- alias today, since the only valid source dtype is F64) and aborts
-- on _f32. Source dtype is read from the handle on the C side.


-- RuntimeDType instances — just the runtime dtype tag (0=f32,
-- 1=f64). The `dtCreate*` free functions below pass it to the
-- device's `primCreate*Streamed` method, which branches to pick the
-- `_f32_streamed_<b>` / `_f64_streamed_<b>` C symbol.

public export
RuntimeDType F32 where
  dtypeTag = 0

public export
RuntimeDType F64 where
  dtypeTag = 1

-- dtCreate* free functions — device × dtype create dispatch.
-- `d` selects the backend (via the `primCreate*Streamed` method),
-- `t` selects the dtype (via `dtypeTag`). Both implicits are pinned
-- at the call site: `{d}` from the enclosing device context, `{t=dt}`
-- by the caller. Signatures match the former RuntimeDType methods
-- (trailing `streamTag : Int`) so existing call sites only gain `{d}`.

public export
dtCreateScalar : {0 d : Device} -> UserDeviceTape d => {0 t : Type} -> RuntimeDType t =>
                 Linked d => Compatible d t =>
                 Double -> Int -> Int -> AnyPtr
dtCreateScalar v rg stream = primCreateScalarStreamed {d} v rg stream (dtypeTag {t})

public export
dtCreate : {0 d : Device} -> UserDeviceTape d => {0 t : Type} -> RuntimeDType t =>
           Linked d => Compatible d t =>
           AnyPtr -> AnyPtr -> Int -> Int -> Int -> AnyPtr
dtCreate dat sh r rg stream = primCreateStreamed {d} dat sh r rg stream (dtypeTag {t})

public export
dtCreate1d : {0 d : Device} -> UserDeviceTape d => {0 t : Type} -> RuntimeDType t =>
             Linked d => Compatible d t =>
             Int -> AnyPtr -> Int -> Int -> AnyPtr
dtCreate1d n dat rg stream = primCreate1dStreamed {d} n dat rg stream (dtypeTag {t})

public export
dtCreate2d : {0 d : Device} -> UserDeviceTape d => {0 t : Type} -> RuntimeDType t =>
             Linked d => Compatible d t =>
             Int -> Int -> AnyPtr -> Int -> Int -> AnyPtr
dtCreate2d r c dat rg stream = primCreate2dStreamed {d} r c dat rg stream (dtypeTag {t})

public export
dtCreateParam1d : {0 d : Device} -> UserDeviceTape d => {0 t : Type} -> RuntimeDType t =>
                  Linked d => Compatible d t =>
                  Int -> AnyPtr -> Int -> AnyPtr
dtCreateParam1d n dat stream = primCreateParam1dStreamed {d} n dat stream (dtypeTag {t})

public export
dtCreateParam2d : {0 d : Device} -> UserDeviceTape d => {0 t : Type} -> RuntimeDType t =>
                  Linked d => Compatible d t =>
                  Int -> Int -> AnyPtr -> Int -> AnyPtr
dtCreateParam2d r c dat stream = primCreateParam2dStreamed {d} r c dat stream (dtypeTag {t})

public export
dtCreateParam3d : {0 d : Device} -> UserDeviceTape d => {0 t : Type} -> RuntimeDType t =>
                  Linked d => Compatible d t =>
                  Int -> Int -> Int -> AnyPtr -> Int -> AnyPtr
dtCreateParam3d a b c dat stream = primCreateParam3dStreamed {d} a b c dat stream (dtypeTag {t})

public export
dtCreateParam4d : {0 d : Device} -> UserDeviceTape d => {0 t : Type} -> RuntimeDType t =>
                  Linked d => Compatible d t =>
                  Int -> Int -> Int -> Int -> AnyPtr -> Int -> AnyPtr
dtCreateParam4d a b c e dat stream = primCreateParam4dStreamed {d} a b c e dat stream (dtypeTag {t})

public export
dtCreateState1d : {0 d : Device} -> UserDeviceTape d => {0 t : Type} -> RuntimeDType t =>
                  Linked d => Compatible d t =>
                  Int -> AnyPtr -> Int -> AnyPtr
dtCreateState1d n dat stream = primCreateState1dStreamed {d} n dat stream (dtypeTag {t})

public export
dtCreateState2d : {0 d : Device} -> UserDeviceTape d => {0 t : Type} -> RuntimeDType t =>
                  Linked d => Compatible d t =>
                  Int -> Int -> AnyPtr -> Int -> AnyPtr
dtCreateState2d r c dat stream = primCreateState2dStreamed {d} r c dat stream (dtypeTag {t})

public export
dtCastFrom : {0 d : Device} -> UserDeviceTape d => {0 t : Type} -> RuntimeDType t =>
             Linked d => Compatible d t =>
             AnyPtr -> Int -> AnyPtr
dtCastFrom tns stream = primCastStreamed {d} tns stream (dtypeTag {t})


----------------------------------------------------------------------
-- Backpropagation: prims for native optimizer
----------------------------------------------------------------------


----------------------------------------------------------------------
-- Native Optimizer
----------------------------------------------------------------------

||| Polyak soft update for twin-network param groups registered under
||| `onlineScope` vs `targetScope`: for each online param, finds the
||| matching target param (same suffix after scope prefix) and blends
|||   target_data ← (1 − tau) · target_data + tau · online_data
||| in-place. Returns the number of param pairs blended. Used by SAC to
||| track target-Q networks.
|||
||| Per-backend: the registry storing the params lives in the backend
||| TU, so dispatch via `primPolyakBlend` from the in-scope
||| `UserDeviceTape d` instance.
export
polyakUpdate : UserDeviceTape d =>
               (tau : Double) -> (onlineScope : String) -> (targetScope : String) -> IO Int
polyakUpdate tau onlineScope targetScope =
  primIO (primPolyakBlend {d} tau onlineScope targetScope)


public export
data ClipMode = NoClip | ValueClip Double | NormClip Double

||| Native optimizer handle. Single step() call updates all
||| parameters in the backend's registry. The `d` phantom pins the
||| optimizer to the backend whose registry it manages — a
||| `NativeOptimizer d` can only step a loss `Tensor [] d dt`.
public export
record NativeOptimizer (0 d : Device) where
  constructor MkNativeOptimizer
  handle : AnyPtr
  clipMode : ClipMode

||| Create a native SGD optimizer.
export
nativeSgd : UserDeviceTape d => Double -> NativeOptimizer d
nativeSgd lr = MkNativeOptimizer (primOptimizerCreateSgd {d} lr) NoClip

||| Create a native RMSprop optimizer (matches PyTorch defaults).
export
nativeRmsprop : UserDeviceTape d =>
                (lr : Double) -> (alpha : Double) -> (eps : Double) ->
                (clipVal : Double) -> (momentum : Double) -> NativeOptimizer d
nativeRmsprop lr alpha eps clipVal momentum =
  MkNativeOptimizer
    (primOptimizerCreateRmsprop {d} lr alpha eps 0.0 momentum)
    (ValueClip clipVal)

||| Create a native Adam optimizer with global norm clipping.
export
nativeAdamGlobalClip : UserDeviceTape d =>
                       (lr : Double) -> (beta1 : Double) -> (beta2 : Double) ->
                       (eps : Double) -> (maxNorm : Double) -> NativeOptimizer d
nativeAdamGlobalClip lr beta1 beta2 eps maxNorm =
  MkNativeOptimizer
    (primOptimizerCreateAdam {d} lr beta1 beta2 eps)
    (NormClip maxNorm)

||| Create a native Adam optimizer that only manages params whose registry
||| paramId starts with `scope`. Empty scope behaves like
||| `nativeAdamGlobalClip`. Used for multi-network setups where each
||| network (e.g. SAC actor / q1 / q2) needs its own optimizer so that
||| gradient leakage from one network's loss doesn't update another
||| network's weights (matches PyTorch's one-optimizer-per-net pattern).
export
nativeAdamGroup : UserDeviceTape d =>
                  (scope : String) ->
                  (lr : Double) -> (beta1 : Double) -> (beta2 : Double) ->
                  (eps : Double) -> (maxNorm : Double) -> NativeOptimizer d
nativeAdamGroup scope lr beta1 beta2 eps maxNorm =
  MkNativeOptimizer
    (primOptimizerCreateAdamGroup {d} lr beta1 beta2 eps scope)
    (NormClip maxNorm)

||| Create a native AdamW optimizer (decoupled weight decay) with global norm clipping.
export
nativeAdamW : UserDeviceTape d =>
              (lr : Double) -> (beta1 : Double) -> (beta2 : Double) ->
              (eps : Double) -> (weightDecay : Double) -> (maxNorm : Double) -> NativeOptimizer d
nativeAdamW lr beta1 beta2 eps wd maxNorm =
  MkNativeOptimizer
    (primOptimizerCreateAdamW {d} lr beta1 beta2 eps wd)
    (NormClip maxNorm)

||| Set a per-parameter learning rate override. Parameters matching the given
||| name will use this LR instead of the optimizer's base LR.
||| Use LR=0 to freeze a parameter. Set LR<0 to revert to base LR.
export
setParamLR : UserDeviceTape d => NativeOptimizer d -> String -> Double -> IO ()
setParamLR opt name lr = primIO (primOptimizerSetParamLr {d} opt.handle name lr)

||| Update the optimizer's base (global) learning rate. Per-parameter
||| overrides set via `setParamLR` remain in effect; only un-overridden
||| params pick up the new base LR. Used to apply LR schedules per epoch.
export
setLearningRate : UserDeviceTape d => NativeOptimizer d -> Double -> IO ()
setLearningRate opt lr = primIO (primOptimizerSetLr {d} opt.handle lr)

-- Fused native train step: zero_grad → backward → clip → step.
-- Fused: zero_grad → backward → clip → step in single C call.
-- Returns loss value (read before step, so not stale).
--
-- After the C call returns, force a Chez minor GC + drain the
-- managed-handle guardian. This is the training-loop drain trigger
-- that lets the mlx refcount-driven lifecycle reclaim per-step
-- intermediate Tensors — without it, the wrap-and-retain on each
-- Tensor's creation keeps its refcount at >=1 indefinitely (Chez
-- doesn't auto-GC under foreign-side pressure alone, and drain is
-- only otherwise called at withNoGrad exit). On tape/torch the drain
-- is essentially a no-op (their retain/release are stubs).
-- After the step, force a Chez major GC then drain all dead wraps via
-- the per-backend dispatch helper `idris-drain-once` (installed by
-- prim__installDrainHelperC). This is the reclamation pump for hot
-- training loops where ops bypass `tape_append` and per-op refcount
-- bookkeeping doesn't fire — without it, the wrap-and-retain on each
-- new Tensor keeps refcounts at >=1 indefinitely.
-- The fused step itself dispatches per-backend via
-- `primNativeTrainStep {d}` (see `UserDeviceTape`); each backend's
-- Scheme wrap carries the same GC + drain epilogue.

----------------------------------------------------------------------
-- GC / RSS
----------------------------------------------------------------------

export
forceGC : IO ()
forceGC = pure ()

%foreign "C:get_rss_mb,libidrisml"
prim__getRssMB : Int

%foreign "C:get_current_rss_mb,libidrisml"
prim__getCurrentRssMB : Int

export
getRssMB : Nat -> Int
getRssMB _ = prim__getRssMB

export
getCurrentRssMB : Nat -> Int
getCurrentRssMB _ = prim__getCurrentRssMB


||| Bulk-convert a Vector of Doubles to a C tensor handle.
||| The C tensor_create_1d function frees the input buffer after copying.
export
bulkToTensor : {0 d : Device} -> UserDeviceTape d => RuntimeDType dt => Linked d => Compatible d dt => {n : Nat} -> Vector n Double -> AnyPtr
bulkToTensor {n} (VArray elems) =
  let nI = cast {to=Int} n
      buf = prim__allocDoubles nI
      buf' = packDoubleBuf buf 0 elems
  in dtCreate1d {d} {t=dt} nI buf' 0 (deviceStreamTag {d})
  where
    packDoubleBuf : AnyPtr -> Int -> Vect k (Scalar Double) -> AnyPtr
    packDoubleBuf buf _ [] = buf
    packDoubleBuf buf off (SArray v :: rest) =
      let buf' = prim__setDouble buf off v
      in packDoubleBuf buf' (off + 1) rest

||| Bulk-convert a Vect of Vectors of Doubles to a [b, i] C tensor handle.
||| The C tensor_create_2d function frees the input buffer after copying.
||| Use to stack a per-sample input batch into a single batched tensor.
export
bulkToTensor2d : {0 d : Device} -> UserDeviceTape d => RuntimeDType dt => Linked d => Compatible d dt => {b, i : Nat} -> Vect b (Vector i Double) -> AnyPtr
bulkToTensor2d {b} {i} rows =
  let bI = cast {to=Int} b
      iI = cast {to=Int} i
      buf = prim__allocDoubles (bI * iI)
      buf' = packRows buf 0 rows
  in dtCreate2d {d} {t=dt} bI iI buf' 0 (deviceStreamTag {d})
  where
    packRow : AnyPtr -> Int -> Vect k (Scalar Double) -> AnyPtr
    packRow buf _ [] = buf
    packRow buf off (SArray v :: rest) =
      let buf' = prim__setDouble buf off v
      in packRow buf' (off + 1) rest
    packRows : AnyPtr -> Int -> Vect k (Vector i Double) -> AnyPtr
    packRows buf _ [] = buf
    packRows buf off (VArray row :: rest) =
      let buf' = packRow buf off row
      in packRows buf' (off + cast {to=Int} i) rest

||| Bulk-convert a Vector of Doubles to a persistent C tensor handle.
||| Persistent tensors survive tape resets — use when data is created once
||| and reused across training epochs.
export
vectorToTensorPersistent : {0 d : Device} -> UserDeviceTape d => RuntimeDType dt => Linked d => Compatible d dt => {n : Nat} -> Vector n Double -> AnyPtr
vectorToTensorPersistent {n} (VArray elems) =
  let nI = cast {to=Int} n
      buf = prim__allocDoubles nI
      buf' = packBuf buf 0 elems
  in dtCreateState1d {d} {t=dt} nI buf' (deviceStreamTag {d})
  where
    packBuf : AnyPtr -> Int -> Vect k (Scalar Double) -> AnyPtr
    packBuf buf _ [] = buf
    packBuf buf off (SArray v :: rest) = packBuf (prim__setDouble buf off v) (off + 1) rest

||| Convert a DataPoint with Doubles to a TensorDataPoint with persistent C tensors.
export
toTDP : {0 d : Device} -> UserDeviceTape d => RuntimeDType dt => Linked d => Compatible d dt => {i, o : Nat} -> DataPoint i o Double -> TensorDataPoint i o
toTDP dp = MkTensorDataPoint (vectorToTensorPersistent {d} {dt} (x dp)) (vectorToTensorPersistent {d} {dt} (y dp))


-- runBackward is defined post-Tensor record below; the type-level
-- gate (Tensor [] d dt WithGrad-> IO ()) lives there.

-- Registry queries dispatch through the in-scope `UserDeviceTape d`:
-- each backend's registry is a separate TU-local table, so `{d}`
-- selects which one is read.

||| Get parameter count (for gradient inspection).
export
getParamCount : UserDeviceTape d => IO Int
getParamCount = primIO (primParamCount {d})

||| Get parameter name by index.
export
getParamName : UserDeviceTape d => Int -> IO String
getParamName i = primIO (primParamName {d} i)

||| Get gradient element for param i, element j.
export
getParamGradAt : UserDeviceTape d => Int -> Int -> IO Double
getParamGradAt i j = primIO (primParamGradItemAt {d} i j)

||| Zero all parameter gradients.
export
zeroAllGrads : UserDeviceTape d => IO ()
zeroAllGrads = primIO (primParamZeroAll {d})

||| Get the name of the active backend ("tape", "mlx", "torch").
||| This is the backend *family*, not the hardware variant — exactly
||| `backendTag {d}` from `UserDeviceTransfer`. (The old C
||| `backend_name` returned the same string; routing through the
||| instance drops the unified-name FFI.)
export
backendName : UserDeviceTransfer d => String
backendName = backendTag {d}

||| Reset profiling counters for backend `d`.
export
profileReset : UserDeviceTape d => IO ()
profileReset = primIO (primProfileReset {d})

||| Print backend `d`'s profile breakdown to stderr.
export
profileReport : UserDeviceTape d => IO ()
profileReport = primIO (primProfileReport {d})

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
-- type. Loss naturally types as `Tensor [] d`.
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
record Tensor (dims : Vect rank Nat) (0 d : Device) (0 dt : DType) (0 g : GradMode) where
  constructor MkTensor
  tensorPtr : AnyPtr
  paramId   : Maybe String

||| A live Tensor handle: retain/release its C-side refcount so a
||| generation-scoped free (e.g. `withNoGrad` exit) spares it.
public export
KeepAlive (Tensor dims d dt g) where
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
|||   copies the tensor into a CPU double buffer; the shape gets
|||   marshalled into a CPU int buffer; dest's `primCreateFromHost`
|||   reconstructs the tensor on the target backend. Both temporary
|||   buffers are explicitly freed via `ioRerun` so Idris-Chez can't
|||   elide the cleanup. The destination tensor is a *fresh* C handle
|||   on the dest backend — registry membership does NOT follow; users
|||   transferring parameters across backends re-register on the dest
|||   side.
|||
||| `paramId` is preserved on the Idris-side `Tensor` record either
||| way; only the C-side registry differs.
export
toDevice : {0 d1 : Type} -> (0 d2 : Type) ->
           UserDeviceTransfer d1 => UserDeviceTransfer d2 =>
           {rank : Nat} -> {dims : Vect rank Nat} ->
           Tensor dims d1 dt WithGrad -> IO (Tensor dims d2 dt WithGrad)
toDevice d2 src =
  if backendTag {d = d1} == backendTag {d = d2}
    then pure (MkTensor
                (primIntraMigrate {d = d2}
                  src.tensorPtr (deviceName {d = d2}))
                src.paramId)
    else do
      let nI       = cast {to=Int} (product dims)
      let dataBuf  = primAllocHost {d = d1} nI
      let dataBuf' = primToHost {d = d1} src.tensorPtr dataBuf
      let rankI    = cast {to=Int} (length dims)
      let shapeBuf = primAllocIntHost {d = d2} rankI
      let shapeBuf' = writeShape shapeBuf 0 dims
      -- Force primCreateFromHost via primIO so the FFI call fires
      -- here, not deferred until destPtr is consumed.
      destPtr <- primIO (\w =>
        MkIORes (primCreateFromHost {d = d2} dataBuf' shapeBuf' rankI 0) w)
      -- N.B. The host buffers (`dataBuf'`, `shapeBuf'`) leak. We
      -- previously freed them here, but in chained cross-backend
      -- `toDevice` calls (TapeDev → TorchDev → MlxDev → TapeDev)
      -- the per-step `tensor_free_doubles_<b>` of the buffer was
      -- racing the next step's reads in unclear ways and crashing
      -- at the third hop. Backend-side `tensor_create_<b>` does
      -- copy the buffer into its own arena/storage, so the buffers
      -- become garbage immediately after primCreateFromHost
      -- returns — but explicitly freeing them broke something we
      -- haven't fully diagnosed. Leak is small (numel doubles +
      -- rank ints per toDevice call); revisit when training-time
      -- transfer becomes hot.
      pure (MkTensor destPtr src.paramId)
  where
    writeShape : AnyPtr -> Int -> Vect r Nat -> AnyPtr
    writeShape buf _ [] = buf
    writeShape buf off (x :: xs) =
      let buf' = primSetIntHost {d = d2} buf off (cast {to=Int} x)
      in writeShape buf' (off + 1) xs

-- EAFP availability gate (runtime hardware-presence half) ------------
--
-- See docs/develop/device-availability-gating.md. The compile-time
-- `Linked` gate (Device.Core) settles "is this backend compiled in";
-- this settles the genuinely-runtime question "is this *linked* device
-- backed by real hardware right now" (e.g. cuda:1 on a 1-GPU box, MPS
-- on a non-Apple host). We answer it the easier-to-ask-forgiveness way:
-- attempt the construction; the backend's C shim wraps the alloc in
-- try/catch and returns a NULL handle on its own exception; we lift
-- NULL -> Left. One source of truth (the real allocation), no TOCTOU,
-- no separate is_available probe to drift. Backends whose construction
-- never fails (tape; mlx stream switch) simply never report Left.

||| Why a device-pinned construction failed. Carries the device's
||| human name (`deviceName {d}`) for diagnostics; the caller decides
||| whether to skip (tests) or hard-fail with a clear message.
public export
data DeviceError : Type where
  DeviceUnavailable : (device : String) -> DeviceError

public export
Show DeviceError where
  show (DeviceUnavailable d) =
    "device unavailable: \"" ++ d ++ "\" is linked but not backed by "
      ++ "usable hardware on this host"

||| Run a device-pinned construction action under EAFP semantics. If
||| the backend's shim returned a NULL handle (it caught its own
||| allocation/transfer exception), surface `Left (DeviceUnavailable
||| (deviceName {d}))`; otherwise `Right` the tensor. This is the one
||| primitive every checked constructor builds on — it composes with
||| *any* existing `IO (Tensor ...)` producer (`tconstScalar`,
||| `tparam2d`, `toDevice`, …) rather than duplicating each.
export
attemptOn : {0 d : Device} -> UserDeviceCore d =>
            IO (Tensor dims d dt g) -> IO (Either DeviceError (Tensor dims d dt g))
attemptOn act = do
  t <- act
  pure $ if prim__handleIsNull t.tensorPtr == 1
           then Left (DeviceUnavailable (deviceName {d}))
           else Right t

||| `toDevice` under the EAFP gate: a move to an absent destination
||| device surfaces as `Left DeviceError` instead of aborting deep in
||| the backend. Wired to the same null-handle primitive as `attemptOn`.
||| The destination construction (`primIntraMigrate` /
||| `primCreateFromHost`) routes through the backend's guarded shim.
export
toDeviceChecked : {0 d1 : Type} -> (0 d2 : Type) ->
                  UserDeviceTransfer d1 => UserDeviceTransfer d2 =>
                  {rank : Nat} -> {dims : Vect rank Nat} ->
                  Tensor dims d1 dt WithGrad ->
                  IO (Either DeviceError (Tensor dims d2 dt WithGrad))
toDeviceChecked d2 src = attemptOn {d = d2} (toDevice d2 src)

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
weakenGrad : UserDeviceTape d => (1 _ : Tensor dims d dt g) -> IO (Tensor dims d dt NoGrad)
weakenGrad (MkTensor ptr pid) = do
  primIO (primSetRequiresGrad {d} ptr 0)
  pure (MkTensor ptr pid)

||| Pure type-level cast between grad-modes. `g` is 0-quantity in
||| `Tensor`'s declaration, so `MkTensor` is polymorphic in `g` and
||| destructure-reconstruct flips the type tag with no runtime work
||| and no type-system bypass. Used by `unfreezeLayer` impls to
||| retype tensor fields after flipping the C-side `requires_grad`
||| flag. Not a control surface for users — to *change* the runtime
||| flag, use `weakenGrad`.
export
retypeGrad : Tensor dims d dt g1 -> Tensor dims d dt g2
retypeGrad (MkTensor ptr pid) = MkTensor ptr pid


||| Type-level aliases for common Tensor shapes. Aliases route shape
||| arithmetic (e.g. `4 * o`) through a Nat-argument slot rather than
||| inlining inside a Vect literal — the latter triggers an Idris 2
||| type-checker hang on multiplicative Nat expressions.
||| (`Tensor [4 * o, i] d` hangs; `TMat (4 * o) i d` works.)
public export
0 TVec : Nat -> Device -> DType -> GradMode -> Type
TVec n d dt g = Tensor [n] d dt g

public export
0 TMat : Nat -> Nat -> Device -> DType -> GradMode -> Type
TMat m n d dt g = Tensor [m, n] d dt g

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

----------------------------------------------------------------------
-- Cross-dtype conversion: lossless via `UpcastableTo`, lossy via
-- explicit `tcastUnsafe`.
----------------------------------------------------------------------

||| Lossless precision upcast within a single dtype family
||| (`F32 → F64`, `Int 16 → Int 32`, `BFloat 16 → BFloat 32`, …).
||| The `UpcastableTo from to` constraint is solved by Idris's
||| auto-search via per-family `LTE m n` instances in `DType.Core`;
||| narrowing casts (`F64 → F32`) and cross-family casts
||| (`UInt 8 → F16`) have no `UpcastableTo` instance and use
||| `tcastUnsafe` (below) instead.
|||
||| Runtime: dispatches through `RuntimeDType to`'s `dtCastFrom`
||| method to the per-dtype `tensor_cast_dtype_<to>` C primitive.
||| Source dtype is read from the handle on the C side; the cast
||| op becomes a node in the autograd graph on backends that trace
||| it (mlx/torch).
export
tcast : {0 d : Device} -> UserDeviceTape d =>
        (UpcastableTo from to, IsDType from, IsDType to, RuntimeDType to, Compatible d to, Linked d) =>
        Tensor dims d from g -> IO (Tensor dims d to g)
tcast v = ioRerun (\_ => MkTensor (dtCastFrom {d} {t=to} v.tensorPtr (deviceStreamTag {d})) Nothing)

||| Explicit precision/dtype cast in ANY direction, including
||| narrowing (`F64 → F32`) and cross-family (`UInt 8 → F16`).
||| The caller takes responsibility for any precision loss or
||| representation change — calling `tcastUnsafe` is the explicit
||| signal that the conversion was intentional (mirrors the
||| `unsafePerformIO` / `believe_me` convention for primitives
||| where the caller takes responsibility).
|||
||| For lossless conversions, prefer `tcast` so the compiler
||| verifies via `UpcastableTo` that no information is lost. Use
||| `tcastUnsafe` only when the conversion is deliberately
||| narrowing or cross-family.
|||
||| Runtime path is the same as `tcast` (both dispatch through
||| `dtCastFrom`); the difference is purely the type-system gate.
export
tcastUnsafe : {0 d : Device} -> UserDeviceTape d =>
              (0 to : DType) -> (IsDType from, IsDType to, RuntimeDType to, Compatible d to, Linked d) =>
              Tensor dims d from g -> IO (Tensor dims d to g)
tcastUnsafe to v = ioRerun (\_ => MkTensor (dtCastFrom {d} {t=to} v.tensorPtr (deviceStreamTag {d})) Nothing)

||| Create a registered learnable [o, i] parameter from a flat (row-major)
||| double buffer. Mirrors Linear.nameLayer's tensor path.
export
tparam2d : {0 d : Device} -> UserDeviceTape d => RuntimeDType dt => Linked d => Compatible d dt => {o, i : Nat} -> (paramId : String) -> AnyPtr -> IO (Tensor [o, i] d dt WithGrad)
tparam2d {o} {i} pid buf = ioRerun (\_ =>
  let oI = cast {to=Int} o
      iI = cast {to=Int} i
      reg = primParamRegister {d} pid (dtCreateParam2d {d} {t=dt} oI iI buf (deviceStreamTag {d}))
  in MkTensor reg (Just pid))

||| Create a registered learnable [n] parameter from a double buffer.
export
tparam1d : {0 d : Device} -> UserDeviceTape d => RuntimeDType dt => Linked d => Compatible d dt => {n : Nat} -> (paramId : String) -> AnyPtr -> IO (Tensor [n] d dt WithGrad)
tparam1d {n} pid buf = ioRerun (\_ =>
  let nI = cast {to=Int} n
      reg = primParamRegister {d} pid (dtCreateParam1d {d} {t=dt} nI buf (deviceStreamTag {d}))
  in MkTensor reg (Just pid))

||| Wrap an existing 1D tensor handle as a non-parameter input.
||| Pure — no FFI side effect, just record construction.
export
tinput1d : {n : Nat} -> AnyPtr -> Tensor [n] d dt WithGrad
tinput1d t = MkTensor t Nothing

||| Wrap an existing 2D tensor handle as a non-parameter input.
||| Pure — no FFI side effect, just record construction.
export
tinput2d : {m, n : Nat} -> AnyPtr -> Tensor [m, n] d dt WithGrad
tinput2d t = MkTensor t Nothing

-- Arithmetic / linear algebra (autograd-tracked) ----------------------

||| Elementwise addition. Both operands share shape.
||| `%inline`: inlines to a direct `prim__add` + `MkTensor` allocation
||| at every call site. Critical for hot-path layers (LSTM/NTM/DNC
||| call this many times per timestep); without inlining, Idris2's
||| Chez codegen wraps each invocation in a closure dispatch that
||| adds ~20µs of Scheme-side overhead per call, accumulating to a
||| 2× regression on recurrent models.
export %inline
tadd : {0 d : Device} -> UserDeviceCore d => Tensor dims d dt g -> Tensor dims d dt g -> IO (Tensor dims d dt g)
tadd a b = ioRerun (\_ => MkTensor (primAdd {d} a.tensorPtr b.tensorPtr) Nothing)

||| Matrix-vector multiply: [m, n] · [n] -> [m]. `%inline` for the
||| same reason as `tadd` (hot path in recurrent forward passes).
export %inline
tmv : {0 d : Device} -> UserDeviceTape d =>
      Tensor [m, n] d dt g -> Tensor [n] d dt g -> IO (Tensor [m] d dt g)
tmv w x = ioRerun (\_ => MkTensor (primMv {d} w.tensorPtr x.tensorPtr) Nothing)

||| Fused 1D linear: y = W[m,n] · x[n] + bias[m]. One C call instead
||| of `tadd (tmv W x) bias` — collapses two FFI hops into one and
||| eliminates the intermediate Idris-side glue. Used by Layer.Linear's
||| applyVar and by NTM/DNC FCs.
export %inline
tlinear : {0 d : Device} -> UserDeviceTape d =>
          Tensor [o, i] d dt g -> Tensor [i] d dt g -> Tensor [o] d dt g -> IO (Tensor [o] d dt g)
tlinear w x bias = ioRerun (\_ =>
  MkTensor (primLinear {d} w.tensorPtr x.tensorPtr bias.tensorPtr) Nothing)

||| Fused batched linear: W[o,i] · X^T[b,i] + bias[o] -> [b, o].
export %inline
tlinear2d : {0 d : Device} -> UserDeviceTape d =>
            Tensor [o, i] d dt g -> Tensor [b, i] d dt g -> Tensor [o] d dt g -> IO (Tensor [b, o] d dt g)
tlinear2d w x bias = ioRerun (\_ =>
  MkTensor (primLinear2d {d} w.tensorPtr x.tensorPtr bias.tensorPtr) Nothing)

-- Per-sample extraction + scalar arithmetic (used by batched RL loss
-- builders: pluck a row from a [b, o] result, then a scalar from the
-- row, then build (q - target)^2 etc.) ---------------------------------

||| Select row `k` from a [b, n] Tensor, returning the n-vector slice.
||| Wraps `prim__select` on dim 0; preserves the autograd graph.
export
trowSelect : {0 d : Device} -> UserDeviceTape d => {b, n : Nat} ->
             Tensor [b, n] d dt g -> Int -> IO (Tensor [n] d dt g)
trowSelect t k = ioRerun (\_ => MkTensor (primSelect {d} t.tensorPtr 0 k) Nothing)

||| Select element `i` from an n-vector, returning a scalar Tensor.
export
telemSelect : {0 d : Device} -> UserDeviceTape d => {n : Nat} ->
              Tensor [n] d dt g -> Int -> IO (Tensor [] d dt g)
telemSelect t i = ioRerun (\_ => MkTensor (primSelect {d} t.tensorPtr 0 i) Nothing)

||| Scalar Tensor from a Double. Takes the value as a runtime argument
||| so Idris/Chez does NOT memoise the FFI result as a module-level
||| constant — same defence as `freshZeroLossT`. Non-grad: the C
||| backend creates a non-persistent scalar that is freed by the next
||| `tape_reset` (i.e. fine to call inside an epoch's loss builder).
export
||| Note: keeps the unified `prim__createScalar` (Phase 1 alias to
||| the primary backend) rather than dispatching via
||| `UserDeviceCore.primCreateScalar`. The op has no Tensor input, so
||| `d` would need to be inferred from the result's use-site and Idris
||| 2's bidirectional inference doesn't reliably push the instance
||| constraint through every call site that just lets-binds the
||| result. For built-in devices this matches the previous behavior
||| (alias to primary); for user-supplied devices, users should
||| construct scalars via their own `UserDeviceCore.primCreateScalar`
||| directly. Same compromise applies to `tparamScalar` and
||| `freshZeroLossT`.
tconstScalar : {0 d : Device} -> UserDeviceTape d => RuntimeDType dt => Linked d => Compatible d dt => Double -> IO (Tensor [] d dt WithGrad)
tconstScalar v = ioRerun (\_ => MkTensor (dtCreateScalar {d} {t=dt} v 0 (deviceStreamTag {d})) Nothing)

||| Subtract two equally-shaped Tensors (autograd-tracked).
export %inline
tsub : {0 d : Device} -> UserDeviceCore d => Tensor dims d dt g -> Tensor dims d dt g -> IO (Tensor dims d dt g)
tsub a b = ioRerun (\_ => MkTensor (primSub {d} a.tensorPtr b.tensorPtr) Nothing)

||| Elementwise multiply two equally-shaped Tensors (autograd-tracked).
export %inline
tmul : {0 d : Device} -> UserDeviceCore d => Tensor dims d dt g -> Tensor dims d dt g -> IO (Tensor dims d dt g)
tmul a b = ioRerun (\_ => MkTensor (primMul {d} a.tensorPtr b.tensorPtr) Nothing)

||| Negate a Tensor (autograd-tracked).
export %inline
tneg : {0 d : Device} -> UserDeviceCore d => Tensor dims d dt g -> IO (Tensor dims d dt g)
tneg a = ioRerun (\_ => MkTensor (primNeg {d} a.tensorPtr) Nothing)

||| Scale a Tensor by a Double (broadcasts the scalar; autograd-tracked).
||| Useful for mean-reduction (`tmulScalar loss (1.0 / cast n)`) and for
||| building per-sample loss expressions where one side of a product is
||| a runtime Double (e.g. DQN target value).
export %inline
tmulScalar : {0 d : Device} -> UserDeviceCore d => Tensor dims d dt g -> Double -> IO (Tensor dims d dt g)
tmulScalar v s = ioRerun (\_ => MkTensor (primMulScalar {d} v.tensorPtr s) Nothing)

||| Elementwise exponential (autograd-tracked).
export %inline
texp : {0 d : Device} -> UserDeviceCore d => Tensor dims d dt g -> IO (Tensor dims d dt g)
texp v = ioRerun (\_ => MkTensor (primExp {d} v.tensorPtr) Nothing)

||| Elementwise natural log (autograd-tracked).
export %inline
tlog : {0 d : Device} -> UserDeviceCore d => Tensor dims d dt g -> IO (Tensor dims d dt g)
tlog v = ioRerun (\_ => MkTensor (primLog {d} v.tensorPtr) Nothing)

||| Create a registered learnable scalar parameter (e.g. SAC's
||| state-independent log_std). Mirrors V1's `param`. The optimizer
||| picks it up automatically by paramId scope.
export
tparamScalar : {0 d : Device} -> UserDeviceTape d => RuntimeDType dt => Linked d => Compatible d dt => (paramId : String) -> (val : Double) -> IO (Tensor [] d dt WithGrad)
tparamScalar pid val = ioRerun (\_ =>
  let ptr = dtCreateScalar {d} {t=dt} val 1 (deviceStreamTag {d})    -- requires_grad=true
      reg = primParamRegister {d} pid ptr
  in MkTensor reg (Just pid))

||| Concatenate two [b, m] / [b, n] TVars along axis 1, producing
||| [b, m + n]. Wraps `prim__concat2dAxis1`. Used by SAC's actor loss
||| to build a [B, ObsDim + ActDim] Q-input from obs + reparametrized
||| action while preserving the autograd path through the action.
export
tconcat2dAxis1 : {0 d : Device} -> UserDeviceTape d => {b, m, n : Nat} ->
                 Tensor [b, m] d dt g -> Tensor [b, n] d dt g ->
                 IO (Tensor [b, m + n] d dt g)
tconcat2dAxis1 a b = ioRerun (\_ => MkTensor (primConcat2dAxis1 {d} a.tensorPtr b.tensorPtr) Nothing)

-- Activations (shape-preserving, pass-through autograd) ---------------
-- All `%inline` for hot-path performance — see `tadd` rationale.

export %inline
ttanh : {0 d : Device} -> UserDeviceCore d => Tensor dims d dt g -> IO (Tensor dims d dt g)
ttanh v = ioRerun (\_ => MkTensor (primTanh {d} v.tensorPtr) Nothing)

export %inline
tsigmoid : {0 d : Device} -> UserDeviceCore d => Tensor dims d dt g -> IO (Tensor dims d dt g)
tsigmoid v = ioRerun (\_ => MkTensor (primSigmoid {d} v.tensorPtr) Nothing)

export %inline
trelu : {0 d : Device} -> UserDeviceCore d => Tensor dims d dt g -> IO (Tensor dims d dt g)
trelu v = ioRerun (\_ => MkTensor (primClampMin {d} v.tensorPtr 0.0) Nothing)

export %inline
tgelu : {0 d : Device} -> UserDeviceTape d => Tensor dims d dt g -> IO (Tensor dims d dt g)
tgelu v = ioRerun (\_ => MkTensor (primGelu {d} v.tensorPtr) Nothing)

export %inline
tsilu : {0 d : Device} -> UserDeviceTape d => Tensor dims d dt g -> IO (Tensor dims d dt g)
tsilu v = ioRerun (\_ => MkTensor (primSilu {d} v.tensorPtr) Nothing)

export %inline
tleakyRelu : {0 d : Device} -> UserDeviceTape d => Double -> Tensor dims d dt g -> IO (Tensor dims d dt g)
tleakyRelu slope v = ioRerun (\_ => MkTensor (primLeakyRelu {d} v.tensorPtr slope) Nothing)

||| Softmax along axis 0 (1D vector).
export %inline
tsoftmax1d : {0 d : Device} -> UserDeviceTape d => {n : Nat} -> Tensor [n] d dt g -> IO (Tensor [n] d dt g)
tsoftmax1d v = ioRerun (\_ => MkTensor (primSoftmax {d} v.tensorPtr 0) Nothing)

||| Log-softmax along axis 0 (1D vector).
export %inline
tlogSoftmax1d : {0 d : Device} -> UserDeviceTape d => {n : Nat} -> Tensor [n] d dt g -> IO (Tensor [n] d dt g)
tlogSoftmax1d v = ioRerun (\_ => MkTensor (primLogSoftmax {d} v.tensorPtr 0) Nothing)

||| Fused LSTM gate computation: combined gates [4 * n] + previous cell [n]
||| → (new hidden [n], new cell [n]). Wraps `prim__lstmGatesPair`.
|||
||| The gate-vector size is encoded statically as `TVec (4 * n) d`
||| (alias for `Tensor [4 * n] d`). Routing the `4 * n` through the
||| `TVec` alias avoids the type-checker hang that direct
||| `Tensor [4 * n] d` triggers.
export
tlstmGatesPair : UserDeviceNN d => {n : Nat} -> TVec (4 * n) d dt g -> TVec n d dt g ->
                 IO (TVec n d dt g, TVec n d dt g)
tlstmGatesPair {n} combined prevCell = ioRerun (\_ =>
  let nI = cast {to=Int} n
      pair = primLstmGatesPair {d} combined.tensorPtr prevCell.tensorPtr nI
  in (MkTensor (primPairFirst {d} pair) Nothing, MkTensor (primPairSecond {d} pair) Nothing))

||| Allocate a zero-initialised persistent state Tensor of size [n].
||| Use for LSTM/RNN/GRU initial hidden + cell state. Persistent =
||| survives tape reset.
export
tzeroState1d : {0 d : Device} -> UserDeviceTape d => RuntimeDType dt => Linked d => Compatible d dt => {n : Nat} -> IO (Tensor [n] d dt g)
tzeroState1d {n} = ioRerun (\_ =>
  let nI = cast {to=Int} n
      buf = prim__allocDoubles nI
  in MkTensor (dtCreateState1d {d} {t=dt} nI buf (deviceStreamTag {d})) Nothing)

||| GRU cell — `nn.GRU` equation. Takes the two `[3 * n]` half-sums:
|||   ih = W_ih @ x + b_ih
|||   hh = W_hh @ h + b_hh
||| (computed by the caller via `tlinear`) plus the previous hidden
||| state. Internally:
|||   z = sigmoid(ih_z + hh_z),  r = sigmoid(ih_r + hh_r)
|||   n = tanh(ih_n + r * hh_n)
|||   h' = (1 - z) * n + z * prev
||| Pre-2026-05-09 this took a single fused `combined = ih + hh`
||| and ignored r (simplified GRU); aligned to the standard
||| `nn.GRU` equation so the example matches what library users
||| expect.
export
tgruCell : UserDeviceNN d => {n : Nat} -> TVec (3 * n) d dt g -> TVec (3 * n) d dt g -> TVec n d dt g -> IO (TVec n d dt g)
tgruCell {n} ih hh prevH = ioRerun (\_ =>
  let nI = cast {to=Int} n
  in MkTensor (primGruCell {d} ih.tensorPtr hh.tensorPtr prevH.tensorPtr nI) Nothing)

-- Scalar boundary --------------------------------------------------

||| Read the scalar value out of a `Tensor [] d`.
export
tensorItem : UserDeviceCore d => Tensor [] d dt g -> Double
tensorItem v = primItem {d} v.tensorPtr

||| Run backward on a loss tensor. The loss MUST be `WithGrad` —
||| a `NoGrad` scalar can't have come from a path the autograd tape
||| recorded, so backward would be a silent no-op at best and a
||| malformed-tape crash at worst. Rejecting at the type level
||| catches "loss computed inside `withNoGrad`, then fed to training"
||| — the bug class the entire `GradMode` refactor exists to prevent.
export
runBackward : UserDeviceTape d => Tensor [] d dt WithGrad -> IO ()
runBackward t = primIO (primBackward {d} t.tensorPtr)

-- Loss (vector targets → scalar loss) ---------------------------------

||| MSE loss over a 1D prediction/target pair. Sum-reduced.
export
tmseLoss : {0 d : Device} -> UserDeviceLinear d => {n : Nat} ->
           Tensor [n] d dt g -> Tensor [n] d dt g -> IO (Tensor [] d dt g)
tmseLoss p t = ioRerun (\_ =>
  let diff = primSub {d} p.tensorPtr t.tensorPtr in
  let sqDiff = primMul {d} diff diff in
  MkTensor (primSum {d} sqDiff) Nothing)

||| NLL loss against a one-hot target. Mirrors
||| `Example.Supervised.nllLossTensor` (divide by n to match the
||| reference's mean reduction).
export
tnllLoss : {0 d : Device} -> UserDeviceNN d => {n : Nat} ->
           Tensor [n] d dt g -> Tensor [n] d dt g -> IO (Tensor [] d dt g)
tnllLoss {n} p t = ioRerun (\_ =>
  let logP = primLogSoftmax {d} p.tensorPtr 0 in
  let prod = primMul {d} logP t.tensorPtr in
  let neg = primNeg {d} (primSum {d} prod) in
  MkTensor (primMulScalar {d} neg (1.0 / cast n)) Nothing)

||| Binary cross-entropy with logits, mean-reduced. Numerically stable
||| (wraps `primBceWithLogits`). For multi-element predictions/targets
||| use `tbceLoss : Tensor [n] d dt g-> Tensor [n] d dt g-> Tensor [] d dt g`;
||| the C op internally averages. Polymorphic in `g`: the loss's
||| grad-mode matches the predictions / targets, so a no-grad eval
||| `tbceLoss` (e.g. inside `withNoGrad`) returns a `NoGrad` scalar
||| that the type system will reject if accidentally fed to
||| `nativeTrainStep`.
export
tbceLoss : {0 d : Device} -> UserDeviceNN d => {n : Nat} ->
           Tensor [n] d dt g -> Tensor [n] d dt g -> IO (Tensor [] d dt g)
tbceLoss p t = ioRerun (\_ =>
  MkTensor (primBceWithLogits {d} p.tensorPtr t.tensorPtr) Nothing)

-- Optimizer shim ------------------------------------------------------

||| Fused native train step on a Tensor loss: zero_grad → backward →
||| clip → step. Reads `prim__item` BEFORE the step so the returned
||| scalar is not stale. Mirrors `nativeTrainStep`.
export
nativeTrainStep : {0 d : Device} -> UserDeviceTape d =>
                  NativeOptimizer d -> Tensor [] d dt WithGrad -> IO Double
nativeTrainStep opt loss = ioRerun (\_ =>
  let clipMode : Int
      clipMode = case opt.clipMode of NoClip => 0; ValueClip _ => 1; NormClip _ => 2
      clipVal  : Double
      clipVal  = case opt.clipMode of NoClip => 0.0; ValueClip v => v; NormClip v => v
      lossVal  = primItem {d} loss.tensorPtr
  in primNativeTrainStep {d} opt.handle clipMode clipVal loss.tensorPtr lossVal)
