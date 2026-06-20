||| Wrapped-handle ABI: Chez guardian plumbing, no-grad/gen brackets,
||| GC/RSS, host-staging + bulk loaders, and registry/diagnostic queries.
module Tensor.Handle

import Data.List
import Data.Maybe
import Data.Vect

import Array
import DType.Core
import Executor
import Tensor.Internal

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
--         by per-backend wraps in Executor/*.idr) or "primary" (set by
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
-- (tensor_retain_handle, tensor_release_handle) succeed. The %foreign "C:..." declarations elsewhere in this file
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
%foreign "scheme:(lambda (dummy) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t)))))) 1)"
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
export
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
-- name FFI bindings used by the old `toExecutor`. They're now unused
-- — `toExecutor` lives on top of `UserExecutorTransfer`'s per-backend
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

-- IDX-format dataset loading (file format Yann LeCun shipped MNIST in;
-- the helpers live in packages/backends/idx.c and are compiled ONCE
-- into libidrisml.dylib, not per-backend — so the dataset surface is
-- intentionally outside the UserExecutor* typeclass dispatch. The
-- Idris side hands the borrowed `idx_image_doubles` pointer to
-- `dtCreate1d` (a thin wrapper around the generic dtype-streamed
-- creator) to construct the tensor; no per-backend `mnist_get_image_<b>`
-- symbol exists.
%foreign "C:idx_load,libidrisml"
export
prim__idxLoad : String -> String -> AnyPtr

%foreign "C:idx_count,libidrisml"
export
prim__idxCount : AnyPtr -> Int

%foreign "C:idx_label_at,libidrisml"
export
prim__idxLabel : AnyPtr -> Int -> Int

%foreign "C:idx_image_doubles,libidrisml"
export
prim__idxImageDoubles : AnyPtr -> Int -> AnyPtr

-- The `idxImage` convenience helper is defined further down, after
-- `dtCreate1d` is in scope.

-- Parameter registry: `primParamRegister {ex}` (UserExecutorTraining) is
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
||| `UserExecutorTraining ex`. `d` doesn't appear in the action type, so
||| callers pin it explicitly (`withNoGrad {ex=ExampleExecutor} ...`).
export
withNoGrad : {0 ex : Executor} -> UserExecutorTraining ex => IO a -> IO a
withNoGrad act = do
  primIO (primNoGradBegin {ex})
  result <- act
  forceMajorGc
  _ <- drainManagedHandles
  primIO (primNoGradEnd {ex})
  pure result

||| `withNoGrad` for blocks whose result carries live tensors created
||| inside the block (e.g. a cached embedding, an inference output kept
||| for later use). `keepAliveRetain` bumps their refcount so the block-
||| exit generation-scoped free spares them, then releases afterwards.
||| The common scalar-returning eval/rollout loops use plain `withNoGrad`.
export
withNoGradKeep : {0 ex : Executor} -> UserExecutorTraining ex => KeepAlive a => IO a -> IO a
withNoGradKeep act = do
  primIO (primNoGradBegin {ex})
  result <- act
  keepAliveRetain result
  forceMajorGc
  _ <- drainManagedHandles
  primIO (primNoGradEnd {ex})
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
withGenFree : {0 ex : Executor} -> UserExecutorTraining ex => KeepAlive a => IO a -> IO a
withGenFree act = do
  primIO (primEpochBegin {ex})
  result <- act
  keepAliveRetain result
  primIO (primEpochEnd {ex})
  keepAliveRelease result
  pure result

----------------------------------------------------------------------
-- Sequencing helper
----------------------------------------------------------------------

-- Force evaluation of first arg, return second.
-- Must use concrete AnyPtr types (not polymorphic) to avoid
-- argument count issues at the FFI boundary.

----------------------------------------------------------------------
-- C-side allocation + bulk-load helpers: moved to Tensor.Internal
----------------------------------------------------------------------

-- 3D batched attention ops

||| Tile a 2D tensor: `[m, n] -> [m*rep0, n*rep1]`. Element `(i, j)` in the
||| output equals element `(i mod m, j mod n)` in the input.

-- Array pointer array: stack scalar Tensor tensorPtrs to create
-- a 1D/2D tensor that preserves the autograd graph.

-- Returns the array for threading

-- N-ary cat: caller retains ownership of the handle array.
-- See tensor_cat in backend.h.

-- Batch [...] tensors into [count, ...]. Equivalent to stack at dim=0.

-- Backend-agnostic raw-bytes reader from a safetensors file. Pure file
-- I/O — no tensor handles, no per-backend dispatch (symbol lives in
-- safetensors.c with no rename). Returns the byte count copied on
-- success, or a negative value on error (missing file/key, malformed
-- header, or `outCap` too small). Used by HF BitNet's ternary-weight
-- load path: HF stores those as uint8 [(o+3)/4, i] with a custom 2-bit
-- encoding the standard `param_load*` dtype gate would refuse.
%foreign "C:safetensors_read_raw_bytes,libidrisml"
prim__safetensorsReadRawBytes : String -> String -> AnyPtr -> Int -> PrimIO Int

||| Read the raw on-disk bytes of a named tensor from a safetensors
||| file into a host buffer. Returns the byte count copied (>= 0) on
||| success, or a negative value on error. The caller owns `outBuf`
||| and must keep it alive across the call.
export
safetensorsReadRawBytes : (path : String) -> (key : String) ->
                          (outBuf : AnyPtr) -> (outCap : Int) ->
                          IO Int
safetensorsReadRawBytes path key buf cap =
  primIO (prim__safetensorsReadRawBytes path key buf cap)

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
||| The underlying C `tensor_create_1d_f64` (via dtCreate1d) frees the
||| input buffer after copying.
export
bulkToTensor : {0 ex : Executor} -> Backend ex dt => {n : Nat} -> Vector n Double -> AnyPtr
bulkToTensor {n} (VArray elems) =
  let nI = cast {to=Int} n
      buf  = prim__allocDoubles nI
      buf' = packDoubleBuf buf 0 elems
  in dtCreate1d {ex} {t=dt} nI buf' 0 (deviceStreamTag {ex})
  where
    packDoubleBuf : AnyPtr -> Int -> Vect k (Scalar Double) -> AnyPtr
    packDoubleBuf buf _ []                   = buf
    packDoubleBuf buf off (SArray v :: rest) =
      let buf' = prim__setDouble buf off v
      in packDoubleBuf buf' (off + 1) rest

||| Bulk-convert a Vect of Vectors of Doubles to a [b, i] C tensor handle.
||| The C tensor_create_2d function frees the input buffer after copying.
||| Use to stack a per-sample input batch into a single batched tensor.
export
bulkToTensor2d : {0 ex : Executor} -> Backend ex dt => {b, i : Nat} -> Vect b (Vector i Double) -> AnyPtr
bulkToTensor2d {b} {i} rows =
  let bI = cast {to=Int} b
      iI   = cast {to=Int} i
      buf  = prim__allocDoubles (bI * iI)
      buf' = packRows buf 0 rows
  in dtCreate2d {ex} {t=dt} bI iI buf' 0 (deviceStreamTag {ex})
  where
    packRow : AnyPtr -> Int -> Vect k (Scalar Double) -> AnyPtr
    packRow buf _ []                   = buf
    packRow buf off (SArray v :: rest) =
      let buf' = prim__setDouble buf off v
      in packRow buf' (off + 1) rest
    packRows : AnyPtr -> Int -> Vect k (Vector i Double) -> AnyPtr
    packRows buf _ []                     = buf
    packRows buf off (VArray row :: rest) =
      let buf' = packRow buf off row
      in packRows buf' (off + cast {to=Int} i) rest

||| Concatenate a list of per-sample [k] tensor handles into one [n*k]
||| handle. Routes through `primCat2 {ex}` so the MLX stream tag follows
||| the type-level device. Generic tensor-stack primitive (used by the
||| batched epoch runners and the `Data.Stream` collator); not a
||| backprop concept, so it lives here next to `bulkToTensor2d`. Pair
||| with `primReshape2d` to get a [n, k] batch.
||| `partial`: the empty-list case crashes (callers always pass a
||| non-empty batch); Backprop carried it under `%default partial`.
export
partial
catAllTensors : {0 ex : Executor} -> UserExecutorLinear ex => List AnyPtr -> AnyPtr
catAllTensors []               = idris_crash "catAllTensors: empty list"
catAllTensors [x]              = x
catAllTensors (x :: y :: rest) = catAllTensors {ex} (primCat2 {ex} x y :: rest)

----------------------------------------------------------------------
-- Handle-array staging — pack a batch of wrapped Tensor handles into a
-- C `TensorHandle*` buffer for the single-FFI `primBatch` collation
-- (DataStream.collate). `alloc`/`free` are raw host-memory helpers
-- (plain C:, not manifest-bound, Idris caches the foreign itself);
-- `set_return` carries a wrapped Tensor arg, so its scheme wrapper
-- unwraps slot 2 before storing the raw handle. All three are pure-typed
-- and reorder-prone — callers sequence each via `ioRerun` / `primIO`.
----------------------------------------------------------------------

%foreign "C:tensor_ptr_array_alloc,libidrisml"
export
prim__ptrArrayAlloc : Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (when (not (top-level-bound? 'idris-ffi-tensor-ptr-array-set-return)) (set-top-level-value! 'idris-ffi-tensor-ptr-array-set-return (foreign-procedure \"tensor_ptr_array_set_return\" (void* int void*) void*))) ((top-level-value 'idris-ffi-tensor-ptr-array-set-return) a0 a1 (vector-ref a2 2)))"
export
prim__ptrArraySet : AnyPtr -> Int -> AnyPtr -> AnyPtr

%foreign "C:tensor_ptr_array_free,libidrisml"
export
prim__ptrArrayFree : AnyPtr -> PrimIO ()

||| Bulk-convert a Vector of Doubles to a persistent C tensor handle.
||| Persistent tensors survive tape resets — use when data is created once
||| and reused across training epochs.
export
vectorToTensorPersistent : {0 ex : Executor} -> Backend ex dt => {n : Nat} -> Vector n Double -> AnyPtr
vectorToTensorPersistent {n} (VArray elems) =
  let nI = cast {to=Int} n
      buf  = prim__allocDoubles nI
      buf' = packBuf buf 0 elems
  in dtCreateState1d {ex} {t=dt} nI buf' (deviceStreamTag {ex})
  where
    packBuf : AnyPtr -> Int -> Vect k (Scalar Double) -> AnyPtr
    packBuf buf _ []                   = buf
    packBuf buf off (SArray v :: rest) = packBuf (prim__setDouble buf off v) (off + 1) rest

-- runBackward is defined post-Tensor record below; the type-level
-- gate (Tensor [] ex dt WithGrad-> IO ()) lives there.

-- Registry queries dispatch through the in-scope `UserExecutorTraining ex`:
-- each backend's registry is a separate TU-local table, so `{ex}`
-- selects which one is read.

||| Get parameter count (for gradient inspection).
export
getParamCount : UserExecutorTraining ex => IO Int
getParamCount = primIO (primParamCount {ex})

||| Get parameter name by index.
export
getParamName : UserExecutorTraining ex => Int -> IO String
getParamName i = primIO (primParamName {ex} i)

||| True if the `i`th registered entry is a non-learnable buffer (saved /
||| loaded by name, but never stepped by the optimizer). Used by the freeze
||| walk to skip buffers.
export
getParamIsBuffer : UserExecutorTraining ex => Int -> IO Bool
getParamIsBuffer i = (/= 0) <$> primIO (primParamIsBuffer {ex} i)

||| Get gradient element for param i, element j.
export
getParamGradAt : UserExecutorTraining ex => Int -> Int -> IO Double
getParamGradAt i j = primIO (primParamGradItemAt {ex} i j)

||| Zero all parameter gradients.
export
zeroAllGrads : UserExecutorTraining ex => IO ()
zeroAllGrads = primIO (primParamZeroAll {ex})

||| Get the name of the active backend ("tape", "mlx", "torch").
||| This is the backend *family*, not the hardware variant — exactly
||| `backendTag {ex}` from `UserExecutorTransfer`. (The old C
||| `backend_name` returned the same string; routing through the
||| instance drops the unified-name FFI.)
export
backendName : UserExecutorTransfer ex => String
backendName = backendTag {ex}

||| Force the backend to release every persistent at::Tensor /
||| mx::array and reset the param registry. Inference programs that
||| return ~hundreds of MB of live tensor handles to `main` hit a
||| post-main libtorch CPUAllocator / OS-cleanup tail of tens of minutes
||| (torch-cpu, mlx-cpu; GPU lanes release async). Calling this at the
||| end of `main` shifts the destructor cascade inside the timed region
||| so the cost is observable + bounded. Cheap on tape (arena reset).
||| Routes via `UserExecutorTraining ex` so the dispatch picks the active
||| backend's suffixed symbol (the unified-name alias machinery was
||| retired in 2026-05; see Makefile lines 318-323).
export
releaseAllPersistent : {0 ex : Executor} -> UserExecutorTraining ex => IO ()
releaseAllPersistent = primIO (primReleaseAllPersistent {ex})

||| Reset the backend's arena + autograd tape between inference
||| forward passes. On tape this drops every intermediate from the
||| previous forward — required for multi-token decode on large
||| models (without it, the arena grows ~GB per forward and OOMs).
||| On torch + mlx, drops forward intermediates + zeros param grads
||| (mild beneficial, no semantic change inside `withNoGrad`).
||| **UNSAFE in training** — clobbers any param grads in flight.
||| Routes via `UserExecutorTraining ex`.
export
resetForEval : {0 ex : Executor} -> UserExecutorTraining ex => IO ()
resetForEval = primIO (primResetForEval {ex})

||| TODO #393 op-submission diagnostic — zero the per-forward op
||| counter on backend `d`. On torch counts every `at::Tensor` wrap
||| in `from_tensor()` (one per graph node); on tape + mlx it's a
||| no-op so this resets a counter that always reads 0.
export
perfReset : {0 ex : Executor} -> UserExecutorTraining ex => IO ()
perfReset = primIO (primPerfReset {ex})

||| Read the current op-submission counter on backend `d`. Pair with
||| `perfReset` for per-forward op counts (`reset` before forward,
||| `perfOpCount` after). On torch this is the number of graph nodes
||| since the last reset; on tape + mlx returns 0.
export
perfOpCount : {0 ex : Executor} -> UserExecutorTraining ex => IO Int
perfOpCount = primIO (primPerfOpCount {ex})

||| Reset profiling counters for backend `d`.
export
profileReset : UserExecutorTraining ex => IO ()
profileReset = primIO (primProfileReset {ex})

||| Print backend `d`'s profile breakdown to stderr.
export
profileReport : UserExecutorTraining ex => IO ()
profileReport = primIO (primProfileReport {ex})
