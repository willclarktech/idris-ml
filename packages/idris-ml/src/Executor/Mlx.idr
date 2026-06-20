||| `MlxExecutor` — `UserExecutorCore` instance for the mlx backend.
|||
||| Forwards to the mlx-suffixed C symbols emitted under Phase 1's
||| `rename_mlx.h` (e.g. `tensor_add_mlx`). Only resolvable at runtime
||| if the build's BACKEND list includes `mlx` (Apple-only).
module Executor.Mlx

import BackendLib
import DType.Core
import Executor.Core
import Hardware
import Preset

----------------------------------------------------------------------
-- Per-symbol bindings to the mlx backend's suffixed C exports
----------------------------------------------------------------------

-- UserExecutorCore (MlxExecutor s) instance methods call the streamed
-- variants below; the trailing `Int` stream_tag is derived from the
-- type-level `s` via `streamTag` (0 = MCpu, 1 = MGpu). The unstreamed
-- `prim__*Mlx` declarations are kept for any caller that hasn't moved
-- to the streamed surface (currently none in this file).

%foreign "scheme:(lambda (a0 a1 a2) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (let ((raw_r ((foreign-procedure \"tensor_create_scalar_mlx_streamed\" (double int int) void*) a0 a1 a2))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__createScalarMlxStreamed : Double -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  ((foreign-procedure \"tensor_free_mlx_streamed\" (void* int) void) (vector-ref a0 2) a1))"
prim__freeMlxStreamed : AnyPtr -> Int -> ()

%foreign "scheme:(lambda (a0 a1)  ((foreign-procedure \"tensor_item_mlx_streamed\" (void* int) double) (vector-ref a0 2) a1))"
prim__itemMlxStreamed : AnyPtr -> Int -> Double

%foreign "scheme:(lambda (a0 a1 a2)  ((foreign-procedure \"tensor_item_1d_mlx_streamed\" (void* int int) double) (vector-ref a0 2) a1 a2))"
prim__item1dMlxStreamed : AnyPtr -> Int -> Int -> Double

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_clone_mlx_streamed\" (void* int) void*) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__cloneMlxStreamed : AnyPtr -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_add_mlx_streamed\" (void* void* int) void*) (vector-ref a0 2) (vector-ref a1 2) a2))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__addMlxStreamed : AnyPtr -> AnyPtr -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_sub_mlx_streamed\" (void* void* int) void*) (vector-ref a0 2) (vector-ref a1 2) a2))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__subMlxStreamed : AnyPtr -> AnyPtr -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_mul_mlx_streamed\" (void* void* int) void*) (vector-ref a0 2) (vector-ref a1 2) a2))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__mulMlxStreamed : AnyPtr -> AnyPtr -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_div_mlx_streamed\" (void* void* int) void*) (vector-ref a0 2) (vector-ref a1 2) a2))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__divMlxStreamed : AnyPtr -> AnyPtr -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_neg_mlx_streamed\" (void* int) void*) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__negMlxStreamed : AnyPtr -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_abs_mlx_streamed\" (void* int) void*) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__absMlxStreamed : AnyPtr -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_exp_mlx_streamed\" (void* int) void*) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__expMlxStreamed : AnyPtr -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_log_mlx_streamed\" (void* int) void*) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__logMlxStreamed : AnyPtr -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_sqrt_mlx_streamed\" (void* int) void*) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__sqrtMlxStreamed : AnyPtr -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_pow_mlx_streamed\" (void* void* int) void*) (vector-ref a0 2) (vector-ref a1 2) a2))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__powMlxStreamed : AnyPtr -> AnyPtr -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_sigmoid_mlx_streamed\" (void* int) void*) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__sigmoidMlxStreamed : AnyPtr -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_tanh_mlx_streamed\" (void* int) void*) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__tanhMlxStreamed : AnyPtr -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_add_scalar_mlx_streamed\" (void* double int) void*) (vector-ref a0 2) a1 a2))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__addScalarMlxStreamed : AnyPtr -> Double -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_mul_scalar_mlx_streamed\" (void* double int) void*) (vector-ref a0 2) a1 a2))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__mulScalarMlxStreamed : AnyPtr -> Double -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_clamp_min_mlx_streamed\" (void* double int) void*) (vector-ref a0 2) a1 a2))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__clampMinMlxStreamed : AnyPtr -> Double -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2 a3)  (let ((raw_r ((foreign-procedure \"tensor_clamp_mlx_streamed\" (void* double double int) void*) (vector-ref a0 2) a1 a2 a3))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__clampMlxStreamed : AnyPtr -> Double -> Double -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_round_mlx_streamed\" (void* int) void*) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__roundMlxStreamed : AnyPtr -> Int -> AnyPtr

----------------------------------------------------------------------
-- MlxStream + MlxExecutor parameterized family
--
-- `MlxExecutor` is parameterized over a stream tag (`MGpu` vs `MCpu`) so
-- that `MlxExecutor MGpu` and `MlxExecutor MCpu` are distinct device types at
-- the type level while sharing one set of `UserExecutor*` instances.
-- The instance bodies derive an `Int` stream tag from `s` via
-- `streamTag` and thread it through the `_mlx_streamed` FFI surface
-- to `mx::StreamContext` on the C side, so each op runs on the
-- stream the type system claimed. Mirrors the `CUDA Nat` precedent
-- in `Executor.idr`.
--
-- Ergonomic aliases `MlxGpu : Type` and `MlxCpu : Type` are exported
-- below so callers can write `Tensor [4] MlxGpu F32 WithGrad`
-- without the constructor noise.
----------------------------------------------------------------------

||| Stream tag for MLX devices.
public export
data MlxStream : Type where
  MGpu : MlxStream
  MCpu : MlxStream

||| MLX device, parameterized over its stream tag.
public export
data MlxExecutor : MlxStream -> Type where
  MkMlxExecutor : MlxExecutor s

||| `MlxExecutor MGpu` alias. Metal GPU stream. Only supports F32; dt
||| has no `Compatible` instance and tensors of `MlxGpu dt` fail to
||| typecheck at the construction site.
public export
MlxGpu : Type
MlxGpu = MlxExecutor MGpu

||| `MlxExecutor MCpu` alias. mlx CPU stream. Supports both F32 and dt.
public export
MlxCpu : Type
MlxCpu = MlxExecutor MCpu

||| Int encoding of an `MlxStream` for the streamed FFI surface.
||| `MCpu → 0`, `MGpu → 1`. Mirrored on the C side by
||| `stream_for_tag(int)` in `backend_mlx.cpp`. Each `UserExecutorCore`
||| (and sibling-interface) method on `MlxExecutor s` derives the tag from
||| `s` and threads it to the corresponding `_mlx_streamed` FFI so
||| the op runs on the correct mlx stream — honouring the type-level
||| device parameter rather than the global `mx::set_default_device`.
public export
streamTag : MlxStream -> Int
streamTag MCpu = 0
streamTag MGpu = 1

%foreign "scheme:(lambda (a0 a1 a2)  (when (not (top-level-bound? 'idris-ffi-tensor-item-2d-mlx)) (set-top-level-value! 'idris-ffi-tensor-item-2d-mlx (foreign-procedure \"tensor_item_2d_mlx\" (void* int int) double))) ((top-level-value 'idris-ffi-tensor-item-2d-mlx) (vector-ref a0 2) a1 a2))"
prim__item2dMlx : AnyPtr -> Int -> Int -> Double
%foreign "scheme:(lambda (a0 a1 a2 a3) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-tensor-one-hot-mlx)) (set-top-level-value! 'idris-ffi-tensor-one-hot-mlx (foreign-procedure \"tensor_one_hot_mlx\" (void* int int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-mlx)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-mlx (foreign-procedure \"tensor_retain_handle_mlx\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-one-hot-mlx) a0 a1 a2 a3))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-mlx) raw_r) wr)))"
prim__oneHotMlx : AnyPtr -> Int -> Int -> Int -> AnyPtr

public export
{s : MlxStream} -> UserExecutorCore (MlxExecutor s) where
  deviceName       = case s of
                       MGpu => "mlx:gpu"
                       MCpu => "mlx:cpu"
  -- >>> GENERATED FROM ffi_manifest.py — gen-executor-instances.py >>>
  primAbs a0             = prim__absMlxStreamed a0 (streamTag s)
  primAdd a0 a1          = prim__addMlxStreamed a0 a1 (streamTag s)
  primAddScalar a0 a1    = prim__addScalarMlxStreamed a0 a1 (streamTag s)
  primClamp a0 a1 a2     = prim__clampMlxStreamed a0 a1 a2 (streamTag s)
  primClampMin a0 a1     = prim__clampMinMlxStreamed a0 a1 (streamTag s)
  primClone a0           = prim__cloneMlxStreamed a0 (streamTag s)
  primCreateScalar a0 a1 = prim__createScalarMlxStreamed a0 a1 (streamTag s)
  primDiv a0 a1          = prim__divMlxStreamed a0 a1 (streamTag s)
  primExp a0             = prim__expMlxStreamed a0 (streamTag s)
  primFree a0            = prim__freeMlxStreamed a0 (streamTag s)
  primItem a0            = prim__itemMlxStreamed a0 (streamTag s)
  primItem1d a0 a1       = prim__item1dMlxStreamed a0 a1 (streamTag s)
  primLog a0             = prim__logMlxStreamed a0 (streamTag s)
  primMul a0 a1          = prim__mulMlxStreamed a0 a1 (streamTag s)
  primMulScalar a0 a1    = prim__mulScalarMlxStreamed a0 a1 (streamTag s)
  primNeg a0             = prim__negMlxStreamed a0 (streamTag s)
  primPow a0 a1          = prim__powMlxStreamed a0 a1 (streamTag s)
  primRound a0           = prim__roundMlxStreamed a0 (streamTag s)
  primSigmoid a0         = prim__sigmoidMlxStreamed a0 (streamTag s)
  primSqrt a0            = prim__sqrtMlxStreamed a0 (streamTag s)
  primSub a0 a1          = prim__subMlxStreamed a0 a1 (streamTag s)
  primTanh a0            = prim__tanhMlxStreamed a0 (streamTag s)
  -- <<< END GENERATED <<<

public export
{s : MlxStream} -> UserExecutorStreamed (MlxExecutor s) where
  deviceStreamTag = streamTag s

----------------------------------------------------------------------
-- Linear-slice FFI bindings (mlx-suffixed, streamed)
----------------------------------------------------------------------

%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_mv_mlx_streamed\" (void* void* int) void*) (vector-ref a0 2) (vector-ref a1 2) a2))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__mvMlxStreamed : AnyPtr -> AnyPtr -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_mm_mlx_streamed\" (void* void* int) void*) (vector-ref a0 2) (vector-ref a1 2) a2))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__mmMlxStreamed : AnyPtr -> AnyPtr -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_matmul_mlx_streamed\" (void* void* int) void*) (vector-ref a0 2) (vector-ref a1 2) a2))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__matmulMlxStreamed : AnyPtr -> AnyPtr -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2 a3)  (let ((raw_r ((foreign-procedure \"tensor_linear_mlx_streamed\" (void* void* void* int) void*) (vector-ref a0 2) (vector-ref a1 2) (vector-ref a2 2) a3))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__linearMlxStreamed : AnyPtr -> AnyPtr -> AnyPtr -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_dot_mlx_streamed\" (void* void* int) void*) (vector-ref a0 2) (vector-ref a1 2) a2))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__dotMlxStreamed : AnyPtr -> AnyPtr -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_outer_mlx_streamed\" (void* void* int) void*) (vector-ref a0 2) (vector-ref a1 2) a2))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__outerMlxStreamed : AnyPtr -> AnyPtr -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_bmm_mlx_streamed\" (void* void* int) void*) (vector-ref a0 2) (vector-ref a1 2) a2))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__bmmMlxStreamed : AnyPtr -> AnyPtr -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2 a3)  (let ((raw_r ((foreign-procedure \"tensor_linear_2d_mlx_streamed\" (void* void* void* int) void*) (vector-ref a0 2) (vector-ref a1 2) (vector-ref a2 2) a3))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__linear2dMlxStreamed : AnyPtr -> AnyPtr -> AnyPtr -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_sum_mlx_streamed\" (void* int) void*) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__sumMlxStreamed : AnyPtr -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_mean_mlx_streamed\" (void* int) void*) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__meanMlxStreamed : AnyPtr -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_min_mlx_streamed\" (void* int) void*) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__tensorMinMlxStreamed : AnyPtr -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_max_mlx_streamed\" (void* int) void*) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__tensorMaxMlxStreamed : AnyPtr -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2 a3)  (let ((raw_r ((foreign-procedure \"tensor_sum_dim_mlx_streamed\" (void* int int int) void*) (vector-ref a0 2) a1 a2 a3))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__sumDimMlxStreamed : AnyPtr -> Int -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2 a3)  (let ((raw_r ((foreign-procedure \"tensor_select_mlx_streamed\" (void* int int int) void*) (vector-ref a0 2) a1 a2 a3))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__selectMlxStreamed : AnyPtr -> Int -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_unsqueeze_mlx_streamed\" (void* int int) void*) (vector-ref a0 2) a1 a2))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__unsqueezeMlxStreamed : AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_squeeze_mlx_streamed\" (void* int int) void*) (vector-ref a0 2) a1 a2))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__squeezeMlxStreamed : AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2 a3)  (let ((raw_r ((foreign-procedure \"tensor_stack_mlx_streamed\" (void* int int int) void*) a0 a1 a2 a3))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__stackMlxStreamed : AnyPtr -> Int -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-batch-mlx)) (set-top-level-value! 'idris-ffi-tensor-batch-mlx (foreign-procedure \"tensor_batch_mlx\" (void* int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-mlx)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-mlx (foreign-procedure \"tensor_retain_handle_mlx\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-batch-mlx) a0 a1))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-mlx) raw_r) wr)))"
prim__batchMlx : AnyPtr -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_view_1d_mlx_streamed\" (void* int int) void*) (vector-ref a0 2) a1 a2))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__view1dMlxStreamed : AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2 a3)  (let ((raw_r ((foreign-procedure \"tensor_view_2d_mlx_streamed\" (void* int int int) void*) (vector-ref a0 2) a1 a2 a3))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__view2dMlxStreamed : AnyPtr -> Int -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_reshape_1d_mlx_streamed\" (void* int int) void*) (vector-ref a0 2) a1 a2))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__reshape1dMlxStreamed : AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2 a3)  (let ((raw_r ((foreign-procedure \"tensor_reshape_2d_mlx_streamed\" (void* int int int) void*) (vector-ref a0 2) a1 a2 a3))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__reshape2dMlxStreamed : AnyPtr -> Int -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2 a3 a4)  (let ((raw_r ((foreign-procedure \"tensor_reshape_3d_mlx_streamed\" (void* int int int int) void*) (vector-ref a0 2) a1 a2 a3 a4))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__reshape3dMlxStreamed : AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2 a3 a4 a5)  (let ((raw_r ((foreign-procedure \"tensor_reshape_4d_mlx_streamed\" (void* int int int int int) void*) (vector-ref a0 2) a1 a2 a3 a4 a5))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__reshape4dMlxStreamed : AnyPtr -> Int -> Int -> Int -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2 a3)  (let ((raw_r ((foreign-procedure \"tensor_tile_2d_mlx_streamed\" (void* int int int) void*) (vector-ref a0 2) a1 a2 a3))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__tile2dMlxStreamed : AnyPtr -> Int -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2 a3 a4)  (let ((raw_r ((foreign-procedure \"tensor_narrow_mlx_streamed\" (void* int int int int) void*) (vector-ref a0 2) a1 a2 a3 a4))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__narrowMlxStreamed : AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_transpose_last2_mlx_streamed\" (void* int) void*) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__transposeLast2MlxStreamed : AnyPtr -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_transpose_2d_mlx_streamed\" (void* int) void*) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__transpose2dMlxStreamed : AnyPtr -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2 a3)  (let ((raw_r ((foreign-procedure \"tensor_cat_mlx_streamed\" (void* int int int) void*) a0 a1 a2 a3))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__catMlxStreamed : AnyPtr -> Int -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_cat2_mlx_streamed\" (void* void* int) void*) (vector-ref a0 2) (vector-ref a1 2) a2))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__cat2MlxStreamed : AnyPtr -> AnyPtr -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_concat_2d_axis1_mlx_streamed\" (void* void* int) void*) (vector-ref a0 2) (vector-ref a1 2) a2))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__concat2dAxis1MlxStreamed : AnyPtr -> AnyPtr -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2 a3)  (let ((raw_r ((foreign-procedure \"tensor_gather_mlx_streamed\" (void* void* int int) void*) (vector-ref a0 2) (vector-ref a1 2) a2 a3))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__gatherMlxStreamed : AnyPtr -> AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2 a3 a4)  (let ((raw_r ((foreign-procedure \"tensor_gather_rows_mlx_streamed\" (void* void* int int int) void*) (vector-ref a0 2) (vector-ref a1 2) a2 a3 a4))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__gatherRowsMlxStreamed : AnyPtr -> AnyPtr -> Int -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2 a3)  (let ((raw_r ((foreign-procedure \"tensor_max_rows_mlx_streamed\" (void* int int int) void*) (vector-ref a0 2) a1 a2 a3))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__maxRowsMlxStreamed : AnyPtr -> Int -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2 a3)  (let ((raw_r ((foreign-procedure \"tensor_scatter_add_mlx_streamed\" (void* void* int int) void*) (vector-ref a0 2) (vector-ref a1 2) a2 a3))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__scatterAddMlxStreamed : AnyPtr -> AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2 a3)  (let ((raw_r ((foreign-procedure \"tensor_argsort_mlx_streamed\" (void* int int int) void*) (vector-ref a0 2) a1 a2 a3))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__argsortMlxStreamed : AnyPtr -> Int -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_cumprod_mlx_streamed\" (void* int int) void*) (vector-ref a0 2) a1 a2))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__cumprodMlxStreamed : AnyPtr -> Int -> Int -> AnyPtr

public export
{s : MlxStream} -> UserExecutorLinear (MlxExecutor s) where
  -- >>> GENERATED FROM ffi_manifest.py — gen-executor-instances.py >>>
  primArgsort a0 a1 a2         = prim__argsortMlxStreamed a0 a1 a2 (streamTag s)
  primBatch                    = prim__batchMlx
  primBmm a0 a1                = prim__bmmMlxStreamed a0 a1 (streamTag s)
  primCat a0 a1 a2             = prim__catMlxStreamed a0 a1 a2 (streamTag s)
  primCat2 a0 a1               = prim__cat2MlxStreamed a0 a1 (streamTag s)
  primConcat2dAxis1 a0 a1      = prim__concat2dAxis1MlxStreamed a0 a1 (streamTag s)
  primCumprod a0 a1            = prim__cumprodMlxStreamed a0 a1 (streamTag s)
  primDot a0 a1                = prim__dotMlxStreamed a0 a1 (streamTag s)
  primGather a0 a1 a2          = prim__gatherMlxStreamed a0 a1 a2 (streamTag s)
  primGatherRows a0 a1 a2 a3   = prim__gatherRowsMlxStreamed a0 a1 a2 a3 (streamTag s)
  primLinear a0 a1 a2          = prim__linearMlxStreamed a0 a1 a2 (streamTag s)
  primLinear2d a0 a1 a2        = prim__linear2dMlxStreamed a0 a1 a2 (streamTag s)
  primMatmul a0 a1             = prim__matmulMlxStreamed a0 a1 (streamTag s)
  primMaxRows a0 a1 a2         = prim__maxRowsMlxStreamed a0 a1 a2 (streamTag s)
  primMean a0                  = prim__meanMlxStreamed a0 (streamTag s)
  primMm a0 a1                 = prim__mmMlxStreamed a0 a1 (streamTag s)
  primMv a0 a1                 = prim__mvMlxStreamed a0 a1 (streamTag s)
  primNarrow a0 a1 a2 a3       = prim__narrowMlxStreamed a0 a1 a2 a3 (streamTag s)
  primOuter a0 a1              = prim__outerMlxStreamed a0 a1 (streamTag s)
  primReshape1d a0 a1          = prim__reshape1dMlxStreamed a0 a1 (streamTag s)
  primReshape2d a0 a1 a2       = prim__reshape2dMlxStreamed a0 a1 a2 (streamTag s)
  primReshape3d a0 a1 a2 a3    = prim__reshape3dMlxStreamed a0 a1 a2 a3 (streamTag s)
  primReshape4d a0 a1 a2 a3 a4 = prim__reshape4dMlxStreamed a0 a1 a2 a3 a4 (streamTag s)
  primScatterAdd a0 a1 a2      = prim__scatterAddMlxStreamed a0 a1 a2 (streamTag s)
  primSelect a0 a1 a2          = prim__selectMlxStreamed a0 a1 a2 (streamTag s)
  primSqueeze a0 a1            = prim__squeezeMlxStreamed a0 a1 (streamTag s)
  primStack a0 a1 a2           = prim__stackMlxStreamed a0 a1 a2 (streamTag s)
  primSum a0                   = prim__sumMlxStreamed a0 (streamTag s)
  primSumDim a0 a1 a2          = prim__sumDimMlxStreamed a0 a1 a2 (streamTag s)
  primTensorMax a0             = prim__tensorMaxMlxStreamed a0 (streamTag s)
  primTensorMin a0             = prim__tensorMinMlxStreamed a0 (streamTag s)
  primTranspose2d a0           = prim__transpose2dMlxStreamed a0 (streamTag s)
  primTransposeLast2 a0        = prim__transposeLast2MlxStreamed a0 (streamTag s)
  primUnsqueeze a0 a1          = prim__unsqueezeMlxStreamed a0 a1 (streamTag s)
  primView1d a0 a1             = prim__view1dMlxStreamed a0 a1 (streamTag s)
  primView2d a0 a1 a2          = prim__view2dMlxStreamed a0 a1 a2 (streamTag s)
  -- <<< END GENERATED <<<

----------------------------------------------------------------------
-- NN-slice FFI bindings (mlx-suffixed)
----------------------------------------------------------------------

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_gelu_mlx_streamed\" (void* int) void*) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__geluMlxStreamed : AnyPtr -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_leaky_relu_mlx_streamed\" (void* double int) void*) (vector-ref a0 2) a1 a2))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__leakyReluMlxStreamed : AnyPtr -> Double -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_silu_mlx_streamed\" (void* int) void*) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__siluMlxStreamed : AnyPtr -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_softplus_mlx_streamed\" (void* int) void*) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__softplusMlxStreamed : AnyPtr -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_softmax_mlx_streamed\" (void* int int) void*) (vector-ref a0 2) a1 a2))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__softmaxMlxStreamed : AnyPtr -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_log_softmax_mlx_streamed\" (void* int int) void*) (vector-ref a0 2) a1 a2))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__logSoftmaxMlxStreamed : AnyPtr -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_softmax_2d_mlx_streamed\" (void* int) void*) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__softmax2dMlxStreamed : AnyPtr -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_log_softmax_2d_mlx_streamed\" (void* int) void*) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__logSoftmax2dMlxStreamed : AnyPtr -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_softmax_3d_mlx_streamed\" (void* int) void*) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__softmax3dMlxStreamed : AnyPtr -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3)  (let ((raw_r ((foreign-procedure \"tensor_masked_fill_mlx_streamed\" (void* void* double int) void*) (vector-ref a0 2) (vector-ref a1 2) a2 a3))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__maskedFillMlxStreamed : AnyPtr -> AnyPtr -> Double -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_expand_mask_mlx_streamed\" (void* int int) void*) (vector-ref a0 2) a1 a2))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__expandMaskMlxStreamed : AnyPtr -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4)  (let ((raw_r ((foreign-procedure \"tensor_layer_norm_2d_mlx_streamed\" (void* void* void* double int) void*) (vector-ref a0 2) (vector-ref a1 2) (vector-ref a2 2) a3 a4))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__layerNorm2dMlxStreamed : AnyPtr -> AnyPtr -> AnyPtr -> Double -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10)  (let ((raw_r ((foreign-procedure \"tensor_batch_norm_mlx_streamed\" (void* void* void* void* void* int int int double double int) void*) (vector-ref a0 2) (vector-ref a1 2) (vector-ref a2 2) (vector-ref a3 2) (vector-ref a4 2) a5 a6 a7 a8 a9 a10))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__batchNormMlxStreamed : AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr -> Int -> Int -> Int -> Double -> Double -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4)  (let ((raw_r ((foreign-procedure \"tensor_dropout_mlx_streamed\" (void* double int int int) void*) (vector-ref a0 2) a1 a2 a3 a4))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__dropoutMlxStreamed : AnyPtr -> Double -> Int -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4)  (let ((raw_r ((foreign-procedure \"tensor_embedding_mlx_streamed\" (void* void* int int int) void*) (vector-ref a0 2) (vector-ref a1 2) a2 a3 a4))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__embeddingMlxStreamed : AnyPtr -> AnyPtr -> Int -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4)  (let ((raw_r ((foreign-procedure \"tensor_embedding_2d_mlx_streamed\" (void* void* int int int) void*) (vector-ref a0 2) (vector-ref a1 2) a2 a3 a4))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__embedding2dMlxStreamed : AnyPtr -> AnyPtr -> Int -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3)  (let ((raw_r ((foreign-procedure \"tensor_cosine_similarity_mlx_streamed\" (void* void* int int) void*) (vector-ref a0 2) (vector-ref a1 2) a2 a3))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__cosineSimilarityMlxStreamed : AnyPtr -> AnyPtr -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4 a5)  (let ((raw_r ((foreign-procedure \"tensor_cross_attention_mlx_streamed\" (void* void* void* void* double int) void*) (vector-ref a0 2) (vector-ref a1 2) (vector-ref a2 2) (vector-ref a3 2) a4 a5))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__crossAttentionMlxStreamed : AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr -> Double -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_bce_with_logits_mlx_streamed\" (void* void* int) void*) (vector-ref a0 2) (vector-ref a1 2) a2))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__bceWithLogitsMlxStreamed : AnyPtr -> AnyPtr -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4)  (let ((raw_r ((foreign-procedure \"tensor_gru_cell_mlx_streamed\" (void* void* void* int int) void*) (vector-ref a0 2) (vector-ref a1 2) (vector-ref a2 2) a3 a4))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__gruCellMlxStreamed : AnyPtr -> AnyPtr -> AnyPtr -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3)  ((foreign-procedure \"tensor_lstm_gates_pair_mlx_streamed\" (void* void* int int) void*) (vector-ref a0 2) (vector-ref a1 2) a2 a3))"
prim__lstmGatesPairMlxStreamed : AnyPtr -> AnyPtr -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_pair_first_mlx_streamed\" (void* int) void*) a0 a1))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__pairFirstMlxStreamed : AnyPtr -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_pair_second_mlx_streamed\" (void* int) void*) a0 a1))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__pairSecondMlxStreamed : AnyPtr -> Int -> AnyPtr

-- Fused inference ops (used by `UserExecutorNN` below; FFI decls
-- moved up from the legacy `Training` slice region so they
-- precede their first use in the NN instance.)
%foreign "scheme:(lambda (a0 a1 a2 a3 a4 a5 a6)  (when (not (top-level-bound? 'idris-ffi-tensor-sdpa-2d-mlx)) (set-top-level-value! 'idris-ffi-tensor-sdpa-2d-mlx (foreign-procedure \"tensor_sdpa_2d_mlx\" (void* void* void* int int int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-mlx)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-mlx (foreign-procedure \"tensor_retain_handle_mlx\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-sdpa-2d-mlx) (vector-ref a0 2) (vector-ref a1 2) (vector-ref a2 2) a3 a4 a5 a6))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-mlx) raw_r) wr)))"
prim__sdpa2dMlx : AnyPtr -> AnyPtr -> AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2)  (when (not (top-level-bound? 'idris-ffi-tensor-rms-norm-2d-mlx)) (set-top-level-value! 'idris-ffi-tensor-rms-norm-2d-mlx (foreign-procedure \"tensor_rms_norm_2d_mlx\" (void* void* double) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-mlx)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-mlx (foreign-procedure \"tensor_retain_handle_mlx\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-rms-norm-2d-mlx) (vector-ref a0 2) (vector-ref a1 2) a2))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-mlx) raw_r) wr)))"
prim__rmsNorm2dMlx : AnyPtr -> AnyPtr -> Double -> AnyPtr
%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-swiglu-2d-mlx)) (set-top-level-value! 'idris-ffi-tensor-swiglu-2d-mlx (foreign-procedure \"tensor_swiglu_2d_mlx\" (void* void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-mlx)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-mlx (foreign-procedure \"tensor_retain_handle_mlx\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-swiglu-2d-mlx) (vector-ref a0 2) (vector-ref a1 2)))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-mlx) raw_r) wr)))"
prim__swiGlu2dMlx : AnyPtr -> AnyPtr -> AnyPtr

public export
{s : MlxStream} -> UserExecutorNN (MlxExecutor s) where
  -- >>> GENERATED FROM ffi_manifest.py — gen-executor-instances.py >>>
  primBatchNorm a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 = prim__batchNormMlxStreamed a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 (streamTag s)
  primBceWithLogits a0 a1                     = prim__bceWithLogitsMlxStreamed a0 a1 (streamTag s)
  primCosineSimilarity a0 a1 a2               = prim__cosineSimilarityMlxStreamed a0 a1 a2 (streamTag s)
  primDropout a0 a1 a2 a3                     = prim__dropoutMlxStreamed a0 a1 a2 a3 (streamTag s)
  primEmbedding a0 a1 a2 a3                   = prim__embeddingMlxStreamed a0 a1 a2 a3 (streamTag s)
  primEmbedding2d a0 a1 a2 a3                 = prim__embedding2dMlxStreamed a0 a1 a2 a3 (streamTag s)
  primExpandMask a0 a1                        = prim__expandMaskMlxStreamed a0 a1 (streamTag s)
  primGelu a0                                 = prim__geluMlxStreamed a0 (streamTag s)
  primGruCell a0 a1 a2 a3                     = prim__gruCellMlxStreamed a0 a1 a2 a3 (streamTag s)
  primLayerNorm2d a0 a1 a2 a3                 = prim__layerNorm2dMlxStreamed a0 a1 a2 a3 (streamTag s)
  primLeakyRelu a0 a1                         = prim__leakyReluMlxStreamed a0 a1 (streamTag s)
  primLogSoftmax a0 a1                        = prim__logSoftmaxMlxStreamed a0 a1 (streamTag s)
  primLogSoftmax2d a0                         = prim__logSoftmax2dMlxStreamed a0 (streamTag s)
  primLstmGatesPair a0 a1 a2                  = prim__lstmGatesPairMlxStreamed a0 a1 a2 (streamTag s)
  primMaskedFill a0 a1 a2                     = prim__maskedFillMlxStreamed a0 a1 a2 (streamTag s)
  primPairFirst a0                            = prim__pairFirstMlxStreamed a0 (streamTag s)
  primPairSecond a0                           = prim__pairSecondMlxStreamed a0 (streamTag s)
  primSilu a0                                 = prim__siluMlxStreamed a0 (streamTag s)
  primSoftmax a0 a1                           = prim__softmaxMlxStreamed a0 a1 (streamTag s)
  primSoftmax2d a0                            = prim__softmax2dMlxStreamed a0 (streamTag s)
  primSoftmax3d a0                            = prim__softmax3dMlxStreamed a0 (streamTag s)
  primSoftplus a0                             = prim__softplusMlxStreamed a0 (streamTag s)
  -- <<< END GENERATED <<<

----------------------------------------------------------------------
-- Conv-slice FFI bindings (mlx-suffixed)
----------------------------------------------------------------------

%foreign "scheme:(lambda (a0 a1 a2 a3 a4 a5)  (let ((raw_r ((foreign-procedure \"tensor_conv1d_mlx_streamed\" (void* void* void* int int int) void*) (vector-ref a0 2) (vector-ref a1 2) (vector-ref a2 2) a3 a4 a5))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__conv1dMlxStreamed : AnyPtr -> AnyPtr -> AnyPtr -> Int -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_conv1d_circular_mlx_streamed\" (void* void* int) void*) (vector-ref a0 2) (vector-ref a1 2) a2))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__conv1dCircularMlxStreamed : AnyPtr -> AnyPtr -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3)  (let ((raw_r ((foreign-procedure \"tensor_avg_pool1d_mlx_streamed\" (void* int int int) void*) (vector-ref a0 2) a1 a2 a3))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__avgPool1dMlxStreamed : AnyPtr -> Int -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3)  (let ((raw_r ((foreign-procedure \"tensor_max_pool1d_mlx_streamed\" (void* int int int) void*) (vector-ref a0 2) a1 a2 a3))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__maxPool1dMlxStreamed : AnyPtr -> Int -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4 a5 a6 a7)  (let ((raw_r ((foreign-procedure \"tensor_conv2d_mlx_streamed\" (void* void* void* int int int int int) void*) (vector-ref a0 2) (vector-ref a1 2) (vector-ref a2 2) a3 a4 a5 a6 a7))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__conv2dMlxStreamed : AnyPtr -> AnyPtr -> AnyPtr -> Int -> Int -> Int -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4 a5 a6 a7)  (let ((raw_r ((foreign-procedure \"tensor_conv2d_batched_mlx_streamed\" (void* void* void* int int int int int) void*) (vector-ref a0 2) (vector-ref a1 2) (vector-ref a2 2) a3 a4 a5 a6 a7))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__conv2dBatchedMlxStreamed : AnyPtr -> AnyPtr -> AnyPtr -> Int -> Int -> Int -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4 a5)  (let ((raw_r ((foreign-procedure \"tensor_avg_pool2d_mlx_streamed\" (void* int int int int int) void*) (vector-ref a0 2) a1 a2 a3 a4 a5))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__avgPool2dMlxStreamed : AnyPtr -> Int -> Int -> Int -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4 a5)  (let ((raw_r ((foreign-procedure \"tensor_max_pool2d_mlx_streamed\" (void* int int int int int) void*) (vector-ref a0 2) a1 a2 a3 a4 a5))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__maxPool2dMlxStreamed : AnyPtr -> Int -> Int -> Int -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4 a5)  (let ((raw_r ((foreign-procedure \"tensor_max_pool2d_batched_mlx_streamed\" (void* int int int int int) void*) (vector-ref a0 2) a1 a2 a3 a4 a5))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__maxPool2dBatchedMlxStreamed : AnyPtr -> Int -> Int -> Int -> Int -> Int -> AnyPtr

public export
{s : MlxStream} -> UserExecutorConv (MlxExecutor s) where
  -- >>> GENERATED FROM ffi_manifest.py — gen-executor-instances.py >>>
  primAvgPool1d a0 a1 a2                 = prim__avgPool1dMlxStreamed a0 a1 a2 (streamTag s)
  primAvgPool2d a0 a1 a2 a3 a4           = prim__avgPool2dMlxStreamed a0 a1 a2 a3 a4 (streamTag s)
  primConv1d a0 a1 a2 a3 a4              = prim__conv1dMlxStreamed a0 a1 a2 a3 a4 (streamTag s)
  primConv1dCircular a0 a1               = prim__conv1dCircularMlxStreamed a0 a1 (streamTag s)
  primConv2d a0 a1 a2 a3 a4 a5 a6        = prim__conv2dMlxStreamed a0 a1 a2 a3 a4 a5 a6 (streamTag s)
  primConv2dBatched a0 a1 a2 a3 a4 a5 a6 = prim__conv2dBatchedMlxStreamed a0 a1 a2 a3 a4 a5 a6 (streamTag s)
  primMaxPool1d a0 a1 a2                 = prim__maxPool1dMlxStreamed a0 a1 a2 (streamTag s)
  primMaxPool2d a0 a1 a2 a3 a4           = prim__maxPool2dMlxStreamed a0 a1 a2 a3 a4 (streamTag s)
  primMaxPool2dBatched a0 a1 a2 a3 a4    = prim__maxPool2dBatchedMlxStreamed a0 a1 a2 a3 a4 (streamTag s)
  -- <<< END GENERATED <<<
----------------------------------------------------------------------
-- Tape-slice FFI bindings (mlx-suffixed)
----------------------------------------------------------------------

%foreign "scheme:(lambda (a0)  (when (not (top-level-bound? 'idris-ffi-tensor-requires-grad-mlx)) (set-top-level-value! 'idris-ffi-tensor-requires-grad-mlx (foreign-procedure \"tensor_requires_grad_mlx\" (void*) int))) ((top-level-value 'idris-ffi-tensor-requires-grad-mlx) (vector-ref a0 2)))"
prim__requiresGradMlx : AnyPtr -> Int
%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-set-requires-grad-mlx)) (set-top-level-value! 'idris-ffi-tensor-set-requires-grad-mlx (foreign-procedure \"tensor_set_requires_grad_mlx\" (void* int) void))) ((top-level-value 'idris-ffi-tensor-set-requires-grad-mlx) (vector-ref a0 2) a1))"
prim__setRequiresGradMlx : AnyPtr -> Int -> PrimIO ()
%foreign "scheme:(lambda (a0)  (when (not (top-level-bound? 'idris-ffi-tensor-backward-mlx)) (set-top-level-value! 'idris-ffi-tensor-backward-mlx (foreign-procedure \"tensor_backward_mlx\" (void*) void))) ((top-level-value 'idris-ffi-tensor-backward-mlx) (vector-ref a0 2)))"
prim__backwardMlx : AnyPtr -> PrimIO ()
%foreign "C:tensor_no_grad_begin_mlx,libidrisml"
prim__noGradBeginMlx : PrimIO ()
%foreign "C:tensor_no_grad_end_mlx,libidrisml"
prim__noGradEndMlx : PrimIO ()
%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_detach_mlx_streamed\" (void* int) void*) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__detachMlxStreamed : AnyPtr -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_with_grad_mlx_streamed\" (void* int) void*) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__withGradMlxStreamed : AnyPtr -> Int -> AnyPtr
%foreign "scheme:(lambda (a0)  (when (not (top-level-bound? 'idris-ffi-tensor-dim-mlx)) (set-top-level-value! 'idris-ffi-tensor-dim-mlx (foreign-procedure \"tensor_dim_mlx\" (void*) int))) ((top-level-value 'idris-ffi-tensor-dim-mlx) (vector-ref a0 2)))"
prim__tensorDimMlx : AnyPtr -> Int
%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-size-mlx)) (set-top-level-value! 'idris-ffi-tensor-size-mlx (foreign-procedure \"tensor_size_mlx\" (void* int) int))) ((top-level-value 'idris-ffi-tensor-size-mlx) (vector-ref a0 2) a1))"
prim__tensorSizeAtMlx : AnyPtr -> Int -> Int
%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-param-register-return-mlx)) (set-top-level-value! 'idris-ffi-param-register-return-mlx (foreign-procedure \"param_register_return_mlx\" (string void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-mlx)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-mlx (foreign-procedure \"tensor_retain_handle_mlx\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-param-register-return-mlx) a0 (vector-ref a1 2)))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-mlx) raw_r) wr)))"
prim__paramRegisterMlx : String -> AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-param-register-buffer-return-mlx)) (set-top-level-value! 'idris-ffi-param-register-buffer-return-mlx (foreign-procedure \"param_register_buffer_return_mlx\" (string void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-mlx)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-mlx (foreign-procedure \"tensor_retain_handle_mlx\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-param-register-buffer-return-mlx) a0 (vector-ref a1 2)))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-mlx) raw_r) wr)))"
prim__paramRegisterBufferMlx : String -> AnyPtr -> AnyPtr
%foreign "C:param_is_buffer_mlx,libidrisml"
prim__paramIsBufferMlx : Int -> PrimIO Int
%foreign "C:polyak_blend_pair_mlx,libidrisml"
prim__polyakBlendPairMlx : Double -> String -> String -> PrimIO Int
%foreign "C:param_count_mlx,libidrisml"
prim__paramCountMlx : PrimIO Int
%foreign "C:param_name_mlx,libidrisml"
prim__paramNameMlx : Int -> PrimIO String
%foreign "C:param_grad_item_at_mlx,libidrisml"
prim__paramGradItemAtMlx : Int -> Int -> PrimIO Double
%foreign "C:param_zero_all_grads_mlx,libidrisml"
prim__paramZeroAllMlx : PrimIO ()
%foreign "C:param_erase_by_prefix_mlx,libidrisml"
prim__paramEraseByPrefixMlx : String -> PrimIO ()
%foreign "C:optimizer_create_sgd_mlx,libidrisml"
prim__optimizerCreateSgdMlx : Double -> AnyPtr
%foreign "C:optimizer_create_rmsprop_mlx,libidrisml"
prim__optimizerCreateRmspropMlx : Double -> Double -> Double -> Double -> Double -> AnyPtr
%foreign "C:optimizer_create_adam_mlx,libidrisml"
prim__optimizerCreateAdamMlx : Double -> Double -> Double -> Double -> AnyPtr
%foreign "C:optimizer_create_adamw_mlx,libidrisml"
prim__optimizerCreateAdamWMlx : Double -> Double -> Double -> Double -> Double -> AnyPtr
%foreign "C:optimizer_set_lr_mlx,libidrisml"
prim__optimizerSetLrMlx : AnyPtr -> Double -> PrimIO ()
%foreign "C:optimizer_set_param_lr_mlx,libidrisml"
prim__optimizerSetParamLrMlx : AnyPtr -> String -> Double -> PrimIO ()
%foreign "C:optimizer_own_param_mlx,libidrisml"
prim__optimizerOwnParamMlx : AnyPtr -> String -> PrimIO ()
%foreign "scheme:(lambda (a0 a1 a2 a3 a4)  (when (not (top-level-bound? 'idris-ffi-native-train-step-mlx)) (set-top-level-value! 'idris-ffi-native-train-step-mlx (foreign-procedure \"native_train_step_mlx\" (void* int double void* double) double))) ((top-level-value 'idris-ffi-native-train-step-mlx) a0 a1 a2 (vector-ref a3 2) a4))"
prim__nativeTrainStepMlx : AnyPtr -> Int -> Double -> AnyPtr -> Double -> Double
%foreign "scheme:(lambda (a0 a1 a2 a3 a4 a5)  (when (not (top-level-bound? 'idris-ffi-native-train-step-scaled-mlx)) (set-top-level-value! 'idris-ffi-native-train-step-scaled-mlx (foreign-procedure \"native_train_step_scaled_mlx\" (void* int double void* double double) double))) ((top-level-value 'idris-ffi-native-train-step-scaled-mlx) a0 a1 a2 (vector-ref a3 2) a4 a5))"
prim__nativeTrainStepScaledMlx : AnyPtr -> Int -> Double -> AnyPtr -> Double -> Double -> Double
%foreign "C:param_save_mlx,libidrisml"
prim__paramSaveMlx : String -> PrimIO Int
%foreign "C:param_save_by_name_mlx,libidrisml"
prim__paramSaveByNameMlx : String -> String -> Int -> PrimIO Int
%foreign "C:param_save_by_name_renamed_mlx,libidrisml"
prim__paramSaveByNameRenamedMlx : String -> String -> String -> Int -> PrimIO Int
%foreign "C:param_load_mlx,libidrisml"
prim__paramLoadMlx : String -> PrimIO Int
%foreign "C:param_load_with_policy_mlx,libidrisml"
prim__paramLoadWithPolicyMlx : String -> Int -> PrimIO Int
%foreign "C:param_load_with_prefix_mlx,libidrisml"
prim__paramLoadWithPrefixMlx : String -> Int -> String -> PrimIO Int
%foreign "C:param_load_renamed_mlx,libidrisml"
prim__paramLoadRenamedMlx : String -> Int -> String -> String -> Int -> PrimIO Int
%foreign "C:optimizer_save_mlx,libidrisml"
prim__optimizerSaveMlx : AnyPtr -> String -> PrimIO Int
%foreign "C:optimizer_load_mlx,libidrisml"
prim__optimizerLoadMlx : AnyPtr -> String -> PrimIO Int
%foreign "C:backend_profile_reset_mlx,libidrisml"
prim__profileResetMlx : PrimIO ()
%foreign "C:backend_profile_report_mlx,libidrisml"
prim__profileReportMlx : PrimIO ()
%foreign "C:tensor_epoch_begin_mlx,libidrisml"
prim__epochBeginMlx : PrimIO ()
%foreign "C:tensor_epoch_end_mlx,libidrisml"
prim__epochEndMlx : PrimIO ()
%foreign "C:backend_release_all_persistent_mlx,libidrisml"
prim__releaseAllPersistentMlx : PrimIO ()
%foreign "C:backend_reset_for_eval_mlx,libidrisml"
prim__resetForEvalMlx : PrimIO ()
%foreign "C:tensor_live_count_mlx,libidrisml"
prim__liveCountMlx : PrimIO Int
%foreign "C:tensor_peak_live_count_mlx,libidrisml"
prim__peakLiveCountMlx : PrimIO Int
%foreign "C:tensor_perf_reset_mlx,libidrisml"
prim__perfResetMlx : PrimIO ()
%foreign "C:tensor_perf_op_count_mlx,libidrisml"
prim__perfOpCountMlx : PrimIO Int

%foreign "scheme:(lambda (a0 a1 a2 a3) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-tensor-create-scalar-streamed-mlx)) (set-top-level-value! 'idris-ffi-tensor-create-scalar-streamed-mlx (foreign-procedure \"tensor_create_scalar_streamed_mlx\" (double int int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-mlx)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-mlx (foreign-procedure \"tensor_retain_handle_mlx\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-create-scalar-streamed-mlx) a0 a1 a2 a3))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-mlx) raw_r) wr)))"
prim__createScalarStreamedMlx : Double -> Int -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4 a5) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-tensor-create-streamed-mlx)) (set-top-level-value! 'idris-ffi-tensor-create-streamed-mlx (foreign-procedure \"tensor_create_streamed_mlx\" (void* void* int int int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-mlx)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-mlx (foreign-procedure \"tensor_retain_handle_mlx\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-create-streamed-mlx) a0 a1 a2 a3 a4 a5))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-mlx) raw_r) wr)))"
prim__createStreamedMlx : AnyPtr -> AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-tensor-create-1d-streamed-mlx)) (set-top-level-value! 'idris-ffi-tensor-create-1d-streamed-mlx (foreign-procedure \"tensor_create_1d_streamed_mlx\" (int void* int int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-mlx)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-mlx (foreign-procedure \"tensor_retain_handle_mlx\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-create-1d-streamed-mlx) a0 a1 a2 a3 a4))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-mlx) raw_r) wr)))"
prim__create1dStreamedMlx : Int -> AnyPtr -> Int -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4 a5) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-tensor-create-2d-streamed-mlx)) (set-top-level-value! 'idris-ffi-tensor-create-2d-streamed-mlx (foreign-procedure \"tensor_create_2d_streamed_mlx\" (int int void* int int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-mlx)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-mlx (foreign-procedure \"tensor_retain_handle_mlx\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-create-2d-streamed-mlx) a0 a1 a2 a3 a4 a5))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-mlx) raw_r) wr)))"
prim__create2dStreamedMlx : Int -> Int -> AnyPtr -> Int -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-tensor-create-param-1d-streamed-mlx)) (set-top-level-value! 'idris-ffi-tensor-create-param-1d-streamed-mlx (foreign-procedure \"tensor_create_param_1d_streamed_mlx\" (int void* int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-mlx)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-mlx (foreign-procedure \"tensor_retain_handle_mlx\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-create-param-1d-streamed-mlx) a0 a1 a2 a3))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-mlx) raw_r) wr)))"
prim__createParam1dStreamedMlx : Int -> AnyPtr -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-tensor-create-param-2d-streamed-mlx)) (set-top-level-value! 'idris-ffi-tensor-create-param-2d-streamed-mlx (foreign-procedure \"tensor_create_param_2d_streamed_mlx\" (int int void* int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-mlx)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-mlx (foreign-procedure \"tensor_retain_handle_mlx\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-create-param-2d-streamed-mlx) a0 a1 a2 a3 a4))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-mlx) raw_r) wr)))"
prim__createParam2dStreamedMlx : Int -> Int -> AnyPtr -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4 a5) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-tensor-create-param-3d-streamed-mlx)) (set-top-level-value! 'idris-ffi-tensor-create-param-3d-streamed-mlx (foreign-procedure \"tensor_create_param_3d_streamed_mlx\" (int int int void* int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-mlx)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-mlx (foreign-procedure \"tensor_retain_handle_mlx\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-create-param-3d-streamed-mlx) a0 a1 a2 a3 a4 a5))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-mlx) raw_r) wr)))"
prim__createParam3dStreamedMlx : Int -> Int -> Int -> AnyPtr -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4 a5 a6) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-tensor-create-param-4d-streamed-mlx)) (set-top-level-value! 'idris-ffi-tensor-create-param-4d-streamed-mlx (foreign-procedure \"tensor_create_param_4d_streamed_mlx\" (int int int int void* int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-mlx)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-mlx (foreign-procedure \"tensor_retain_handle_mlx\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-create-param-4d-streamed-mlx) a0 a1 a2 a3 a4 a5 a6))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-mlx) raw_r) wr)))"
prim__createParam4dStreamedMlx : Int -> Int -> Int -> Int -> AnyPtr -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-tensor-create-state-1d-streamed-mlx)) (set-top-level-value! 'idris-ffi-tensor-create-state-1d-streamed-mlx (foreign-procedure \"tensor_create_state_1d_streamed_mlx\" (int void* int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-mlx)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-mlx (foreign-procedure \"tensor_retain_handle_mlx\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-create-state-1d-streamed-mlx) a0 a1 a2 a3))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-mlx) raw_r) wr)))"
prim__createState1dStreamedMlx : Int -> AnyPtr -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-tensor-create-state-2d-streamed-mlx)) (set-top-level-value! 'idris-ffi-tensor-create-state-2d-streamed-mlx (foreign-procedure \"tensor_create_state_2d_streamed_mlx\" (int int void* int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-mlx)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-mlx (foreign-procedure \"tensor_retain_handle_mlx\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-create-state-2d-streamed-mlx) a0 a1 a2 a3 a4))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-mlx) raw_r) wr)))"
prim__createState2dStreamedMlx : Int -> Int -> AnyPtr -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-tensor-cast-dtype-streamed-mlx)) (set-top-level-value! 'idris-ffi-tensor-cast-dtype-streamed-mlx (foreign-procedure \"tensor_cast_dtype_streamed_mlx\" (void* int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-mlx)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-mlx (foreign-procedure \"tensor_retain_handle_mlx\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-cast-dtype-streamed-mlx) (vector-ref a0 2) a1 a2))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-mlx) raw_r) wr)))"
prim__castStreamedMlx : AnyPtr -> Int -> Int -> AnyPtr

-- Fused param create + in-place init. Mlx's C-side port slots stay
-- nullptr until Phase 7 lands the impl (mx::random::normal etc.); the
-- shared trampoline in `dtype_streamed.c` aborts loud if called. See
-- the matching block in Executor/Tape.idr for the rationale.
%foreign "scheme:(lambda (a0 a1 a2 a3 a4) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-tensor-create-param-1d-normal-streamed-mlx)) (set-top-level-value! 'idris-ffi-tensor-create-param-1d-normal-streamed-mlx (foreign-procedure \"tensor_create_param_1d_normal_streamed_mlx\" (int double double int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-mlx)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-mlx (foreign-procedure \"tensor_retain_handle_mlx\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-create-param-1d-normal-streamed-mlx) a0 a1 a2 a3 a4))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-mlx) raw_r) wr)))"
prim__createParam1dNormalStreamedMlx : Int -> Double -> Double -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4 a5) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-tensor-create-param-2d-normal-streamed-mlx)) (set-top-level-value! 'idris-ffi-tensor-create-param-2d-normal-streamed-mlx (foreign-procedure \"tensor_create_param_2d_normal_streamed_mlx\" (int int double double int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-mlx)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-mlx (foreign-procedure \"tensor_retain_handle_mlx\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-create-param-2d-normal-streamed-mlx) a0 a1 a2 a3 a4 a5))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-mlx) raw_r) wr)))"
prim__createParam2dNormalStreamedMlx : Int -> Int -> Double -> Double -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4 a5 a6) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-tensor-create-param-3d-normal-streamed-mlx)) (set-top-level-value! 'idris-ffi-tensor-create-param-3d-normal-streamed-mlx (foreign-procedure \"tensor_create_param_3d_normal_streamed_mlx\" (int int int double double int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-mlx)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-mlx (foreign-procedure \"tensor_retain_handle_mlx\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-create-param-3d-normal-streamed-mlx) a0 a1 a2 a3 a4 a5 a6))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-mlx) raw_r) wr)))"
prim__createParam3dNormalStreamedMlx : Int -> Int -> Int -> Double -> Double -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4 a5 a6 a7) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-tensor-create-param-4d-normal-streamed-mlx)) (set-top-level-value! 'idris-ffi-tensor-create-param-4d-normal-streamed-mlx (foreign-procedure \"tensor_create_param_4d_normal_streamed_mlx\" (int int int int double double int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-mlx)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-mlx (foreign-procedure \"tensor_retain_handle_mlx\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-create-param-4d-normal-streamed-mlx) a0 a1 a2 a3 a4 a5 a6 a7))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-mlx) raw_r) wr)))"
prim__createParam4dNormalStreamedMlx : Int -> Int -> Int -> Int -> Double -> Double -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-tensor-create-param-1d-const-streamed-mlx)) (set-top-level-value! 'idris-ffi-tensor-create-param-1d-const-streamed-mlx (foreign-procedure \"tensor_create_param_1d_const_streamed_mlx\" (int double int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-mlx)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-mlx (foreign-procedure \"tensor_retain_handle_mlx\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-create-param-1d-const-streamed-mlx) a0 a1 a2 a3))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-mlx) raw_r) wr)))"
prim__createParam1dConstStreamedMlx : Int -> Double -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-tensor-create-param-2d-const-streamed-mlx)) (set-top-level-value! 'idris-ffi-tensor-create-param-2d-const-streamed-mlx (foreign-procedure \"tensor_create_param_2d_const_streamed_mlx\" (int int double int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-mlx)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-mlx (foreign-procedure \"tensor_retain_handle_mlx\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-create-param-2d-const-streamed-mlx) a0 a1 a2 a3 a4))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-mlx) raw_r) wr)))"
prim__createParam2dConstStreamedMlx : Int -> Int -> Double -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4 a5) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-tensor-create-param-3d-const-streamed-mlx)) (set-top-level-value! 'idris-ffi-tensor-create-param-3d-const-streamed-mlx (foreign-procedure \"tensor_create_param_3d_const_streamed_mlx\" (int int int double int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-mlx)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-mlx (foreign-procedure \"tensor_retain_handle_mlx\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-create-param-3d-const-streamed-mlx) a0 a1 a2 a3 a4 a5))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-mlx) raw_r) wr)))"
prim__createParam3dConstStreamedMlx : Int -> Int -> Int -> Double -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4 a5 a6) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-tensor-create-param-4d-const-streamed-mlx)) (set-top-level-value! 'idris-ffi-tensor-create-param-4d-const-streamed-mlx (foreign-procedure \"tensor_create_param_4d_const_streamed_mlx\" (int int int int double int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-mlx)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-mlx (foreign-procedure \"tensor_retain_handle_mlx\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-create-param-4d-const-streamed-mlx) a0 a1 a2 a3 a4 a5 a6))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-mlx) raw_r) wr)))"
prim__createParam4dConstStreamedMlx : Int -> Int -> Int -> Int -> Double -> Int -> Int -> AnyPtr
%foreign "C:tensor_set_init_seed_streamed_mlx,libidrisml"
prim__setInitSeedStreamedMlx : Bits64 -> Int -> PrimIO ()

public export
{s : MlxStream} -> UserExecutorOptimizations (MlxExecutor s) where
  -- >>> GENERATED FROM ffi_manifest.py — gen-executor-instances.py >>>
  primCreateParam1dConstStreamed    = prim__createParam1dConstStreamedMlx
  primCreateParam1dNormalStreamed   = prim__createParam1dNormalStreamedMlx
  primCreateParam2dConstStreamed    = prim__createParam2dConstStreamedMlx
  primCreateParam2dNormalStreamed   = prim__createParam2dNormalStreamedMlx
  primCreateParam3dConstStreamed    = prim__createParam3dConstStreamedMlx
  primCreateParam3dNormalStreamed   = prim__createParam3dNormalStreamedMlx
  primCreateParam4dConstStreamed    = prim__createParam4dConstStreamedMlx
  primCreateParam4dNormalStreamed   = prim__createParam4dNormalStreamedMlx
  primCrossAttention a0 a1 a2 a3 a4 = prim__crossAttentionMlxStreamed a0 a1 a2 a3 a4 (streamTag s)
  primPolyakBlendPair               = prim__polyakBlendPairMlx
  primRmsNorm2d                     = prim__rmsNorm2dMlx
  primSdpa2d                        = prim__sdpa2dMlx
  primSwiGlu2d                      = prim__swiGlu2dMlx
  primTile2d a0 a1 a2               = prim__tile2dMlxStreamed a0 a1 a2 (streamTag s)
  -- <<< END GENERATED <<<

public export
{s : MlxStream} -> UserExecutorAutograd (MlxExecutor s) where
  -- >>> GENERATED FROM ffi_manifest.py — gen-executor-instances.py >>>
  primBackward        = prim__backwardMlx
  primDetach a0       = prim__detachMlxStreamed a0 (streamTag s)
  primNoGradBegin     = prim__noGradBeginMlx
  primNoGradEnd       = prim__noGradEndMlx
  primRequiresGrad    = prim__requiresGradMlx
  primSetRequiresGrad = prim__setRequiresGradMlx
  primWithGrad a0     = prim__withGradMlxStreamed a0 (streamTag s)
  -- <<< END GENERATED <<<

public export
{s : MlxStream} -> UserExecutorParamRegistry (MlxExecutor s) where
  -- >>> GENERATED FROM ffi_manifest.py — gen-executor-instances.py >>>
  primParamCount          = prim__paramCountMlx
  primParamEraseByPrefix  = prim__paramEraseByPrefixMlx
  primParamGradItemAt     = prim__paramGradItemAtMlx
  primParamIsBuffer       = prim__paramIsBufferMlx
  primParamName           = prim__paramNameMlx
  primParamRegister       = prim__paramRegisterMlx
  primParamRegisterBuffer = prim__paramRegisterBufferMlx
  primParamZeroAll        = prim__paramZeroAllMlx
  -- <<< END GENERATED <<<

public export
{s : MlxStream} -> UserExecutorOptimizer (MlxExecutor s) where
  -- >>> GENERATED FROM ffi_manifest.py — gen-executor-instances.py >>>
  primNativeTrainStep        = prim__nativeTrainStepMlx
  primNativeTrainStepScaled  = prim__nativeTrainStepScaledMlx
  primOptimizerCreateAdam    = prim__optimizerCreateAdamMlx
  primOptimizerCreateAdamW   = prim__optimizerCreateAdamWMlx
  primOptimizerCreateRmsprop = prim__optimizerCreateRmspropMlx
  primOptimizerCreateSgd     = prim__optimizerCreateSgdMlx
  primOptimizerOwnParam      = prim__optimizerOwnParamMlx
  primOptimizerSetLr         = prim__optimizerSetLrMlx
  primOptimizerSetParamLr    = prim__optimizerSetParamLrMlx
  -- <<< END GENERATED <<<

public export
{s : MlxStream} -> UserExecutorSerialize (MlxExecutor s) where
  -- >>> GENERATED FROM ffi_manifest.py — gen-executor-instances.py >>>
  primOptimizerLoad          = prim__optimizerLoadMlx
  primOptimizerSave          = prim__optimizerSaveMlx
  primParamLoad              = prim__paramLoadMlx
  primParamLoadRenamed       = prim__paramLoadRenamedMlx
  primParamLoadWithPolicy    = prim__paramLoadWithPolicyMlx
  primParamLoadWithPrefix    = prim__paramLoadWithPrefixMlx
  primParamSave              = prim__paramSaveMlx
  primParamSaveByName        = prim__paramSaveByNameMlx
  primParamSaveByNameRenamed = prim__paramSaveByNameRenamedMlx
  -- <<< END GENERATED <<<

public export
{s : MlxStream} -> UserExecutorMemoryHygiene (MlxExecutor s) where
  -- >>> GENERATED FROM ffi_manifest.py — gen-executor-instances.py >>>
  primEpochBegin           = prim__epochBeginMlx
  primEpochEnd             = prim__epochEndMlx
  primReleaseAllPersistent = prim__releaseAllPersistentMlx
  primResetForEval         = prim__resetForEvalMlx
  -- <<< END GENERATED <<<

public export
{s : MlxStream} -> UserExecutorDiagnostics (MlxExecutor s) where
  -- >>> GENERATED FROM ffi_manifest.py — gen-executor-instances.py >>>
  primLiveCount     = prim__liveCountMlx
  primPeakLiveCount = prim__peakLiveCountMlx
  primPerfOpCount   = prim__perfOpCountMlx
  -- <<< END GENERATED <<<

public export
{s : MlxStream} -> UserExecutorProfiling (MlxExecutor s) where
  -- >>> GENERATED FROM ffi_manifest.py — gen-executor-instances.py >>>
  primPerfReset     = prim__perfResetMlx
  primProfileReport = prim__profileReportMlx
  primProfileReset  = prim__profileResetMlx
  -- <<< END GENERATED <<<

public export
{s : MlxStream} -> UserExecutorTensorCreate (MlxExecutor s) where
  -- >>> GENERATED FROM ffi_manifest.py — gen-executor-instances.py >>>
  primCastStreamed          = prim__castStreamedMlx
  primCreate1dStreamed      = prim__create1dStreamedMlx
  primCreate2dStreamed      = prim__create2dStreamedMlx
  primCreateParam1dStreamed = prim__createParam1dStreamedMlx
  primCreateParam2dStreamed = prim__createParam2dStreamedMlx
  primCreateParam3dStreamed = prim__createParam3dStreamedMlx
  primCreateParam4dStreamed = prim__createParam4dStreamedMlx
  primCreateScalarStreamed  = prim__createScalarStreamedMlx
  primCreateState1dStreamed = prim__createState1dStreamedMlx
  primCreateState2dStreamed = prim__createState2dStreamedMlx
  primCreateStreamed        = prim__createStreamedMlx
  primItem2d                = prim__item2dMlx
  primOneHot                = prim__oneHotMlx
  primSetInitSeedStreamed   = prim__setInitSeedStreamedMlx
  primTensorDim             = prim__tensorDimMlx
  primTensorSizeAt          = prim__tensorSizeAtMlx
  -- <<< END GENERATED <<<

public export
{s : MlxStream} -> UserExecutorTraining (MlxExecutor s) where
  -- >>> GENERATED FROM ffi_manifest.py — gen-executor-instances.py >>>
  -- <<< END GENERATED <<<
----------------------------------------------------------------------
-- Compatible (device, dtype) instances
--
-- `MlxCpu` (`MlxExecutor MCpu`) supports F32, F64, BF16, and F16. mlx's
-- CPU stream has fp64 kernel coverage (see mlx/backend/cpu/{unary,
-- binary}.h `case float64` branches); bfloat16 + float16 storage
-- work under `mx::bfloat16` and `mx::float16` (mlx ships scalar
-- `bfloat16_t` from arm_bf16.h on Apple Silicon, or `_MLX_BFloat16`
-- struct elsewhere; same shape for `float16_t`).
--
-- The 2026-05-18 mlx-runtime-fp64 work routes `RuntimeDType F64` to
-- `tensor_create_*_f64` symbols that allocate `mx::float64`; the
-- corresponding 2026-05-31 mlx-bf16 + mlx-f16 work added the bf16/f16
-- siblings. The type-level claim is honored at allocation.
-- Constant-pool audit (2026-06-06): the 28 `mx::float32` hits across
-- `backend_mlx/` are all already correctly mixed-dtype-aware. The
-- dropout F32 chain casts to operand dtype before multiply
-- (`dropout.cpp:23`); the optimizer's `opt_dtype = mx::float32`
-- default is overridden by the dtype-discovery loop at
-- `optimizer.cpp:292-298`; `kF32_*` singletons in `precision.h:35-37`
-- are only consumed by dropout + cast-replay (both correct); the F32
-- stage in BF16/F16 narrowing constructors (`precision.h:79,92,104`)
-- is intentional. The +62% mlx-gpu BF16 vs F32 wall on HfLlama-1B
-- `runGenerate` (perf-changes.md 2026-05-31) therefore must come
-- from somewhere outside the constant pool — `runGenerate` is
-- inference, which bypasses dropout entirely (`dropout.cpp:17`).
-- The likely real culprit is kernel-level mlx-Metal BF16
-- performance vs F32 on Apple Silicon; tracked in the reframed
-- TODO row "Audit mlx fused-op + constant pool dtype handling".
--
-- `MlxGpu` (`MlxExecutor MGpu`) supports F32, BF16, and F16. Metal GPUs
-- dropped float64 support in mlx 0.31 (`Compatible (MlxExecutor MGpu) F64`
-- stays deliberately missing — the PyTorch runtime "Float64 not
-- supported on Metal" error lifted to compile time), but M3+ has
-- hardware bfloat16 + float16 in Metal so the BF16 and F16 instances
-- are admissible.
----------------------------------------------------------------------

public export
Compatible (MlxExecutor MCpu) F64 where

public export
Compatible (MlxExecutor MCpu) F32 where

public export
Compatible (MlxExecutor MCpu) BF16 where

public export
Compatible (MlxExecutor MCpu) F16 where

public export
Compatible (MlxExecutor MGpu) F32 where

public export
Compatible (MlxExecutor MGpu) BF16 where

public export
Compatible (MlxExecutor MGpu) F16 where

-- I32 instances: bulk creation, cast, and readback all wired through
-- `backend_mlx/training/dtype_dispatch.cpp` (dtag=10) and the
-- precision.h `mx_to_doubles` / `mx_read_double` / `mx_i32_from_doubles`
-- helpers. randn-init for I32 params is deliberately not wired
-- (semantically meaningless); construct I32 tensors via the bulk path.
public export
Compatible (MlxExecutor MCpu) I32 where
public export
Compatible (MlxExecutor MGpu) I32 where

-- DELIBERATELY NO `Compatible (MlxExecutor MGpu) F64` instance — Metal
-- has no fp64. (Other Int* + bool stay unwired on both streams.)

-- Sub-byte quantization dtypes (#411 BitNet b1.58). CPU stream only —
-- mlx's Metal sub-byte support requires custom kernels that arrive in
-- B3. The CPU instance is enough to validate the typeclass surface +
-- exercise pack/unpack via the shared C helpers.
public export
Compatible (MlxExecutor MCpu) Ternary where
public export
Compatible (MlxExecutor MGpu) Ternary where
public export
Compatible (MlxExecutor MCpu) Binary where
public export
Compatible (MlxExecutor MGpu) Binary where

----------------------------------------------------------------------
-- UserExecutorTransfer instance (cross-backend transfer surface)
--
-- mlx routes the intra-backend hardware migration through its
-- stream-switch mechanism rather than libtorch-style `.to()`; the
-- existing `tensor_to_device_mlx` C symbol no-ops because mlx's
-- arrays are device-agnostic at the metadata level (the stream tag
-- is what drives where compute runs). The runtime stream is picked
-- by `deviceStreamTag` on the `UserExecutorCore (MlxExecutor s)` instance,
-- so an intra-backend `toExecutor` (MCpu→MGpu) actually has to land
-- back through host memory for stream-switch to be observable; we
-- preserve the parametric implementation here for shape parity
-- with TapeExecutor / TorchExecutor.
----------------------------------------------------------------------

%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-to-doubles-return-mlx)) (set-top-level-value! 'idris-ffi-tensor-to-doubles-return-mlx (foreign-procedure \"tensor_to_doubles_return_mlx\" (void* void*) void*))) ((top-level-value 'idris-ffi-tensor-to-doubles-return-mlx) (vector-ref a0 2) a1))"
prim__toHostMlx : AnyPtr -> AnyPtr -> AnyPtr

-- Host buffer helpers — unified across backends, see Executor/Tape.idr.
%foreign "C:tensor_alloc_doubles,libidrisml"
prim__allocHostMlx : Int -> AnyPtr

%foreign "C:tensor_free_doubles,libidrisml"
prim__freeHostMlx : AnyPtr -> PrimIO ()

%foreign "C:tensor_alloc_ints,libidrisml"
prim__allocIntHostMlx : Int -> AnyPtr

%foreign "C:tensor_free_ints,libidrisml"
prim__freeIntHostMlx : AnyPtr -> PrimIO ()

%foreign "C:tensor_write_int_return,libidrisml"
prim__setIntHostMlx : AnyPtr -> Int -> Int -> AnyPtr

||| Dtag-aware create-from-host: delegates to the dtag-dispatch
||| `prim__createStreamedMlx` so destination storage matches the
||| type-level `dt` instead of unconditionally constructing F32
||| (mlx's `tensor_create` default — note the *opposite* lie to
||| tape/torch's F64). The stream is threaded per-instance below.
prim__createFromHostMlx : Int -> AnyPtr -> AnyPtr -> Int -> Int -> Int -> AnyPtr
prim__createFromHostMlx stream dat sh rank rg dtag =
  prim__createStreamedMlx dat sh rank rg stream dtag

%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-to-device-mlx)) (set-top-level-value! 'idris-ffi-tensor-to-device-mlx (foreign-procedure \"tensor_to_device_mlx\" (void* string) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-mlx)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-mlx (foreign-procedure \"tensor_retain_handle_mlx\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-to-device-mlx) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-mlx) raw_r) wr)))"
prim__intraMigrateMlx : AnyPtr -> String -> AnyPtr

public export
{s : MlxStream} -> UserExecutorTransfer (MlxExecutor s) where
  backendTag         = "mlx"
  -- >>> GENERATED FROM ffi_manifest.py — gen-executor-instances.py >>>
  primAllocHost    = prim__allocHostMlx
  primAllocIntHost = prim__allocIntHostMlx
  primFreeHost     = prim__freeHostMlx
  primFreeIntHost  = prim__freeIntHostMlx
  primIntraMigrate = prim__intraMigrateMlx
  primSetIntHost   = prim__setIntHostMlx
  primToHost       = prim__toHostMlx
  -- <<< END GENERATED <<<
  -- Hand-written overrides:
  primCreateFromHost = prim__createFromHostMlx (streamTag s)

----------------------------------------------------------------------
-- UserExecutorQuant instance (#411 BitNet b1.58)
----------------------------------------------------------------------
--
-- Mlx unpacks the 2-bit codes to int8 at construction (storage is
-- `mx::array` with dtype `mx::int8`); the forward dequants via
-- `mx::astype` then runs `mx::matmul`. The streamed variants take a
-- trailing stream-tag arg (managed by hand below — manifest-driven
-- wrappers don't cover `*_mlx_streamed` compound names). See
-- design-decisions.md "Per-backend ternary storage" + backend_mlx/
-- nn/quantization/bitlinear.cpp.

%foreign "scheme:(lambda (a0 a1 a2 a3 a4 a5) (let ((raw_r ((foreign-procedure \"tensor_create_ternary_packed_2d_mlx_streamed\" (void* int int int int int) void*) a0 a1 a2 a3 a4 a5))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__createTernaryPacked2dMlxStreamed : AnyPtr -> Int -> Int -> Int -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2 a3 a4) (let ((raw_r ((foreign-procedure \"tensor_bitlinear_fwd_mlx_streamed\" (void* void* void* void* int) void*) (vector-ref a0 2) (vector-ref a1 2) (vector-ref a2 2) (vector-ref a3 2) a4))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__bitlinearFwdMlxStreamed : AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1) (let ((raw_r ((foreign-procedure \"tensor_absmean_per_row_2d_mlx_streamed\" (void* int) void*) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__absmeanPerRow2dMlxStreamed : AnyPtr -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2) (let ((raw_r ((foreign-procedure \"tensor_ternary_quant_with_scale_2d_mlx_streamed\" (void* void* int) void*) (vector-ref a0 2) (vector-ref a1 2) a2))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__ternaryQuantWithScale2dMlxStreamed : AnyPtr -> AnyPtr -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2 a3) (let ((raw_r ((foreign-procedure \"tensor_create_ternary_from_hf_packed_2d_mlx_streamed\" (void* int int int) void*) a0 a1 a2 a3))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__createTernaryFromHfPacked2dMlxStreamed : AnyPtr -> Int -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2 a3 a4 a5 a6 a7) (let ((raw_r ((foreign-procedure \"tensor_bitlinear_fwd_hf_quant_mlx_streamed\" (void* double void* void* int void* double int) void*) (vector-ref a0 2) a1 (vector-ref a2 2) (if a3 (vector-ref a3 2) 0) a4 (if a5 (vector-ref a5 2) 0) a6 a7))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__bitlinearFwdHfQuantMlxStreamed : AnyPtr -> Double -> AnyPtr -> AnyPtr -> Int -> AnyPtr -> Double -> Int -> AnyPtr

public export
{s : MlxStream} -> UserExecutorQuant (MlxExecutor s) where
  -- >>> GENERATED FROM ffi_manifest.py — gen-executor-instances.py >>>
  -- <<< END GENERATED <<<
  -- Hand-written overrides:
  primCreateTernaryPacked2d bytes bc o i rg =
    prim__createTernaryPacked2dMlxStreamed bytes bc o i rg (streamTag s)
  primBitlinearFwd w sc x b =
    prim__bitlinearFwdMlxStreamed w sc x b (streamTag s)
  primBitlinearFwdHfQuant w ws x b urn rnw eps =
    prim__bitlinearFwdHfQuantMlxStreamed w ws x b urn rnw eps (streamTag s)
  primAbsmeanPerRow2d w =
    prim__absmeanPerRow2dMlxStreamed w (streamTag s)
  primTernaryQuantWithScale2d w sc =
    prim__ternaryQuantWithScale2dMlxStreamed w sc (streamTag s)
  primCreateTernaryFromHfPacked2d bytes o i =
    prim__createTernaryFromHfPacked2dMlxStreamed bytes o i (streamTag s)

----------------------------------------------------------------------
-- HardwareClass: mlx CPU stream is host CPU, GPU stream is Apple GPU.
----------------------------------------------------------------------

public export
{s : MlxStream} -> HardwareClassed (MlxExecutor s) where
  hardwareClass = case s of
    MCpu => HostCpu
    MGpu => AppleGpu

----------------------------------------------------------------------
-- Hardware (type-level): map each mlx stream to its kind tag.
----------------------------------------------------------------------

public export RunsOn (MlxExecutor MCpu) Cpu      where
public export RunsOn (MlxExecutor MGpu) AppleGpu where

----------------------------------------------------------------------
-- Backend (type-level): every mlx stream is provided by MlxBackend.
----------------------------------------------------------------------

public export
{s : MlxStream} -> RunsVia (MlxExecutor s) MlxBackend where

----------------------------------------------------------------------
-- Preset: per-Hardware defaults for mlx.
--   * Cpu      → MlxExecutor MCpu + F64
--   * AppleGpu → MlxExecutor MGpu + F32   (Metal stream is F32-only since mlx 0.31)
-- mlx is macOS-only, so no Cuda preset.
----------------------------------------------------------------------

public export
Preset MlxBackend Cpu where
  presetExecutor = MlxExecutor MCpu
  presetDType    = F64

public export
Preset MlxBackend AppleGpu where
  presetExecutor = MlxExecutor MGpu
  presetDType    = F32
