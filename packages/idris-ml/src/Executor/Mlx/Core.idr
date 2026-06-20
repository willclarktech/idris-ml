||| Executor type + Core / Streamed / HardwareClassed instance
||| slices (lifecycle, elementwise arithmetic, stream tag, hw class).
module Executor.Mlx.Core

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
export
prim__item2dMlx : AnyPtr -> Int -> Int -> Double
%foreign "scheme:(lambda (a0 a1 a2 a3) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-tensor-one-hot-mlx)) (set-top-level-value! 'idris-ffi-tensor-one-hot-mlx (foreign-procedure \"tensor_one_hot_mlx\" (void* int int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-mlx)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-mlx (foreign-procedure \"tensor_retain_handle_mlx\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-one-hot-mlx) a0 a1 a2 a3))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-mlx) raw_r) wr)))"
export
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
