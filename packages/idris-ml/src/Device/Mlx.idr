||| `MlxDev` — `UserDeviceCore` instance for the mlx backend.
|||
||| Forwards to the mlx-suffixed C symbols emitted under Phase 1's
||| `rename_mlx.h` (e.g. `tensor_add_mlx`). Only resolvable at runtime
||| if the build's BACKEND list includes `mlx` (Apple-only).
module Device.Mlx

import Device.Core
import DType.Core


----------------------------------------------------------------------
-- Per-symbol bindings to the mlx backend's suffixed C exports
----------------------------------------------------------------------

-- UserDeviceCore (MlxDev s) instance methods call the streamed
-- variants below; the trailing `Int` stream_tag is derived from the
-- type-level `s` via `streamTag` (0 = MCpu, 1 = MGpu). The unstreamed
-- `prim__*Mlx` declarations are kept for any caller that hasn't moved
-- to the streamed surface (currently none in this file).

%foreign "scheme:(lambda (a0 a1 a2) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (let ((raw_r ((foreign-procedure \"tensor_create_scalar_mlx_streamed\" (double int int) void*) a0 a1 a2))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__createScalarMlxStreamed : Double -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2 a3 a4) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (let ((raw_r ((foreign-procedure \"tensor_create_mlx_streamed\" (void* void* int int int) void*) a0 a1 a2 a3 a4))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__createMlxStreamed : AnyPtr -> AnyPtr -> Int -> Int -> Int -> AnyPtr

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


----------------------------------------------------------------------
-- MlxStream + MlxDev parameterized family
--
-- `MlxDev` is parameterized over a stream tag (`MGpu` vs `MCpu`) so
-- that `MlxDev MGpu` and `MlxDev MCpu` are distinct device types at
-- the type level while sharing one set of `UserDevice*` instances.
-- The instance bodies derive an `Int` stream tag from `s` via
-- `streamTag` and thread it through the `_mlx_streamed` FFI surface
-- to `mx::StreamContext` on the C side, so each op runs on the
-- stream the type system claimed. Mirrors the `CUDA Nat` precedent
-- in `Device.idr`.
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
data MlxDev : MlxStream -> Type where
  MkMlxDev : MlxDev s

||| `MlxDev MGpu` alias. Metal GPU stream. Only supports F32; dt
||| has no `Compatible` instance and tensors of `MlxGpu dt` fail to
||| typecheck at the construction site.
public export
MlxGpu : Type
MlxGpu = MlxDev MGpu

||| `MlxDev MCpu` alias. mlx CPU stream. Supports both F32 and dt.
public export
MlxCpu : Type
MlxCpu = MlxDev MCpu

||| Int encoding of an `MlxStream` for the streamed FFI surface.
||| `MCpu → 0`, `MGpu → 1`. Mirrored on the C side by
||| `stream_for_tag(int)` in `backend_mlx.cpp`. Each `UserDeviceCore`
||| (and sibling-interface) method on `MlxDev s` derives the tag from
||| `s` and threads it to the corresponding `_mlx_streamed` FFI so
||| the op runs on the correct mlx stream — honouring the type-level
||| device parameter rather than the global `mx::set_default_device`.
public export
streamTag : MlxStream -> Int
streamTag MCpu = 0
streamTag MGpu = 1

%foreign "scheme:(lambda (a0 a1 a2)  (when (not (top-level-bound? 'idris-ffi-tensor-item-2d-mlx)) (set-top-level-value! 'idris-ffi-tensor-item-2d-mlx (foreign-procedure \"tensor_item_2d_mlx\" (void* int int) double))) ((top-level-value 'idris-ffi-tensor-item-2d-mlx) (vector-ref a0 2) a1 a2))"
prim__item2dMlx : AnyPtr -> Int -> Int -> Double
%foreign "scheme:(lambda (a0 a1 a2) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-mnist-get-image-mlx)) (set-top-level-value! 'idris-ffi-mnist-get-image-mlx (foreign-procedure \"mnist_get_image_mlx\" (void* int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-mlx)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-mlx (foreign-procedure \"tensor_retain_handle_mlx\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-mnist-get-image-mlx) a0 a1 a2))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-mlx) raw_r) wr)))"
prim__mnistGetImageMlx : AnyPtr -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-tensor-one-hot-mlx)) (set-top-level-value! 'idris-ffi-tensor-one-hot-mlx (foreign-procedure \"tensor_one_hot_mlx\" (void* int int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-mlx)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-mlx (foreign-procedure \"tensor_retain_handle_mlx\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-one-hot-mlx) a0 a1 a2 a3))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-mlx) raw_r) wr)))"
prim__oneHotMlx : AnyPtr -> Int -> Int -> Int -> AnyPtr

public export
{s : MlxStream} -> UserDeviceCore (MlxDev s) where
  deviceName       = case s of
                       MGpu => "mlx:gpu"
                       MCpu => "mlx:cpu"
  deviceStreamTag  = streamTag s
  primCreateScalar v rg          = prim__createScalarMlxStreamed v rg (streamTag s)
  primCreate d sh r rg           = prim__createMlxStreamed d sh r rg (streamTag s)
  primFree h                     = prim__freeMlxStreamed h (streamTag s)
  primItem h                     = prim__itemMlxStreamed h (streamTag s)
  primItem1d h i                 = prim__item1dMlxStreamed h i (streamTag s)
  primClone h                    = prim__cloneMlxStreamed h (streamTag s)
  primAdd a b                    = prim__addMlxStreamed a b (streamTag s)
  primSub a b                    = prim__subMlxStreamed a b (streamTag s)
  primMul a b                    = prim__mulMlxStreamed a b (streamTag s)
  primDiv a b                    = prim__divMlxStreamed a b (streamTag s)
  primNeg a                      = prim__negMlxStreamed a (streamTag s)
  primAbs a                      = prim__absMlxStreamed a (streamTag s)
  primExp a                      = prim__expMlxStreamed a (streamTag s)
  primLog a                      = prim__logMlxStreamed a (streamTag s)
  primSqrt a                     = prim__sqrtMlxStreamed a (streamTag s)
  primPow b e                    = prim__powMlxStreamed b e (streamTag s)
  primSigmoid a                  = prim__sigmoidMlxStreamed a (streamTag s)
  primTanh a                     = prim__tanhMlxStreamed a (streamTag s)
  primAddScalar a v              = prim__addScalarMlxStreamed a v (streamTag s)
  primMulScalar a v              = prim__mulScalarMlxStreamed a v (streamTag s)
  primClampMin a v               = prim__clampMinMlxStreamed a v (streamTag s)
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

%foreign "scheme:(lambda (a0 a1 a2 a3)  (let ((raw_r ((foreign-procedure \"tensor_scatter_add_mlx_streamed\" (void* void* int int) void*) (vector-ref a0 2) (vector-ref a1 2) a2 a3))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__scatterAddMlxStreamed : AnyPtr -> AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2 a3)  (let ((raw_r ((foreign-procedure \"tensor_argsort_mlx_streamed\" (void* int int int) void*) (vector-ref a0 2) a1 a2 a3))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__argsortMlxStreamed : AnyPtr -> Int -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_cumprod_mlx_streamed\" (void* int int) void*) (vector-ref a0 2) a1 a2))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__cumprodMlxStreamed : AnyPtr -> Int -> Int -> AnyPtr


public export
{s : MlxStream} -> UserDeviceLinear (MlxDev s) where
  primMv a b               = prim__mvMlxStreamed a b (streamTag s)
  primMm a b               = prim__mmMlxStreamed a b (streamTag s)
  primMatmul a b           = prim__matmulMlxStreamed a b (streamTag s)
  primLinear w x bs        = prim__linearMlxStreamed w x bs (streamTag s)
  primDot a b              = prim__dotMlxStreamed a b (streamTag s)
  primOuter a b            = prim__outerMlxStreamed a b (streamTag s)
  primBmm a b              = prim__bmmMlxStreamed a b (streamTag s)
  primLinear2d w x bs      = prim__linear2dMlxStreamed w x bs (streamTag s)
  primSum a                = prim__sumMlxStreamed a (streamTag s)
  primMean a               = prim__meanMlxStreamed a (streamTag s)
  primTensorMin a          = prim__tensorMinMlxStreamed a (streamTag s)
  primTensorMax a          = prim__tensorMaxMlxStreamed a (streamTag s)
  primSumDim a d k         = prim__sumDimMlxStreamed a d k (streamTag s)
  primSelect a d i         = prim__selectMlxStreamed a d i (streamTag s)
  primUnsqueeze a d        = prim__unsqueezeMlxStreamed a d (streamTag s)
  primSqueeze a d          = prim__squeezeMlxStreamed a d (streamTag s)
  primStack ts c d         = prim__stackMlxStreamed ts c d (streamTag s)
  primView1d a n           = prim__view1dMlxStreamed a n (streamTag s)
  primView2d a r c         = prim__view2dMlxStreamed a r c (streamTag s)
  primReshape1d a n        = prim__reshape1dMlxStreamed a n (streamTag s)
  primReshape2d a r c      = prim__reshape2dMlxStreamed a r c (streamTag s)
  primReshape3d a d0 d1 d2 = prim__reshape3dMlxStreamed a d0 d1 d2 (streamTag s)
  primReshape4d a d0 d1 d2 d3 = prim__reshape4dMlxStreamed a d0 d1 d2 d3 (streamTag s)
  primTile2d a r0 r1       = prim__tile2dMlxStreamed a r0 r1 (streamTag s)
  primNarrow a d st ln     = prim__narrowMlxStreamed a d st ln (streamTag s)
  primTransposeLast2 a     = prim__transposeLast2MlxStreamed a (streamTag s)
  primTranspose2d a        = prim__transpose2dMlxStreamed a (streamTag s)
  primCat ts c d           = prim__catMlxStreamed ts c d (streamTag s)
  primCat2 a b             = prim__cat2MlxStreamed a b (streamTag s)
  primConcat2dAxis1 a b    = prim__concat2dAxis1MlxStreamed a b (streamTag s)
  primGather a i d         = prim__gatherMlxStreamed a i d (streamTag s)
  primScatterAdd a i d     = prim__scatterAddMlxStreamed a i d (streamTag s)
  primArgsort a d desc     = prim__argsortMlxStreamed a d desc (streamTag s)
  primCumprod a d          = prim__cumprodMlxStreamed a d (streamTag s)


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


public export
{s : MlxStream} -> UserDeviceNN (MlxDev s) where
  primGelu a = prim__geluMlxStreamed a (streamTag s)
  primLeakyRelu a b = prim__leakyReluMlxStreamed a b (streamTag s)
  primSilu a = prim__siluMlxStreamed a (streamTag s)
  primSoftplus a = prim__softplusMlxStreamed a (streamTag s)
  primSoftmax a b = prim__softmaxMlxStreamed a b (streamTag s)
  primLogSoftmax a b = prim__logSoftmaxMlxStreamed a b (streamTag s)
  primSoftmax2d a = prim__softmax2dMlxStreamed a (streamTag s)
  primLogSoftmax2d a = prim__logSoftmax2dMlxStreamed a (streamTag s)
  primSoftmax3d a = prim__softmax3dMlxStreamed a (streamTag s)
  primMaskedFill a b c = prim__maskedFillMlxStreamed a b c (streamTag s)
  primExpandMask a b = prim__expandMaskMlxStreamed a b (streamTag s)
  primLayerNorm2d a b c d = prim__layerNorm2dMlxStreamed a b c d (streamTag s)
  primBatchNorm a b c d e f g h i j = prim__batchNormMlxStreamed a b c d e f g h i j (streamTag s)
  primDropout a b c d = prim__dropoutMlxStreamed a b c d (streamTag s)
  primEmbedding a b c d = prim__embeddingMlxStreamed a b c d (streamTag s)
  primCosineSimilarity a b c = prim__cosineSimilarityMlxStreamed a b c (streamTag s)
  primCrossAttention a b c d e = prim__crossAttentionMlxStreamed a b c d e (streamTag s)
  primBceWithLogits a b = prim__bceWithLogitsMlxStreamed a b (streamTag s)
  primGruCell a b c d = prim__gruCellMlxStreamed a b c d (streamTag s)
  primLstmGatesPair a b c = prim__lstmGatesPairMlxStreamed a b c (streamTag s)
  primPairFirst a = prim__pairFirstMlxStreamed a (streamTag s)
  primPairSecond a = prim__pairSecondMlxStreamed a (streamTag s)
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
{s : MlxStream} -> UserDeviceConv (MlxDev s) where
  primConv1d a b c d e = prim__conv1dMlxStreamed a b c d e (streamTag s)
  primConv1dCircular a b = prim__conv1dCircularMlxStreamed a b (streamTag s)
  primAvgPool1d a b c = prim__avgPool1dMlxStreamed a b c (streamTag s)
  primMaxPool1d a b c = prim__maxPool1dMlxStreamed a b c (streamTag s)
  primConv2d a b c d e f g = prim__conv2dMlxStreamed a b c d e f g (streamTag s)
  primConv2dBatched a b c d e f g = prim__conv2dBatchedMlxStreamed a b c d e f g (streamTag s)
  primAvgPool2d a b c d e = prim__avgPool2dMlxStreamed a b c d e (streamTag s)
  primMaxPool2d a b c d e = prim__maxPool2dMlxStreamed a b c d e (streamTag s)
  primMaxPool2dBatched a b c d e = prim__maxPool2dBatchedMlxStreamed a b c d e (streamTag s)
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
%foreign "C:polyak_blend_mlx,libidrisml"
prim__polyakBlendMlx : Double -> String -> String -> PrimIO Int
%foreign "C:param_count_mlx,libidrisml"
prim__paramCountMlx : PrimIO Int
%foreign "C:param_name_mlx,libidrisml"
prim__paramNameMlx : Int -> PrimIO String
%foreign "C:param_grad_item_at_mlx,libidrisml"
prim__paramGradItemAtMlx : Int -> Int -> PrimIO Double
%foreign "C:param_zero_all_grads_mlx,libidrisml"
prim__paramZeroAllMlx : PrimIO ()
%foreign "C:optimizer_create_sgd_mlx,libidrisml"
prim__optimizerCreateSgdMlx : Double -> AnyPtr
%foreign "C:optimizer_create_rmsprop_mlx,libidrisml"
prim__optimizerCreateRmspropMlx : Double -> Double -> Double -> Double -> Double -> AnyPtr
%foreign "C:optimizer_create_adam_mlx,libidrisml"
prim__optimizerCreateAdamMlx : Double -> Double -> Double -> Double -> AnyPtr
%foreign "C:optimizer_create_adam_group_mlx,libidrisml"
prim__optimizerCreateAdamGroupMlx : Double -> Double -> Double -> Double -> String -> AnyPtr
%foreign "C:optimizer_create_adamw_mlx,libidrisml"
prim__optimizerCreateAdamWMlx : Double -> Double -> Double -> Double -> Double -> AnyPtr
%foreign "C:optimizer_set_lr_mlx,libidrisml"
prim__optimizerSetLrMlx : AnyPtr -> Double -> PrimIO ()
%foreign "C:optimizer_set_param_lr_mlx,libidrisml"
prim__optimizerSetParamLrMlx : AnyPtr -> String -> Double -> PrimIO ()
%foreign "scheme:(lambda (a0 a1 a2 a3 a4)  (when (not (top-level-bound? 'idris-ffi-native-train-step-mlx)) (set-top-level-value! 'idris-ffi-native-train-step-mlx (foreign-procedure \"native_train_step_mlx\" (void* int double void* double) double))) ((top-level-value 'idris-ffi-native-train-step-mlx) a0 a1 a2 (vector-ref a3 2) a4))"
prim__nativeTrainStepMlx : AnyPtr -> Int -> Double -> AnyPtr -> Double -> Double
%foreign "scheme:(lambda (a0 a1 a2 a3 a4 a5)  (when (not (top-level-bound? 'idris-ffi-native-train-step-scaled-mlx)) (set-top-level-value! 'idris-ffi-native-train-step-scaled-mlx (foreign-procedure \"native_train_step_scaled_mlx\" (void* int double void* double double) double))) ((top-level-value 'idris-ffi-native-train-step-scaled-mlx) a0 a1 a2 (vector-ref a3 2) a4 a5))"
prim__nativeTrainStepScaledMlx : AnyPtr -> Int -> Double -> AnyPtr -> Double -> Double -> Double
%foreign "C:param_save_mlx,libidrisml"
prim__paramSaveMlx : String -> PrimIO Int
%foreign "C:param_load_mlx,libidrisml"
prim__paramLoadMlx : String -> PrimIO Int
%foreign "C:param_load_with_policy_mlx,libidrisml"
prim__paramLoadWithPolicyMlx : String -> Int -> PrimIO Int
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
prim__liveCountMlx : Int -> Int
%foreign "C:tensor_peak_live_count_mlx,libidrisml"
prim__peakLiveCountMlx : Int -> Int
%foreign "C:tensor_perf_reset_mlx,libidrisml"
prim__perfResetMlx : PrimIO ()
%foreign "C:tensor_perf_op_count_mlx,libidrisml"
prim__perfOpCountMlx : PrimIO Int
%foreign "scheme:(lambda (a0 a1 a2 a3 a4 a5 a6)  (when (not (top-level-bound? 'idris-ffi-tensor-sdpa-2d-mlx)) (set-top-level-value! 'idris-ffi-tensor-sdpa-2d-mlx (foreign-procedure \"tensor_sdpa_2d_mlx\" (void* void* void* int int int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-mlx)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-mlx (foreign-procedure \"tensor_retain_handle_mlx\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-sdpa-2d-mlx) (vector-ref a0 2) (vector-ref a1 2) (vector-ref a2 2) a3 a4 a5 a6))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-mlx) raw_r) wr)))"
prim__sdpa2dMlx : AnyPtr -> AnyPtr -> AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr


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
-- the matching block in Device/Tape.idr for the rationale.
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
prim__setInitSeedStreamedMlx : Bits64 -> Int -> ()

public export
{s : MlxStream} -> UserDeviceTraining (MlxDev s) where
  primCreateScalarStreamed        = prim__createScalarStreamedMlx
  primCreateStreamed              = prim__createStreamedMlx
  primCreate1dStreamed            = prim__create1dStreamedMlx
  primCreate2dStreamed            = prim__create2dStreamedMlx
  primCreateParam1dStreamed       = prim__createParam1dStreamedMlx
  primCreateParam2dStreamed       = prim__createParam2dStreamedMlx
  primCreateParam3dStreamed       = prim__createParam3dStreamedMlx
  primCreateParam4dStreamed       = prim__createParam4dStreamedMlx
  primCreateState1dStreamed       = prim__createState1dStreamedMlx
  primCreateState2dStreamed       = prim__createState2dStreamedMlx
  primCastStreamed                = prim__castStreamedMlx
  primCreateParam1dNormalStreamed = prim__createParam1dNormalStreamedMlx
  primCreateParam2dNormalStreamed = prim__createParam2dNormalStreamedMlx
  primCreateParam3dNormalStreamed = prim__createParam3dNormalStreamedMlx
  primCreateParam4dNormalStreamed = prim__createParam4dNormalStreamedMlx
  primCreateParam1dConstStreamed  = prim__createParam1dConstStreamedMlx
  primCreateParam2dConstStreamed  = prim__createParam2dConstStreamedMlx
  primCreateParam3dConstStreamed  = prim__createParam3dConstStreamedMlx
  primCreateParam4dConstStreamed  = prim__createParam4dConstStreamedMlx
  primSetInitSeedStreamed         = prim__setInitSeedStreamedMlx
  primRequiresGrad         = prim__requiresGradMlx
  primSetRequiresGrad      = prim__setRequiresGradMlx
  primBackward             = prim__backwardMlx
  primNoGradBegin          = prim__noGradBeginMlx
  primNoGradEnd            = prim__noGradEndMlx
  primDetach a = prim__detachMlxStreamed a (streamTag s)
  primWithGrad a = prim__withGradMlxStreamed a (streamTag s)
  primTensorDim            = prim__tensorDimMlx
  primTensorSizeAt         = prim__tensorSizeAtMlx
  primParamRegister        = prim__paramRegisterMlx
  primItem2d               = prim__item2dMlx
  primMnistGetImage        = prim__mnistGetImageMlx
  primOneHot               = prim__oneHotMlx
  primPolyakBlend          = prim__polyakBlendMlx
  primParamCount           = prim__paramCountMlx
  primParamName            = prim__paramNameMlx
  primParamGradItemAt      = prim__paramGradItemAtMlx
  primParamZeroAll         = prim__paramZeroAllMlx
  primOptimizerCreateSgd       = prim__optimizerCreateSgdMlx
  primOptimizerCreateRmsprop   = prim__optimizerCreateRmspropMlx
  primOptimizerCreateAdam      = prim__optimizerCreateAdamMlx
  primOptimizerCreateAdamGroup = prim__optimizerCreateAdamGroupMlx
  primOptimizerCreateAdamW     = prim__optimizerCreateAdamWMlx
  primOptimizerSetLr           = prim__optimizerSetLrMlx
  primOptimizerSetParamLr      = prim__optimizerSetParamLrMlx
  primNativeTrainStep          = prim__nativeTrainStepMlx
  primNativeTrainStepScaled    = prim__nativeTrainStepScaledMlx
  primParamSave                = prim__paramSaveMlx
  primParamLoad                = prim__paramLoadMlx
  primParamLoadWithPolicy      = prim__paramLoadWithPolicyMlx
  primOptimizerSave            = prim__optimizerSaveMlx
  primOptimizerLoad            = prim__optimizerLoadMlx
  primProfileReset             = prim__profileResetMlx
  primProfileReport            = prim__profileReportMlx
  primEpochBegin               = prim__epochBeginMlx
  primEpochEnd                 = prim__epochEndMlx
  primReleaseAllPersistent     = prim__releaseAllPersistentMlx
  primResetForEval             = prim__resetForEvalMlx
  primLiveCount                = prim__liveCountMlx
  primPeakLiveCount            = prim__peakLiveCountMlx
  primPerfReset                = prim__perfResetMlx
  primPerfOpCount              = prim__perfOpCountMlx
  primSdpa2d                   = prim__sdpa2dMlx


----------------------------------------------------------------------
-- Compatible (device, dtype) instances
--
-- `MlxCpu` (`MlxDev MCpu`) supports F32, F64, BF16, and F16. mlx's
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
-- Outstanding: ~72 hardcoded `mx::float32` constants in fused-op
-- kernels mix dtype with fp64/bf16/f16 inputs and produce wrong
-- math — being audited (see TODO row "Audit mlx fused-op + constant
-- pool dtype handling").
--
-- `MlxGpu` (`MlxDev MGpu`) supports F32, BF16, and F16. Metal GPUs
-- dropped float64 support in mlx 0.31 (`Compatible (MlxDev MGpu) F64`
-- stays deliberately missing — the PyTorch runtime "Float64 not
-- supported on Metal" error lifted to compile time), but M3+ has
-- hardware bfloat16 + float16 in Metal so the BF16 and F16 instances
-- are admissible.
----------------------------------------------------------------------

public export
Compatible (MlxDev MCpu) F64 where

public export
Compatible (MlxDev MCpu) F32 where

public export
Compatible (MlxDev MCpu) BF16 where

public export
Compatible (MlxDev MCpu) F16 where

public export
Compatible (MlxDev MGpu) F32 where

public export
Compatible (MlxDev MGpu) BF16 where

public export
Compatible (MlxDev MGpu) F16 where

-- DELIBERATELY NO `Compatible (MlxDev MGpu) F64` instance — Metal
-- has no fp64. (Int* + bool stay unwired on both streams.)

-- Sub-byte quantization dtypes (#411 BitNet b1.58). CPU stream only —
-- mlx's Metal sub-byte support requires custom kernels that arrive in
-- B3. The CPU instance is enough to validate the typeclass surface +
-- exercise pack/unpack via the shared C helpers.
public export
Compatible (MlxDev MCpu) Ternary where
public export
Compatible (MlxDev MGpu) Ternary where
public export
Compatible (MlxDev MCpu) Binary where
public export
Compatible (MlxDev MGpu) Binary where


----------------------------------------------------------------------
-- UserDeviceTransfer instance (cross-backend transfer surface)
--
-- mlx routes the intra-backend hardware migration through its
-- stream-switch mechanism rather than libtorch-style `.to()`; the
-- existing `tensor_to_device_mlx` C symbol no-ops because mlx's
-- arrays are device-agnostic at the metadata level (the stream tag
-- is what drives where compute runs). The runtime stream is picked
-- by `deviceStreamTag` on the `UserDeviceCore (MlxDev s)` instance,
-- so an intra-backend `toDevice` (MCpu→MGpu) actually has to land
-- back through host memory for stream-switch to be observable; we
-- preserve the parametric implementation here for shape parity
-- with TapeDev / TorchDev.
----------------------------------------------------------------------

%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-to-doubles-mlx)) (set-top-level-value! 'idris-ffi-tensor-to-doubles-mlx (foreign-procedure \"tensor_to_doubles_mlx\" (void* void*) void))) ((top-level-value 'idris-ffi-tensor-to-doubles-mlx) (vector-ref a0 2) a1))"
prim__toHostMlx : AnyPtr -> AnyPtr -> AnyPtr

-- Host buffer helpers — unified across backends, see Device/Tape.idr.
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

%foreign "scheme:(lambda (a0 a1 a2 a3) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-tensor-create-mlx)) (set-top-level-value! 'idris-ffi-tensor-create-mlx (foreign-procedure \"tensor_create_mlx\" (void* void* int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-mlx)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-mlx (foreign-procedure \"tensor_retain_handle_mlx\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-create-mlx) a0 a1 a2 a3))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-mlx) raw_r) wr)))"
prim__createFromHostMlx : AnyPtr -> AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-to-device-mlx)) (set-top-level-value! 'idris-ffi-tensor-to-device-mlx (foreign-procedure \"tensor_to_device_mlx\" (void* string) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-mlx)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-mlx (foreign-procedure \"tensor_retain_handle_mlx\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-to-device-mlx) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-mlx) raw_r) wr)))"
prim__intraMigrateMlx : AnyPtr -> String -> AnyPtr

public export
{s : MlxStream} -> UserDeviceTransfer (MlxDev s) where
  backendTag         = "mlx"
  primToHost         = prim__toHostMlx
  primAllocHost      = prim__allocHostMlx
  primFreeHost       = prim__freeHostMlx
  primAllocIntHost   = prim__allocIntHostMlx
  primFreeIntHost    = prim__freeIntHostMlx
  primSetIntHost     = prim__setIntHostMlx
  primCreateFromHost = prim__createFromHostMlx
  primIntraMigrate   = prim__intraMigrateMlx


----------------------------------------------------------------------
-- UserDeviceQuant instance (#411 BitNet b1.58)
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

public export
{s : MlxStream} -> UserDeviceQuant (MlxDev s) where
  primCreateTernaryPacked2d bytes bc o i rg =
    prim__createTernaryPacked2dMlxStreamed bytes bc o i rg (streamTag s)
  primBitlinearFwd w sc x b =
    prim__bitlinearFwdMlxStreamed w sc x b (streamTag s)
  primAbsmeanPerRow2d w =
    prim__absmeanPerRow2dMlxStreamed w (streamTag s)
  primTernaryQuantWithScale2d w sc =
    prim__ternaryQuantWithScale2dMlxStreamed w sc (streamTag s)


----------------------------------------------------------------------
-- HardwareClass: mlx CPU stream is host CPU, GPU stream is Apple GPU.
----------------------------------------------------------------------

public export
{s : MlxStream} -> HardwareClassed (MlxDev s) where
  hardwareClass = case s of
    MCpu => HostCpu
    MGpu => AppleGpu
