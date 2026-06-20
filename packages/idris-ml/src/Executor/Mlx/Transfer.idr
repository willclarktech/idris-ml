||| Cross-backend transfer + quantization instance slices.
module Executor.Mlx.Transfer

import BackendLib
import DType.Core
import Executor.Core
import public Executor.Mlx.Training
import Hardware
import Preset

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
