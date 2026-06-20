||| Cross-backend transfer + quantization instance slices.
module Executor.Torch.Transfer

import BackendLib
import DType.Core
import Executor.Core
import public Executor.Torch.Training
import Hardware
import Preset

----------------------------------------------------------------------
-- Compatible (TorchExecutor, dt).
--
-- F32 is admitted on every hardware variant (CPU / MPS / CUDA), F64
-- on CPU and CUDA. **MPS + F64 is deliberately NOT compatible**:
-- libtorch's MPS backend rejects F64 tensor *construction* outright
-- (`Cannot convert a MPS Tensor to float64 dtype`), not just at op
-- dispatch — so admitting the combination would let the type
-- system mint a value the runtime can't represent. Users wanting
-- F64-precision on MPS hardware should pin to `(TorchExecutor TCpu) F64`
-- or `(TorchExecutor (TCuda n)) F64`. Mirrors the
-- `Compatible (MlxExecutor MGpu) F64`-rejection demo for mlx.
----------------------------------------------------------------------

public export
{d : TorchHwDev} -> Compatible (TorchExecutor d) F32 where

public export
Compatible (TorchExecutor TCpu) F64 where

public export
{n : Nat} -> Compatible (TorchExecutor (TCuda n)) F64 where

-- Inference-only dtypes (2026-05-22): BF16/F16/Int*/Bool on TCpu + TCuda.
-- MPS BF16 added 2026-05-28 (opt-in via TORCH_DTYPE=BF16 BuildConfig
-- cell). The earlier "MPS deliberately excluded" exclusion was retired
-- after the Llama-3.2-1B perf push showed BF16 storage halves the
-- memory footprint (5 GB → 2.5 GB) and the libtorch MPS backend has
-- shipped BF16 kernel coverage for the ops the HF forward exercises.
-- F16 + Int* / Bool on MPS stay excluded — F16's reduced-precision
-- training support is unproven, and Int* + Bool on MPS run into the
-- same construction-time rejection as F64 (Metal storage support is
-- per-version). Wiring is torch-only; tape/mlx have no instances.
public export
Compatible (TorchExecutor TCpu) BF16 where
public export
{n : Nat} -> Compatible (TorchExecutor (TCuda n)) BF16 where
public export
Compatible (TorchExecutor TMps) BF16 where
public export
Compatible (TorchExecutor TCpu) F16 where
public export
{n : Nat} -> Compatible (TorchExecutor (TCuda n)) F16 where
public export
Compatible (TorchExecutor TCpu) I8 where
public export
{n : Nat} -> Compatible (TorchExecutor (TCuda n)) I8 where
public export
Compatible (TorchExecutor TCpu) I16 where
public export
{n : Nat} -> Compatible (TorchExecutor (TCuda n)) I16 where
public export
Compatible (TorchExecutor TCpu) I32 where
public export
{n : Nat} -> Compatible (TorchExecutor (TCuda n)) I32 where
public export
Compatible (TorchExecutor TCpu) I64 where
public export
{n : Nat} -> Compatible (TorchExecutor (TCuda n)) I64 where
public export
Compatible (TorchExecutor TCpu) U8 where
public export
{n : Nat} -> Compatible (TorchExecutor (TCuda n)) U8 where
public export
Compatible (TorchExecutor TCpu) Bool where
public export
{n : Nat} -> Compatible (TorchExecutor (TCuda n)) Bool where

-- Sub-byte quantization dtypes (#411 BitNet b1.58). CPU + CUDA only —
-- libtorch MPS lacks the construction-side sub-byte storage routing
-- (mirrors the Int* / Bool MPS exclusion). The Idris-side Compatible
-- gate is the structural prereq; per-backend kernels arrive in B3.
public export
Compatible (TorchExecutor TCpu) Ternary where
public export
Compatible (TorchExecutor TMps) Ternary where
public export
{n : Nat} -> Compatible (TorchExecutor (TCuda n)) Ternary where
public export
Compatible (TorchExecutor TCpu) Binary where
public export
Compatible (TorchExecutor TMps) Binary where
public export
{n : Nat} -> Compatible (TorchExecutor (TCuda n)) Binary where

----------------------------------------------------------------------
-- UserExecutorTransfer instance (cross-backend transfer surface)
--
-- The torch hardware-migrate path is the only one that does real
-- work: `tensor_to_device_torch(handle, "mps"|"cuda:n")` migrates a
-- libtorch tensor in place between CPU, MPS, and CUDA without
-- allocating a fresh handle, preserving param-registry membership.
----------------------------------------------------------------------

%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-to-doubles-return-torch)) (set-top-level-value! 'idris-ffi-tensor-to-doubles-return-torch (foreign-procedure \"tensor_to_doubles_return_torch\" (void* void*) void*))) ((top-level-value 'idris-ffi-tensor-to-doubles-return-torch) (vector-ref a0 2) a1))"
prim__toHostTorch : AnyPtr -> AnyPtr -> AnyPtr

-- Host buffer helpers — unified across backends, see Executor/Tape.idr.
%foreign "C:tensor_alloc_doubles,libidrisml"
prim__allocHostTorch : Int -> AnyPtr

%foreign "C:tensor_free_doubles,libidrisml"
prim__freeHostTorch : AnyPtr -> PrimIO ()

%foreign "C:tensor_alloc_ints,libidrisml"
prim__allocIntHostTorch : Int -> AnyPtr

%foreign "C:tensor_free_ints,libidrisml"
prim__freeIntHostTorch : AnyPtr -> PrimIO ()

%foreign "C:tensor_write_int_return,libidrisml"
prim__setIntHostTorch : AnyPtr -> Int -> Int -> AnyPtr

||| Create from host data + auto-migrate to the target torch hw.
||| Calls the dtag-dispatch `prim__createStreamedTorch` (stream tag
||| pinned to 0 — streams are an mlx concept; the create lands on
||| CPU with storage matching the type-level `dt`) then
||| `tensor_to_device_torch(handle, "mps"|"cuda:n")` so the returned
||| tensor is on the right hardware variant. Constructing in the
||| right dtype *before* the migrate is load-bearing: MPS rejects
||| F64, so the pre-dtag F64-always create made every F32 hop onto
||| TMps abort inside `tensor_to_device`.
prim__createFromHostTorch : (d : TorchHwDev) -> AnyPtr -> AnyPtr -> Int -> Int -> Int -> AnyPtr
prim__createFromHostTorch d dat sh rank rg dtag =
  prim__toDeviceTorch (prim__createStreamedTorch dat sh rank rg 0 dtag) (torchHwDevName d)

public export
{d : TorchHwDev} -> UserExecutorTransfer (TorchExecutor d) where
  backendTag         = "torch"
  -- >>> GENERATED FROM ffi_manifest.py — gen-executor-instances.py >>>
  primAllocHost    = prim__allocHostTorch
  primAllocIntHost = prim__allocIntHostTorch
  primFreeHost     = prim__freeHostTorch
  primFreeIntHost  = prim__freeIntHostTorch
  primSetIntHost   = prim__setIntHostTorch
  primToHost       = prim__toHostTorch
  -- <<< END GENERATED <<<
  -- Hand-written overrides:
  primCreateFromHost        = prim__createFromHostTorch d
  primIntraMigrate h hwName =
    prim__toDeviceTorch h hwName

----------------------------------------------------------------------
-- UserExecutorQuant instance (#411 BitNet b1.58)
----------------------------------------------------------------------
--
-- Torch unpacks the 2-bit codes to int8 at construction (storage is
-- `at::Tensor` with `at::ScalarType::Char`); the forward dequants
-- via `.to(scale.dtype())` then runs `at::matmul`. See
-- design-decisions.md "Per-backend ternary storage" + backend_torch/
-- nn/quantization/bitlinear.cpp.

%foreign "scheme:(lambda (a0 a1 a2 a3 a4) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-tensor-create-ternary-packed-2d-torch)) (set-top-level-value! 'idris-ffi-tensor-create-ternary-packed-2d-torch (foreign-procedure \"tensor_create_ternary_packed_2d_torch\" (void* int int int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-create-ternary-packed-2d-torch) a0 a1 a2 a3 a4))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__createTernaryPacked2dTorch : AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2 a3)  (when (not (top-level-bound? 'idris-ffi-tensor-bitlinear-fwd-torch)) (set-top-level-value! 'idris-ffi-tensor-bitlinear-fwd-torch (foreign-procedure \"tensor_bitlinear_fwd_torch\" (void* void* void* void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-bitlinear-fwd-torch) (vector-ref a0 2) (vector-ref a1 2) (vector-ref a2 2) (vector-ref a3 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__bitlinearFwdTorch : AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (when (not (top-level-bound? 'idris-ffi-tensor-absmean-per-row-2d-torch)) (set-top-level-value! 'idris-ffi-tensor-absmean-per-row-2d-torch (foreign-procedure \"tensor_absmean_per_row_2d_torch\" (void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-absmean-per-row-2d-torch) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__absmeanPerRow2dTorch : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-ternary-quant-with-scale-2d-torch)) (set-top-level-value! 'idris-ffi-tensor-ternary-quant-with-scale-2d-torch (foreign-procedure \"tensor_ternary_quant_with_scale_2d_torch\" (void* void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-ternary-quant-with-scale-2d-torch) (vector-ref a0 2) (vector-ref a1 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__ternaryQuantWithScale2dTorch : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-tensor-create-ternary-from-hf-packed-2d-torch)) (set-top-level-value! 'idris-ffi-tensor-create-ternary-from-hf-packed-2d-torch (foreign-procedure \"tensor_create_ternary_from_hf_packed_2d_torch\" (void* int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-create-ternary-from-hf-packed-2d-torch) a0 a1 a2))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__createTernaryFromHfPacked2dTorch : AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2 a3 a4 a5 a6)  (when (not (top-level-bound? 'idris-ffi-tensor-bitlinear-fwd-hf-quant-torch)) (set-top-level-value! 'idris-ffi-tensor-bitlinear-fwd-hf-quant-torch (foreign-procedure \"tensor_bitlinear_fwd_hf_quant_torch\" (void* double void* void* int void* double) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-bitlinear-fwd-hf-quant-torch) (vector-ref a0 2) a1 (vector-ref a2 2) (vector-ref a3 2) a4 (vector-ref a5 2) a6))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__bitlinearFwdHfQuantTorch : AnyPtr -> Double -> AnyPtr -> AnyPtr -> Int -> AnyPtr -> Double -> AnyPtr

public export
{d : TorchHwDev} -> UserExecutorQuant (TorchExecutor d) where
  -- >>> GENERATED FROM ffi_manifest.py — gen-executor-instances.py >>>
  primAbsmeanPerRow2d             = prim__absmeanPerRow2dTorch
  primBitlinearFwd                = prim__bitlinearFwdTorch
  primBitlinearFwdHfQuant         = prim__bitlinearFwdHfQuantTorch
  primCreateTernaryFromHfPacked2d = prim__createTernaryFromHfPacked2dTorch
  primCreateTernaryPacked2d       = prim__createTernaryPacked2dTorch
  primTernaryQuantWithScale2d     = prim__ternaryQuantWithScale2dTorch
  -- <<< END GENERATED <<<
