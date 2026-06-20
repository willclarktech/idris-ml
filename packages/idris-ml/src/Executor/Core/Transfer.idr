||| Cross-backend transfer surface (UserExecutorTransfer) + the
||| quantization slice (UserExecutorQuant).
module Executor.Core.Transfer

import Executor.Core.Compute
import Executor.Core.Kind

----------------------------------------------------------------------
-- UserExecutorTransfer — cross-backend tensor transfer surface
--
-- Backends that implement this can act as source or destination for
-- the generic `toExecutor` in `Tensor.idr`. The interface bundles
-- everything `toExecutor` needs to:
--   (a) recognise the backend at runtime (via `backendTag`);
--   (b) migrate a handle in place when source and dest share a
--       backend (via `primIntraMigrate`); and
--   (c) round-trip through host memory when they don't (via the
--       `primToHost` / `primCreateFromHost` pair plus the host
--       buffer-alloc helpers).
--
-- The five built-in backends today (tape, torch CPU/MPS/CUDA, mlx
-- CPU/GPU) all implement this. Users adding a BYO backend that
-- wants to plug into the generic `toExecutor` machinery declare their
-- own instance with a globally-unique `backendTag` (convention:
-- namespace as "user/<name>" to avoid colliding with built-ins).
----------------------------------------------------------------------

||| Cross-backend transfer surface. See module-level docs above.
public export
interface UserExecutorCore ex => UserExecutorTransfer (0 ex : Executor) where
  ||| Globally unique string identifying the backend (NOT the
  ||| hardware variant). Built-ins reserve "tape", "torch", "mlx".
  ||| BYO backends should namespace with "user/<name>". `toExecutor`
  ||| compares tags to decide intra-vs-cross-backend path; a
  ||| collision would route an intra fast-path through a foreign
  ||| backend's C symbols and crash on handle type mismatch.
  backendTag : String

  ||| Read a tensor's contents into a caller-allocated host double
  ||| buffer of `tensor_numel(handle)` slots. Returns the buffer so
  ||| Idris-Chez can't elide the FFI; threaded downstream into
  ||| `primCreateFromHost` on the destination backend.
  primToHost : AnyPtr -> AnyPtr -> AnyPtr

  ||| Allocate / free a host double buffer of `n` slots. Backend-
  ||| neutral host memory (calloc/free under the hood).
  primAllocHost : Int -> AnyPtr
  primFreeHost  : AnyPtr -> PrimIO ()

  ||| Allocate / write / free a host int buffer of `n` slots. Used
  ||| by `toExecutor` to build the shape array that
  ||| `primCreateFromHost` consumes.
  primAllocIntHost : Int -> AnyPtr
  primFreeIntHost  : AnyPtr -> PrimIO ()
  ||| Write `val` to `buf[idx]` and return `buf` (for threading).
  primSetIntHost   : AnyPtr -> Int -> Int -> AnyPtr

  ||| Create a tensor on this device from a host-allocated double
  ||| buffer + int shape buffer. The (data, shape, rank, rg, dtag)
  ||| tuple matches `tensor_create_streamed`'s ABI minus the stream
  ||| tag (each backend pins its own stream / hw routing internally,
  ||| including the migration to the hardware variant, e.g. TMps, so
  ||| the returned handle is on the right hw). `dtag` is the
  ||| `RuntimeDType` tag — destination storage must match the
  ||| type-level `dt`, not silently default to F64 (tape/torch) or
  ||| F32 (mlx) like the plain `tensor_create` it used to wrap.
  primCreateFromHost : AnyPtr -> AnyPtr -> Int -> Int -> Int -> AnyPtr

  ||| Intra-backend hardware migration. Only sound when caller has
  ||| verified shared backend via `backendTag`. Mutates the
  ||| underlying tensor in place where the backend supports it;
  ||| preserves param-registry membership.
  primIntraMigrate : AnyPtr -> String -> AnyPtr

----------------------------------------------------------------------
-- UserExecutorQuant — quantization slice (BitNet b1.58 → #411)
----------------------------------------------------------------------

||| Opt-in slice for quantization ops. The three built-in backends
||| (tape, torch, mlx) implement it; BYO backends opt in only if they
||| want BitNet b1.58. Subclass of `UserExecutorCore` so a `UserExecutorQuant
||| d =>` constraint also brings the lifecycle + arithmetic surface.
|||
||| `primCreateTernaryPacked2d` takes (host-byte-buffer, byte_count, o,
||| i, requires_grad) and builds a `[o, i]` Ternary tensor with
||| dtype_tag = DT_TERNARY (25). Per-backend storage layout — packed
||| 2-bit on tape, unpacked int8 on torch/mlx — is hidden behind this
||| ABI; see design-decisions.md "Per-backend ternary storage".
|||
||| `primBitlinearFwd` runs y = (W_ternary .* scale[:, None]) @ x +
||| bias with W decoded inline (tape) or via int8-cast (torch/mlx).
||| Inference-only; STE-aware training is filed as a follow-up to #411.
|||
||| `primAbsmeanPerRow2d` returns the per-row absmean of a float [o, i]
||| weight: scale[j] = mean_k(|w[j, k]|), shape [o], same dtype as `w`.
||| `primTernaryQuantWithScale2d` takes the weight + that scale and
||| produces a Ternary tensor via per-row round-and-clamp. Together
||| they're the load-time recipe for converting an HF-stored F-dtype
||| BitNet checkpoint into our packed-ternary tag — see
||| `packages/pytorch/torch_ref/models/bitlinear.py`
||| `absmean_ternary_quant` for the reference implementation. Both
||| NoGrad; the pair runs once per linear at load.
|||
||| `primCreateTernaryFromHfPacked2d` reads HF's `[(o + 3) / 4, i]`
||| uint8 buffer (microsoft/bitnet-b1.58-2B-4T-style storage with
||| `{-1, 0, +1} -> {0, 1, 2}` codes packed along axis 0) and
||| produces a Ternary tensor in our layout. One-shot at safetensors
||| load.
public export
interface UserExecutorCore ex => UserExecutorQuant (0 ex : Executor) where
  primCreateTernaryPacked2d       : AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr
  primBitlinearFwd                : AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr
  primAbsmeanPerRow2d             : AnyPtr -> AnyPtr
  primTernaryQuantWithScale2d     : AnyPtr -> AnyPtr -> AnyPtr
  primCreateTernaryFromHfPacked2d : AnyPtr -> Int -> Int -> AnyPtr
  primBitlinearFwdHfQuant         : AnyPtr -> Double -> AnyPtr -> AnyPtr -> Int -> AnyPtr -> Double -> AnyPtr
