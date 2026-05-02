||| Cross-stream smoke test for `MlxDev MCpu` vs `MlxDev MGpu`.
|||
||| Two checks in one program:
|||
||| (1) Direct-op smoke: builds tensors on both the mlx CPU stream
|||     (`MlxCpu` + F64) and the mlx GPU stream (`MlxGpu` + F32), runs
|||     a `tneg` on each, and reads back the result. The op dispatch
|||     goes through `UserDeviceCore (MlxDev s)` which threads
|||     `streamTag s` into the `_mlx_streamed` C primitives, opening
|||     an `mx::StreamContext` from the cached `cpu_stream` /
|||     `gpu_stream` per call.
|||
||| (2) Layer-forward smoke: builds a `LinearState 4 4 (MlxDev MGpu) F32`
|||     and runs its `applyVar` on a `MlxGpu F32` input. Exercises the
|||     full L59 op cascade *and* the L60 creation cascade —
|||     `linearLayer` allocates weights/biases via `tparam2d {d} {dt}`
|||     which dispatches through the (now stream-aware) `RuntimeDType`
|||     interface, threading `deviceStreamTag {d=MlxDev MGpu} = 1`
|||     into `tensor_create_param_2d_f32_mlx_streamed`. Pre-L60 the
|||     param allocation hit the env stream regardless of the
|||     type-level `d`; pre-L59 the forward ops similarly hit the env
|||     stream. Both holes are closed.
|||
||| (3) `bulkToTensor` round-trip: explicitly stream-routes the
|||     state-tensor creation path used by data marshalling. Reads
|||     back values via `primItem1d {d}` — also stream-routed.
|||
||| Pre-L60: both `MlxCpu F64` and `MlxGpu F32` collapsed to whatever
||| stream `MLX_DEVICE` selected at process start for the typeclass-
||| routed ops AND for all dt-keyed creation primitives. Post-L60:
||| both axes (operator dispatch + dtype-cascade allocation) route
||| through the typeclass surface, with the stream tag derived from
||| the destination device's `UserDeviceCore.deviceStreamTag` method.
||| Each tensor's lifecycle runs on the stream its type says,
||| regardless of `MLX_DEVICE`.
|||
||| Mlx-only example. Skipped under tape / torch builds (its surface
||| references `MlxCpu` / `MlxGpu` which are only meaningfully linked
||| against the mlx backend).
module Example.MlxStreamDemo

import Data.Vect

import Backprop
import Array
import Device
import Device.Mlx
import Layer
import Tensor


-- Same input both sides; the value-level result must agree (modulo
-- F32-vs-F64 precision) since the math is the same — only the
-- stream the op runs on differs.
inputCpu : Vector 4 Double
inputCpu = VArray [SArray 1.0, SArray 2.0, SArray 3.0, SArray 4.0]

inputGpu : Vector 4 Double
inputGpu = VArray [SArray 5.0, SArray 6.0, SArray 7.0, SArray 8.0]


readVec4 : {0 d : Device} -> UserDeviceCore d => Tensor [4] d dt g -> Vect 4 Double
readVec4 v = [ primItem1d {d} v.tensorPtr 0
             , primItem1d {d} v.tensorPtr 1
             , primItem1d {d} v.tensorPtr 2
             , primItem1d {d} v.tensorPtr 3
             ]


showVec : Vect 4 Double -> String
showVec [a, b, c, d] = "[" ++ show a ++ ", " ++ show b ++ ", "
                         ++ show c ++ ", " ++ show d ++ "]"


main : IO ()
main = do
  putStrLn "=== mlx stream demo (MlxCpu F64 || MlxGpu F32) ==="

  -- CPU-stream tensor at F64. `UserDeviceCore (MlxDev MCpu)`'s
  -- `primNeg` derives `streamTag MCpu = 0` and threads it.
  let aCpuPtr = bulkToTensor {d=MlxCpu} {dt=F64} inputCpu
      aCpu    = the (Tensor [4] MlxCpu F64 WithGrad) (MkTensor aCpuPtr Nothing)
  negCpu <- tneg aCpu
  putStrLn $ "MlxCpu F64  input  : " ++ showVec (readVec4 aCpu)
  putStrLn $ "MlxCpu F64  -input : " ++ showVec (readVec4 negCpu)

  -- GPU-stream tensor at F32. `UserDeviceCore (MlxDev MGpu)`'s
  -- `primNeg` derives `streamTag MGpu = 1` and threads it.
  let bGpuPtr = bulkToTensor {d=MlxGpu} {dt=F32} inputGpu
      bGpu    = the (Tensor [4] MlxGpu F32 WithGrad) (MkTensor bGpuPtr Nothing)
  negGpu <- tneg bGpu
  putStrLn $ "MlxGpu F32  input  : " ++ showVec (readVec4 bGpu)
  putStrLn $ "MlxGpu F32  -input : " ++ showVec (readVec4 negGpu)

  -- Layer forward: a Linear on MlxGpu F32. Every `prim__*` inside
  -- `tlinear` now routes through `primLinear {d=MlxDev MGpu}`, which
  -- threads `streamTag MGpu = 1` into `tensor_linear_mlx_streamed`.
  -- Pre-L59 the call collapsed to the env stream regardless of the
  -- type-level d.
  putStrLn ""
  putStrLn "=== layer forward on MlxGpu F32 ==="
  linGpu <- linearLayer {d = MlxDev MGpu} {dt = F32} {i = 4} {o = 4} "lin_gpu_demo"
  (_, linOutGpu) <- applyVar linGpu bGpu
  putStrLn $ "MlxGpu F32  linOut : " ++ showVec (readVec4 linOutGpu)

  let expectCpu : Vect 4 Double = [-1.0, -2.0, -3.0, -4.0]
      expectGpu : Vect 4 Double = [-5.0, -6.0, -7.0, -8.0]
      cpuOk = readVec4 negCpu == expectCpu
      gpuOk = readVec4 negGpu == expectGpu
  -- Linear output values depend on the random Xavier init — we don't
  -- check values, just that the forward ran without aborting on the
  -- GPU stream (which is the regression gate L59 introduces).
  if cpuOk && gpuOk
    then putStrLn "PASS"
    else putStrLn "FAIL: cross-stream values diverged from expectation"
