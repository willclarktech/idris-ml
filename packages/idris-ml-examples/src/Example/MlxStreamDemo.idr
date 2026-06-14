||| Cross-stream smoke test for `MlxExecutor MCpu` vs `MlxExecutor MGpu`,
||| on the v1 Nn surface.
|||
||| Two checks in one program:
|||
||| (1) Direct-op smoke: builds tensors on both the mlx CPU stream
|||     (`MlxCpu` + F64) and the mlx GPU stream (`MlxGpu` + F32), runs
|||     a `tneg` on each, and reads back the result. The op dispatch
|||     goes through `UserExecutorCore (MlxExecutor s)` which threads
|||     `streamTag s` into the `_mlx_streamed` C primitives.
|||
||| (2) Layer-forward smoke: builds an `Nn.Linear 4 4 (MlxExecutor MGpu) F32`
|||     and runs `forward` on a `MlxGpu F32` input. Exercises the full op
|||     cascade *and* the creation cascade — `linear` allocates
|||     weights/biases via `tparam2dNormal {ex} {dt}` which dispatches
|||     through the (stream-aware) `RuntimeDType` interface, threading
|||     `deviceStreamTag {ex=MlxExecutor MGpu} = 1`. Each tensor's
|||     lifecycle runs on the stream its type says, regardless of
|||     `MLX_DEVICE`.
|||
||| Mlx-only example. Built standalone in the mlx lane (not in the
||| examples ipkg); references `MlxCpu` / `MlxGpu` which are only linked
||| against the mlx backend, so it does not typecheck on tape / torch.
module Example.MlxStreamDemo

import Data.Vect

import BuildConfig
import Executor.Mlx
import ML

-- Same input both sides; the value-level result must agree (modulo
-- F32-vs-F64 precision) since the math is the same — only the stream
-- the op runs on differs.
inputCpu : Vect 4 Double
inputCpu = [1.0, 2.0, 3.0, 4.0]

inputGpu : Vect 4 Double
inputGpu = [5.0, 6.0, 7.0, 8.0]

readVec4 : {0 ex : Executor} -> UserExecutorCore ex => Tensor [4] ex dt g -> Vect 4 Double
readVec4 v = [ primItem1d {ex} v.tensorPtr 0
             , primItem1d {ex} v.tensorPtr 1
             , primItem1d {ex} v.tensorPtr 2
             , primItem1d {ex} v.tensorPtr 3
             ]

-- Read the single row of a [1, 4] forward output.
readRow4 : {0 ex : Executor} -> UserExecutorCore ex => Tensor [1, 4] ex dt g -> Vect 4 Double
readRow4 v = [ primItem2d {ex} v.tensorPtr 0 0
             , primItem2d {ex} v.tensorPtr 0 1
             , primItem2d {ex} v.tensorPtr 0 2
             , primItem2d {ex} v.tensorPtr 0 3
             ]

showVec : Vect 4 Double -> String
showVec [a, b, c, d] = "[" ++ show a ++ ", " ++ show b ++ ", "
                         ++ show c ++ ", " ++ show d ++ "]"

%default partial

main : IO ()
main = do
  requireMachine {m = ChosenMachine}
  putStrLn "=== mlx stream demo (MlxCpu F64 || MlxGpu F32) ==="

  -- CPU-stream tensor at F64. `primNeg` derives `streamTag MCpu = 0`.
  aCpu : Tensor [4] MlxCpu F64 WithGrad <- tensor {dims=[4]} (FromVect inputCpu)
  negCpu <- tneg aCpu
  putStrLn $ "MlxCpu F64  input  : " ++ showVec (readVec4 aCpu)
  putStrLn $ "MlxCpu F64  -input : " ++ showVec (readVec4 negCpu)

  -- GPU-stream tensor at F32. `primNeg` derives `streamTag MGpu = 1`.
  bGpu : Tensor [4] MlxGpu F32 WithGrad <- tensor {dims=[4]} (FromVect inputGpu)
  negGpu <- tneg bGpu
  putStrLn $ "MlxGpu F32  input  : " ++ showVec (readVec4 bGpu)
  putStrLn $ "MlxGpu F32  -input : " ++ showVec (readVec4 negGpu)

  -- Layer forward: an Nn.Linear on MlxGpu F32. The param allocation and
  -- the forward ops both route through the typeclass surface, threading
  -- `streamTag MGpu = 1`.
  putStrLn ""
  putStrLn "=== layer forward on MlxGpu F32 ==="
  linGpu <- runInit (linear {ex=MlxExecutor MGpu} {dt=F32} {i=4} {o=4})
  inGpu  <- tensor {dims=[1, 4]} (FromVect inputGpu)
  linOutGpu <- forward {b=1} linGpu (retypeGrad inGpu)
  putStrLn $ "MlxGpu F32  linOut : " ++ showVec (readRow4 linOutGpu)

  let expectCpu : Vect 4 Double = [-1.0, -2.0, -3.0, -4.0]
      expectGpu : Vect 4 Double = [-5.0, -6.0, -7.0, -8.0]
      cpuOk = readVec4 negCpu == expectCpu
      gpuOk = readVec4 negGpu == expectGpu
  -- Linear output depends on the random init — we don't check values,
  -- just that the forward ran without aborting on the GPU stream.
  if cpuOk && gpuOk
    then putStrLn "PASS"
    else putStrLn "FAIL: cross-stream values diverged from expectation"
