||| Live cross-backend tensor transfer demo.
|||
||| Exercises `toExecutor` (Tensor.idr) — the backendTag-aware
||| migration introduced by Phase 6 of the 2026-05-19 device-
||| taxonomy refactor — across every (backend, hardware) cell that
||| runs on Apple Silicon:
|||
|||   TapeExecutor          (tape backend, host CPU,        F64 only)
|||   TorchExecutor TCpu    (libtorch on host CPU,           F32 + F64)
|||   TorchExecutor TMps    (libtorch on Apple Metal,        F32 only)
|||   MlxExecutor MCpu      (mlx CPU stream,                 F32 + F64)
|||   MlxExecutor MGpu      (mlx Metal stream,               F32 only)
|||
||| CUDA cells (`TorchExecutor (TCuda n)`) compile but aren't exercised
||| here — no CUDA hardware on the macOS CI lane.
|||
||| Requires a multi-backend build so all three sets of C symbols
||| (`tensor_*_tape`, `tensor_*_torch`, `tensor_*_mlx`) are linked:
|||
|||     make BACKEND=tape,torch,mlx example-transfer
|||
||| The Makefile target `example-transfer` invokes the multi-backend
||| build directly. Without all three backends linked, the program
||| crashes at FFI resolution on the first hop into the missing
||| backend.
|||
||| The disk-based SafeTensors round-trip (the historical
||| `Example/Transfer.idr` content) moved to `Example/Checkpoint.idr`.
module Example.Transfer

import Data.List
import Data.Vect

import BuildConfig
import Ml.Executor
import Ml.Tensor
import Ml.Util

----------------------------------------------------------------------
-- Helpers
----------------------------------------------------------------------

||| Create a 4-element Tensor on the destination backend using its
||| `UserExecutorTransfer.primCreateFromHost` directly. Bypasses
||| `bulkToTensor`, which routes via unified-name C symbols — those
||| land on the primary backend at link time, regardless of the
||| type-level `d`. For cross-backend transfer demos we need fresh
||| tensors to actually live on the *specific* dest backend.
|||
||| Every side-effecting step (alloc, write, create, free) goes
||| through `primIO` so Idris-2's Chez codegen sequences them with
||| `%World` instead of let-laziness. A naive let-chain version
||| would get hoisted to a module-level CSE'd constant whose
||| lambda body still references buffers allocated at module load —
||| every subsequent call to the same constant would re-run the
||| frees on the same pointers and trip libsystem_malloc's
||| "pointer being freed was not allocated" abort. The same
||| structure exists (and is tested) in `Test.Transfer.makeVec4`.
makeVec4 : {0 ex : Type} -> {0 dt : DType} ->
           UserExecutorTransfer ex => RuntimeDType dt => Compatible ex dt =>
           (Double, Double, Double, Double) ->
           IO (Tensor [4] ex dt WithGrad)
makeVec4 (a, b, c, dd) = do
  buf  <- primIO (\w => MkIORes (primAllocHost   {ex} 4)        w)
  buf1 <- primIO (\w => MkIORes (prim__setDouble buf  0 a)     w)
  buf2 <- primIO (\w => MkIORes (prim__setDouble buf1 1 b)     w)
  buf3 <- primIO (\w => MkIORes (prim__setDouble buf2 2 c)     w)
  buf4 <- primIO (\w => MkIORes (prim__setDouble buf3 3 dd)    w)
  sh   <- primIO (\w => MkIORes (primAllocIntHost {ex} 1)       w)
  sh1  <- primIO (\w => MkIORes (primSetIntHost   {ex} sh 0 4)  w)
  ptr  <- primIO (\w =>
            MkIORes (primCreateFromHost {ex} buf4 sh1 1 1 (dtypeTag {t=dt})) w)
  primIO (primFreeIntHost {ex} sh1)
  primIO (primFreeHost    {ex} buf4)
  pure (MkTensor ptr Nothing)

||| Read all four values out via the backend's `primItem1d`. Returns
||| F64 doubles regardless of the tensor's storage dtype (the C side
||| promotes F32 to double on readback). The `{ex}` annotations
||| pin the typeclass dispatch — without them, Idris can't infer
||| which backend's `primItem1d` to call from a bare `AnyPtr`.
read4 : {0 ex : Type} -> {0 dt : DType} -> UserExecutorCore ex =>
        Tensor [4] ex dt WithGrad ->
        (Double, Double, Double, Double)
read4 t =
  ( primItem1d {ex} t.tensorPtr 0
  , primItem1d {ex} t.tensorPtr 1
  , primItem1d {ex} t.tensorPtr 2
  , primItem1d {ex} t.tensorPtr 3 )

expected : (Double, Double, Double, Double)
expected = (1.0, 2.0, 3.0, 4.0)

||| Report a hop's values + verify within F32 round-trip tolerance.
||| F32 cells lose precision past ~1e-7 of the input magnitude; these
||| inputs (1.0–4.0) are exactly representable in F32, so the
||| tolerance is conservative-but-safe.
||| Returns True on match, False on mismatch.
reportStep : String -> (Double, Double, Double, Double) -> IO Bool
reportStep label (a, b, c, d) = do
  let (ea, eb, ec, ed) = Transfer.expected
  let delta            = abs (a - ea) + abs (b - eb) + abs (c - ec) + abs (d - ed)
  let ok               = delta < 1.0e-6
  putStrLn $ "  " ++ pad 22 label ++ "[" ++ show a ++ ", " ++ show b ++
             ", " ++ show c ++ ", " ++ show d ++ "]" ++
             (if ok then " ✓" else " ✗ MISMATCH (delta=" ++ show delta ++ ")")
  pure ok
  where
    pad : Nat -> String -> String
    pad n s =
      if length s >= n
        then s
        else s ++ pack (List.replicate (n `minus` length s) ' ')

----------------------------------------------------------------------
-- F64 hop: TapeExecutor ↔ TorchExecutor TCpu ↔ MlxExecutor MCpu
--
-- Exercises cross-backend transfers (differing backendTag → host
-- round-trip). Three distinct backends, all on host CPU silicon,
-- all admitting F64. Round-tripping through the chain should
-- preserve values exactly (no dtype cast).
----------------------------------------------------------------------

hopF64 : IO Bool
hopF64 = do
  putStrLn "=== F64 hop (host-CPU silicon, 3 backends) ==="
  putStrLn "    Starts on TapeExecutor, hops cross-backend to TorchExecutor TCpu,"
  putStrLn "    then to MlxExecutor MCpu, back to TapeExecutor. Each transition is"
  putStrLn "    a backendTag-mismatch → host-buffer round-trip."
  putStrLn ""

  v_tape <- makeVec4 {ex=TapeExecutor} {dt = F64} expected
  ok1 <- reportStep "TapeExecutor:"          (read4 v_tape)

  v_torch <- toExecutor (TorchExecutor TCpu) v_tape
  ok2 <- reportStep "→ TorchExecutor TCpu:"  (read4 v_torch)

  v_mlx <- toExecutor (MlxExecutor MCpu) v_torch
  ok3 <- reportStep "→ MlxExecutor MCpu:"    (read4 v_mlx)

  v_back <- toExecutor TapeExecutor v_mlx
  ok4 <- reportStep "→ TapeExecutor (back):" (read4 v_back)

  pure (ok1 && ok2 && ok3 && ok4)

----------------------------------------------------------------------
-- F32 hop: TorchExecutor TCpu ↔ TorchExecutor TMps ↔ MlxExecutor MGpu ↔ MlxExecutor MCpu
--
-- Exercises both intra-backend fast paths (matching backendTag →
-- in-place hardware migration via libtorch's `.to()` / mlx's
-- stream switch) and cross-backend round-trips (host buffer hop).
-- Every leg constructs real F32 storage — `primCreateFromHost`
-- threads the RuntimeDType tag, so the F32 create lands directly.
----------------------------------------------------------------------

hopF32 : IO Bool
hopF32 = do
  putStrLn ""
  putStrLn "=== F32 hop (includes Metal GPU cells, 4 cells) ==="
  putStrLn "    Starts on TorchExecutor TCpu as F32. Hops intra-torch"
  putStrLn "    (fast path via libtorch's `.to('mps')`) to TorchExecutor"
  putStrLn "    TMps, cross-backend to MlxExecutor MGpu, intra-mlx to"
  putStrLn "    MlxExecutor MCpu, back cross-backend to TorchExecutor"
  putStrLn "    TCpu."
  putStrLn ""

  v_torch_cpu <- makeVec4 {ex=TorchExecutor TCpu} {dt = F32} expected
  ok1 <- reportStep "TorchExecutor TCpu (F32):"  (read4 v_torch_cpu)

  v_torch_mps <- toExecutor (TorchExecutor TMps) v_torch_cpu
  ok2 <- reportStep "→ TorchExecutor TMps:"      (read4 v_torch_mps)

  v_mlx_gpu <- toExecutor (MlxExecutor MGpu) v_torch_mps
  ok3 <- reportStep "→ MlxExecutor MGpu:"        (read4 v_mlx_gpu)

  v_mlx_cpu <- toExecutor (MlxExecutor MCpu) v_mlx_gpu
  ok4 <- reportStep "→ MlxExecutor MCpu:"        (read4 v_mlx_cpu)

  v_back <- toExecutor (TorchExecutor TCpu) v_mlx_cpu
  ok5 <- reportStep "→ TorchExecutor TCpu:"      (read4 v_back)

  pure (ok1 && ok2 && ok3 && ok4 && ok5)

----------------------------------------------------------------------
-- Main
----------------------------------------------------------------------

main : IO ()
main = do
  requireMachine {m = ChosenMachine}
  putStrLn "=== Live cross-backend Tensor transfer demo ==="
  putStrLn ""
  putStrLn "Expected values at every hop: [1.0, 2.0, 3.0, 4.0]"
  putStrLn ""

  ok64 <- hopF64
  ok32 <- hopF32

  putStrLn ""
  let overall = ok64 && ok32
  putStrLn $ "RESULT\tf64=" ++ (if ok64 then "ok" else "FAIL") ++
             "\tf32=" ++ (if ok32 then "ok" else "FAIL") ++
             "\toverall=" ++ (if overall then "ok" else "FAIL")
