||| Live cross-backend tensor transfer demo.
|||
||| Exercises `toDevice` (Tensor.idr) — the backendTag-aware
||| migration introduced by Phase 6 of the 2026-05-19 device-
||| taxonomy refactor — across every (backend, hardware) cell that
||| runs on Apple Silicon:
|||
|||   TapeDev          (tape backend, host CPU,        F64 only)
|||   TorchDev TCpu    (libtorch on host CPU,           F32 + F64)
|||   TorchDev TMps    (libtorch on Apple Metal,        F32 only)
|||   MlxDev MCpu      (mlx CPU stream,                 F32 + F64)
|||   MlxDev MGpu      (mlx Metal stream,               F32 only)
|||
||| CUDA cells (`TorchDev (TCuda n)`) compile but aren't exercised
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

import Device
import Tensor
import Util


----------------------------------------------------------------------
-- Helpers
----------------------------------------------------------------------

||| Create a 4-element Tensor on the destination backend using its
||| `UserDeviceTransfer.primCreateFromHost` directly. Bypasses
||| `bulkToTensor`, which routes via unified-name C symbols — those
||| land on the primary backend at link time, regardless of the
||| type-level `d`. For cross-backend transfer demos we need fresh
||| tensors to actually live on the *specific* dest backend.
makeVec4 : {0 d : Type} -> {0 dt : DType} ->
           UserDeviceTransfer d => Compatible d dt =>
           (Double, Double, Double, Double) ->
           IO (Tensor [4] d dt WithGrad)
makeVec4 (a, b, c, dd) = do
  let buf  = primAllocHost    {d} 4
  let buf1 = prim__setDouble  buf  0 a
  let buf2 = prim__setDouble  buf1 1 b
  let buf3 = prim__setDouble  buf2 2 c
  let buf4 = prim__setDouble  buf3 3 dd
  let sh   = primAllocIntHost {d} 1
  let sh1  = primSetIntHost   {d} sh 0 4
  let ptr  = primCreateFromHost {d} buf4 sh1 1 1  -- rg=1 → WithGrad
  primIO (\w => MkIORes (primFreeIntHost {d} sh1)   w)
  primIO (\w => MkIORes (primFreeHost    {d} buf4) w)
  pure (MkTensor ptr Nothing)

||| Read all four values out via the backend's `primItem1d`. Returns
||| F64 doubles regardless of the tensor's storage dtype (the C side
||| promotes F32 to double on readback). The `{d}` annotations
||| pin the typeclass dispatch — without them, Idris can't infer
||| which backend's `primItem1d` to call from a bare `AnyPtr`.
read4 : {0 d : Type} -> {0 dt : DType} -> UserDeviceCore d =>
        Tensor [4] d dt WithGrad ->
        (Double, Double, Double, Double)
read4 t =
  ( primItem1d {d} t.tensorPtr 0
  , primItem1d {d} t.tensorPtr 1
  , primItem1d {d} t.tensorPtr 2
  , primItem1d {d} t.tensorPtr 3 )

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
  let delta = abs (a - ea) + abs (b - eb) + abs (c - ec) + abs (d - ed)
  let ok = delta < 1.0e-6
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
-- F64 hop: TapeDev ↔ TorchDev TCpu ↔ MlxDev MCpu
--
-- Exercises cross-backend transfers (differing backendTag → host
-- round-trip). Three distinct backends, all on host CPU silicon,
-- all admitting F64. Round-tripping through the chain should
-- preserve values exactly (no dtype cast).
----------------------------------------------------------------------

hopF64 : IO Bool
hopF64 = do
  putStrLn "=== F64 hop (host-CPU silicon, 3 backends) ==="
  putStrLn "    Starts on TapeDev, hops cross-backend to TorchDev TCpu,"
  putStrLn "    then to MlxDev MCpu, back to TapeDev. Each transition is"
  putStrLn "    a backendTag-mismatch → host-buffer round-trip."
  putStrLn ""

  v_tape <- makeVec4 {d = TapeDev} {dt = F64} expected
  ok1 <- reportStep "TapeDev:"          (read4 v_tape)

  v_torch <- toDevice (TorchDev TCpu) v_tape
  ok2 <- reportStep "→ TorchDev TCpu:"  (read4 v_torch)

  v_mlx <- toDevice (MlxDev MCpu) v_torch
  ok3 <- reportStep "→ MlxDev MCpu:"    (read4 v_mlx)

  v_back <- toDevice TapeDev v_mlx
  ok4 <- reportStep "→ TapeDev (back):" (read4 v_back)

  pure (ok1 && ok2 && ok3 && ok4)


----------------------------------------------------------------------
-- F32 hop: TorchDev TCpu ↔ TorchDev TMps ↔ MlxDev MGpu ↔ MlxDev MCpu
--
-- Exercises both intra-backend fast paths (matching backendTag →
-- in-place hardware migration via libtorch's `.to()` / mlx's
-- stream switch) and cross-backend round-trips (host buffer hop).
-- F32 only — TapeDev is excluded because it doesn't admit F32
-- (no parallel `float*` arena, see TODO row "Broaden runtime
-- dtype coverage across backends").
----------------------------------------------------------------------

hopF32 : IO Bool
hopF32 = do
  putStrLn ""
  putStrLn "=== F32 hop (includes Metal GPU cells, 4 cells) ==="
  putStrLn "    Starts on TorchDev TCpu (built F64, narrowed to F32"
  putStrLn "    via tcastUnsafe — see TODO 'Broaden runtime dtype"
  putStrLn "    coverage' for why primCreateFromHost is F64-only on"
  putStrLn "    torch today). Hops intra-torch (fast path via"
  putStrLn "    libtorch's `.to('mps')`) to TorchDev TMps, cross-"
  putStrLn "    backend to MlxDev MGpu, intra-mlx to MlxDev MCpu,"
  putStrLn "    back cross-backend to TorchDev TCpu."
  putStrLn ""

  -- Build F64 then narrow to F32 (exactly representable for these
  -- integer inputs). This sidesteps the primCreateFromHost dtype
  -- gap on the torch backend (always lands F64) — once the cascade
  -- threads dt through tensor_create_torch, makeVec4 {dt=F32} will
  -- work directly and this tcastUnsafe step can go.
  v_torch_cpu64 <- makeVec4 {d = TorchDev TCpu} {dt = F64} expected
  v_torch_cpu   <- tcastUnsafe F32 v_torch_cpu64
  ok1 <- reportStep "TorchDev TCpu (F32):"  (read4 v_torch_cpu)

  v_torch_mps <- toDevice (TorchDev TMps) v_torch_cpu
  ok2 <- reportStep "→ TorchDev TMps:"      (read4 v_torch_mps)

  v_mlx_gpu <- toDevice (MlxDev MGpu) v_torch_mps
  ok3 <- reportStep "→ MlxDev MGpu:"        (read4 v_mlx_gpu)

  v_mlx_cpu <- toDevice (MlxDev MCpu) v_mlx_gpu
  ok4 <- reportStep "→ MlxDev MCpu:"        (read4 v_mlx_cpu)

  v_back <- toDevice (TorchDev TCpu) v_mlx_cpu
  ok5 <- reportStep "→ TorchDev TCpu:"      (read4 v_back)

  pure (ok1 && ok2 && ok3 && ok4 && ok5)


----------------------------------------------------------------------
-- Main
----------------------------------------------------------------------

main : IO ()
main = do
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
