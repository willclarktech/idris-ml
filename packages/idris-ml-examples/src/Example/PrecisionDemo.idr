||| F32 / F64 precision artifact + cross-backend hop demo.
|||
||| Unblocked by tape's F32 storage + kernel coverage: every cell
||| this demo exercises — `TapeExecutor` (F32 + F64 trainable),
||| `TorchExecutor TCpu`, `MlxExecutor MCpu` — is now first-class for both
||| precisions. The example tells the precision story in three
||| numbered parts:
|||
|||   1. F64 → F32 narrowing on `TapeExecutor` (`tcastUnsafe`). Source
|||      values are deliberately not exactly representable in F32
|||      (π-, e-, √2-truncations), so the readback delta is
|||      non-zero AND inside the F32 epsilon band — proving the
|||      cast actually narrowed (lossy) without diverging.
|||
|||   2. F32 → F64 widening on `TapeExecutor` (`tcast` via the lossless
|||      `UpcastableTo F32 F64` instance). The widened readback is
|||      bit-for-bit identical to the F32 readback: F32 ⊂ F64, so
|||      promoting an F32 value into F64 storage adds no new
|||      precision; the original F64 source is gone.
|||
|||   3. Cross-backend F32 hop `TapeExecutor → TorchExecutor TCpu → MlxExecutor MCpu
|||      → TapeExecutor`. Each transition is a backendTag-mismatch host-
|||      buffer round-trip; F32 bits survive every hop because the
|||      F32-as-double host carrier is exact in both directions.
|||
||| Requires the multi-backend build so all three sets of C
||| symbols are linked:
|||
|||     make BACKEND=tape,torch,mlx example-precision-demo
|||
||| (The Makefile target forces this build internally — see the
||| `example-precision-demo` recipe.)
|||
||| Companion demo to `Example.PrecisionCheckpoint` (which exercises
||| the same dtype story over the *on-disk* SafeTensors round-trip)
||| and `Example.Transfer` (which exercises cross-backend hops on
||| exactly-representable F32 values, hiding the precision angle).
module Example.PrecisionDemo

import Data.List
import Data.Vect

import BuildConfig
import Executor
import Tensor

----------------------------------------------------------------------
-- Helpers
----------------------------------------------------------------------

||| 3-element host-buffer creation on the destination backend.
||| Same shape as `Example.Transfer.makeVec4` — every side-effecting
||| step goes through `primIO` so the FFI sequence is `%World`-
||| ordered. A let-chain would risk being hoisted to a CSE'd
||| module-level constant whose lambda still references buffers
||| allocated at module load, double-freeing on the next call.
makeVec3 : {0 ex : Type} -> {0 dt : DType} ->
           UserExecutorTransfer ex => Compatible ex dt => RuntimeDType dt =>
           (Double, Double, Double) ->
           IO (Tensor [3] ex dt WithGrad)
makeVec3 (a, b, c) = do
  buf  <- primIO (\w => MkIORes (primAllocHost   {ex} 3)        w)
  buf1 <- primIO (\w => MkIORes (prim__setDouble buf  0 a)     w)
  buf2 <- primIO (\w => MkIORes (prim__setDouble buf1 1 b)     w)
  buf3 <- primIO (\w => MkIORes (prim__setDouble buf2 2 c)     w)
  sh   <- primIO (\w => MkIORes (primAllocIntHost {ex} 1)       w)
  sh1  <- primIO (\w => MkIORes (primSetIntHost   {ex} sh 0 3)  w)
  ptr  <- primIO (\w =>
            MkIORes (primCreateFromHost {ex} buf3 sh1 1 1 (dtypeTag {t=dt})) w)
  primIO (primFreeIntHost {ex} sh1)
  primIO (primFreeHost    {ex} buf3)
  pure (MkTensor ptr Nothing)

||| Read all three values out via the backend's `primItem1d`.
||| Returns F64 doubles regardless of storage dtype — the C side
||| promotes F32 to double on readback. `{ex}` pins the typeclass
||| dispatch (Idris can't infer it from a bare `AnyPtr`).
read3 : {0 ex : Type} -> {0 dt : DType} -> UserExecutorCore ex =>
        Tensor [3] ex dt WithGrad ->
        (Double, Double, Double)
read3 t =
  ( primItem1d {ex} t.tensorPtr 0
  , primItem1d {ex} t.tensorPtr 1
  , primItem1d {ex} t.tensorPtr 2 )

||| F64 source values chosen to exhibit F32 precision artifacts —
||| transcendental-ish constants whose F32 nearest differs from
||| the F64 representation past the 7th decimal:
|||   3.14159265358979       — π;    F32 nearest 3.1415927410125732
|||   2.718281828459045      — e;    F32 nearest 2.7182817459106445
|||   1.4142135623730951     — √2;   F32 nearest 1.4142135381698608
sourceF64 : (Double, Double, Double)
sourceF64 = (3.14159265358979, 2.718281828459045, 1.4142135623730951)

showTriple : (Double, Double, Double) -> String
showTriple (a, b, c) =
  "[" ++ show a ++ ", " ++ show b ++ ", " ++ show c ++ "]"

maxAbsDelta : (Double, Double, Double) -> (Double, Double, Double) -> Double
maxAbsDelta (a, b, c) (d, e, f) =
  max (abs (a - d)) (max (abs (b - e)) (abs (c - f)))

||| Label-pad helper (local copy of the same trick `Transfer` uses
||| inline). Right-pads a short label so subsequent columns line up.
padN : Nat -> String -> String
padN n s =
  if length s >= n then s
  else s ++ pack (List.replicate (n `minus` length s) ' ')

||| F32 epsilon is ~1.19e-7 of magnitude. For values ~1..4 a
||| narrowing F64→F32 cast lands at most ~5e-7 from the F64
||| input — comfortably under 1e-6. Tight enough that a
||| catastrophic narrowing bug shows up; loose enough not to
||| flake on minor F32-rounding-mode differences.
f32RelTol : Double
f32RelTol = 1.0e-6

----------------------------------------------------------------------
-- Part 1: F64 → F32 narrowing (lossy) on TapeExecutor
----------------------------------------------------------------------

partOne_F32LossyCast : IO Bool
partOne_F32LossyCast = do
  putStrLn "=== Part 1: F64 → F32 narrowing on TapeExecutor ==="
  src <- makeVec3 {ex=TapeExecutor} {dt = F64} sourceF64
  let srcVals = read3 src
  putStrLn $ "  " ++ padN 22 "F64 source:"          ++ showTriple srcVals

  narrow <- tcastUnsafe F32 src
  let f32Vals = read3 narrow
  putStrLn $ "  " ++ padN 22 "→ tcastUnsafe F32:"   ++ showTriple f32Vals
  let dF32 = maxAbsDelta srcVals f32Vals
  putStrLn $ "  " ++ padN 22 "max abs delta:"       ++ show dF32

  let lossyArtifact = dF32 > 0.0           -- F32 cast actually narrowed
  let withinF32Eps  = dF32 < f32RelTol      -- but stays in F32 epsilon
  if lossyArtifact && withinF32Eps
    then do putStrLn "  → F32 lossy cast: OK"
            pure True
    else do putStrLn "  → F32 lossy cast: FAIL"
            pure False

----------------------------------------------------------------------
-- Part 2: F32 → F64 widening (lossless) via `tcast`
----------------------------------------------------------------------

partTwo_F32ToF64Upcast : IO Bool
partTwo_F32ToF64Upcast = do
  putStrLn ""
  putStrLn "=== Part 2: F32 → F64 widening on TapeExecutor ==="
  src    <- makeVec3 {ex=TapeExecutor} {dt = F64} sourceF64
  narrow <- tcastUnsafe F32 src
  let f32Vals = read3 narrow
  putStrLn $ "  " ++ padN 22 "F32 storage:"   ++ showTriple f32Vals

  -- `tcast` (not `tcastUnsafe`) because `UpcastableTo F32 F64` is in
  -- scope via the `LTE 32 64` instance in the Float family. The
  -- widened value-as-double is bit-for-bit identical to the F32
  -- readback: promoting F32 into F64 storage adds no new precision;
  -- the original F64 source is gone.
  widened <- tcast {to = F64} narrow
  let f64Vals = read3 widened
  putStrLn $ "  " ++ padN 22 "→ tcast F64:"   ++ showTriple f64Vals
  let dUpcast = maxAbsDelta f32Vals f64Vals
  putStrLn $ "  " ++ padN 22 "delta (F32→F64):" ++ show dUpcast

  if dUpcast == 0.0
    then do putStrLn "  → upcast: OK (exact, no further precision loss)"
            pure True
    else do putStrLn "  → upcast: FAIL (non-zero delta)"
            pure False

----------------------------------------------------------------------
-- Part 3: F32 cross-backend hop
----------------------------------------------------------------------

partThree_F32Hop : IO Bool
partThree_F32Hop = do
  putStrLn ""
  putStrLn "=== Part 3: F32 hop TapeExecutor → TorchExecutor TCpu → MlxExecutor MCpu → TapeExecutor ==="
  src_f64 <- makeVec3 {ex=TapeExecutor} {dt = F64} sourceF64
  v_tape  <- tcastUnsafe F32 src_f64
  let startVals = read3 v_tape
  putStrLn $ "  " ++ padN 30 "TapeExecutor F32:"          ++ showTriple startVals

  v_torch <- toExecutor (TorchExecutor TCpu) v_tape
  let torchVals = read3 v_torch
  putStrLn $ "  " ++ padN 30 "→ TorchExecutor TCpu F32:"  ++ showTriple torchVals

  v_mlx <- toExecutor (MlxExecutor MCpu) v_torch
  let mlxVals = read3 v_mlx
  putStrLn $ "  " ++ padN 30 "→ MlxExecutor MCpu F32:"    ++ showTriple mlxVals

  v_back <- toExecutor TapeExecutor v_mlx
  let backVals = read3 v_back
  putStrLn $ "  " ++ padN 30 "→ TapeExecutor F32 (back):" ++ showTriple backVals

  let totalDelta = maxAbsDelta startVals backVals
  putStrLn $ "  " ++ padN 30 "max delta start↔back:" ++ show totalDelta

  if totalDelta == 0.0
    then do putStrLn "  → hop: OK (F32 bits preserved across all 4 hops)"
            pure True
    else do putStrLn "  → hop: FAIL (drift across hops)"
            pure False

----------------------------------------------------------------------
-- Main
----------------------------------------------------------------------

main : IO ()
main = do
  requireMachine {m = ChosenMachine}
  putStrLn "=== PrecisionDemo: F32/F64 cast + cross-backend hop ==="
  putStrLn ""
  putStrLn "F32: 4 bytes/elem, ~7 significant decimal digits."
  putStrLn "F64: 8 bytes/elem, ~15-17 significant decimal digits."
  putStrLn "(BF16/F16 inference storage now also live on tape; not"
  putStrLn " exercised here — see Example.PrecisionCheckpoint for"
  putStrLn " the F32 ↔ F64 disk round-trip story.)"
  putStrLn ""

  ok1 <- partOne_F32LossyCast
  ok2 <- partTwo_F32ToF64Upcast
  ok3 <- partThree_F32Hop

  let overall = ok1 && ok2 && ok3
  putStrLn ""
  putStrLn $ "RESULT\tf32_lossy=" ++ (if ok1 then "ok" else "FAIL") ++
             "\tupcast=" ++ (if ok2 then "ok" else "FAIL") ++
             "\thop=" ++ (if ok3 then "ok" else "FAIL") ++
             "\toverall=" ++ (if overall then "ok" else "FAIL")
