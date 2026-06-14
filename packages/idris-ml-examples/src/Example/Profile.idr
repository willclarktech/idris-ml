-- | NTM Copy forward-pass profiler, on the v1 Nn/fit surface.
-- |
-- | Times the two-phase copy-task epoch (encode all input rows into
-- | memory, decode the copied rows back) on an LSTM-controller NTM,
-- | and dumps the backend per-op profile. Keeps a manual warmup +
-- | timed loop (not `fit`) for fine-grained timing control.

module Example.Profile

import Data.List
import Data.Vect
import System
import System.Clock

import BuildConfig
import Compat.Random
import ML.Simple

----------------------------------------------------------------------
-- Timing
----------------------------------------------------------------------

elapsedMs : Clock Monotonic -> Clock Monotonic -> Double
elapsedMs t0 t1 =
  let s = cast {to=Double} (seconds t1 - seconds t0)
      ns = cast {to=Double} (nanoseconds t1 - nanoseconds t0)
  in s * 1000.0 + ns / 1000000.0

padL : Nat -> String -> String
padL n s = pack (replicate (minus n (length s)) ' ') ++ s

showMs : Double -> String
showMs d =
  let whole = cast {to=Integer} d
      frac = cast {to=Integer} (abs ((d - cast whole) * 10))
  in show whole ++ "." ++ show frac

fmtMs : Double -> String
fmtMs d = padL 10 (showMs d)

----------------------------------------------------------------------
-- NTM dims (matches NtmCopy.idr)
----------------------------------------------------------------------

W : Nat
W = 8

InputW : Nat
InputW = S W

OutputW : Nat
OutputW = W

N : Nat
N = 128

M : Nat
M = 20

H : Nat
H = 100

BatchSize : Nat
BatchSize = 16

Model : Type
Model = Ntm N M H InputW OutputW Ex F WithGrad

----------------------------------------------------------------------
-- Copy-task data (raw values; device tensors built fresh per epoch)
----------------------------------------------------------------------

Seq : Type
Seq = (List (Vect InputW Double), List (Vect OutputW Double))

randomInt : (lo, hi : Nat) -> IO Nat
randomInt lo hi = do
  n <- randomRIO (cast {to=Int32} (natToInteger lo), cast {to=Int32} (natToInteger hi))
  pure (fromInteger (cast {to=Integer} n))

randomBitVec : (w : Nat) -> IO (Vect w Double)
randomBitVec w = traverse (\_ => do b <- randomRIO (the Int32 0, 1)
                                    pure (if b == 1 then 1.0 else 0.0))
                          (Vect.replicate w ())

genCopySeq : (seqLen : Nat) -> IO Seq
genCopySeq seqLen = do
  dataRows <- sequence (List.replicate seqLen (randomBitVec W))
  let inputRows = map (\r => r ++ [0.0]) dataRows ++ [Vect.replicate W 0.0 ++ [1.0]]
  pure (inputRows, dataRows)

genBatch : (n, minLen, maxLen : Nat) -> IO (List Seq)
genBatch Z _ _               = pure []
genBatch (S k) minLen maxLen = do
  len <- randomInt minLen maxLen
  dp <- genCopySeq len
  rest <- genBatch k minLen maxLen
  pure (dp :: rest)

----------------------------------------------------------------------
-- Two-phase loss (identical shape to NtmCopy)
----------------------------------------------------------------------

zeroIn : IO (Tensor [InputW] Ex F WithGrad)
zeroIn = retypeGrad <$> tensor {dims = [InputW]} (Const 0.0)

sumLosses : List (Tensor [] Ex F WithGrad) -> IO (Tensor [] Ex F WithGrad)
sumLosses []        = assert_total $ idris_crash "Profile.sumLosses: empty"
sumLosses (x :: xs) = go x xs
  where
    go : Tensor [] Ex F WithGrad -> List (Tensor [] Ex F WithGrad) -> IO (Tensor [] Ex F WithGrad)
    go acc []        = pure acc
    go acc (y :: ys) = do s <- tadd acc y; go s ys

encodeAll : Model -> List (Vect InputW Double) -> IO Model
encodeAll cell []            = pure cell
encodeAll cell (row :: rest) = do
  x <- retypeGrad <$> tensor {dims = [InputW]} (FromVect row)
  (cell', _) <- recurStep cell x
  encodeAll cell' rest

decodeLosses : Model -> List (Vect OutputW Double) -> IO (List (Tensor [] Ex F WithGrad))
decodeLosses _ []                = pure []
decodeLosses cell (trow :: rest) = do
  z <- zeroIn
  (cell', out) <- recurStep cell z
  y <- retypeGrad <$> tensor {dims = [OutputW]} (FromVect trow)
  l <- tbceLoss out y
  ls <- decodeLosses cell' rest
  pure (l :: ls)

twoPhaseLoss : Model -> Seq -> IO (Tensor [] Ex F WithGrad)
twoPhaseLoss model (encIns, targs) = do
  enc <- encodeAll (recurReset model) encIns
  ls  <- decodeLosses enc targs
  s   <- sumLosses ls
  (1.0 / cast (length targs)) *: s

recurEpoch : Optimizer Ex -> Model -> List Seq -> IO (Model, Double)
recurEpoch opt model batch = do
  ls   <- traverse (twoPhaseLoss model) batch
  s    <- sumLosses ls
  mean <- (1.0 / cast (length batch)) *: s
  d    <- nativeTrainStep opt mean
  pure (model, d)

----------------------------------------------------------------------
-- Profile loop
----------------------------------------------------------------------

profileLoop : Optimizer Ex -> List Seq -> Model -> Nat -> Nat -> IO Model
profileLoop opt batch model cur count =
  if cur >= count
    then pure model
    else do
      t0 <- clockTime Monotonic
      (model', lossVal) <- recurEpoch opt model batch
      t1 <- clockTime Monotonic
      putStrLn $ padL 5 (show (cur + 1)) ++ fmtMs (elapsedMs t0 t1) ++ "    " ++ show lossVal
      profileLoop opt batch model' (cur + 1) count

warmup : Optimizer Ex -> Model -> Nat -> IO Model
warmup _ m 0       = pure m
warmup opt m (S k) = do
  batch <- genBatch BatchSize 1 20
  (m', loss) <- recurEpoch opt m batch
  putStrLn $ "  warmup " ++ show (5 `minus` k) ++ ": loss=" ++ show loss
  warmup opt m' k

----------------------------------------------------------------------
-- Main
----------------------------------------------------------------------

%default partial

main : IO ()
main = do
  requireMachine {m = ChosenMachine}
  srand 123456
  tsetInitSeed {ex = Ex} 123456

  putStrLn "=== NTM Copy Forward-Pass Profile ==="
  putStrLn $ "Architecture: N=" ++ show N ++ " M=" ++ show M ++ " H=" ++ show H
  putStrLn $ "Batch=" ++ show BatchSize ++ " seqLen=1-20"
  putStrLn ""

  model <- runInit (ntm {n = N} {m = M} {h = H} {i = InputW} {o = OutputW})
  opt <- rmsprop 0.0001 {alpha = 0.95} {momentum = 0.9}
                 ({ clip := NormClip 10.0 } defaultOpts)

  tGen0 <- clockTime Monotonic
  dataPoints <- genBatch BatchSize 1 20
  tGen1 <- clockTime Monotonic
  putStrLn $ "Data generation: " ++ showMs (elapsedMs tGen0 tGen1) ++ " ms"
  putStrLn ""

  putStrLn "Warmup (5 epochs)..."
  warmModel <- warmup opt model 5
  putStrLn ""

  putStrLn $ padL 5 "Epoch" ++ padL 10 "Total(ms)" ++ "    Loss"
  profileReset {ex=Ex}
  _ <- profileLoop opt dataPoints warmModel 0 10

  putStrLn ""
  profileReport {ex=Ex}
  putStrLn "Done."
