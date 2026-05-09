module Example.BenchLstmBoundary

import Data.List
import Data.Vect
import System
import System.Clock
import Compat.Random

import DataPoint
import Device
import Endofunctor
import Floating
import Init
import Layer
import Layer.Lstm
import Layer.Linear
import Math
import Sampler
import Tensor
import Train
import Util
import Variable


-- Path C P3-1.5: boundary-cost calibration.
--
-- Compares per-iteration cost of:
-- (A) `forwardVarTensor` on existing LstmState (production fast path)
-- (B) `forwardVar` on existing LstmState (forces `applyVar` Vect-pack/unpack)
--
-- B/A is the upper bound on what Path C migration could save for callers
-- that go through `forwardVar` (e.g. `epochNative`, `epochRecurrentNative`,
-- `epochTwoPhaseBceNative`).
--
-- The audit showed only 1/28 examples (Profile.idr — a profiler, not
-- production) uses those scalarising epoch runners. Result here calibrates
-- whether even Profile.idr would benefit.


----------------------------------------------------------------------
-- Helpers
----------------------------------------------------------------------

randomTensor1d : (n : Nat) -> IO AnyPtr
randomTensor1d n = do
  vals <- traverse (\_ => randomRIO (-0.1, 0.1)) (Vect.replicate n ())
  let nI = cast {to=Int} n
      buf = prim__allocDoubles nI
      buf' = packInto buf 0 vals
  pure (prim__createState1d nI buf')
  where
    packInto : AnyPtr -> Int -> Vect k Double -> AnyPtr
    packInto b _ [] = b
    packInto b o (x :: rest) = packInto (prim__setDouble b o x) (o + 1) rest

-- Build a Vector n (Variable CPU) from a tensor handle (uses prim__select per element,
-- mirroring how `forwardVar` ingests data).
tensorToVarVect : (n : Nat) -> AnyPtr -> Vector n (Variable CPU)
tensorToVarVect n t = VTensor (tensorToScalars t 0 n)


----------------------------------------------------------------------
-- Path A: forwardVarTensor (no Vect boundary)
----------------------------------------------------------------------

benchPathA : {i, o : Nat} -> NativeOptimizer -> AnyPtr -> AnyPtr ->
             Network i [] o (Variable CPU) -> Nat -> IO Double
benchPathA _ _ _ _ Z = pure 0.0
benchPathA opt inputT zerosT model (S k) = do
  let (_, predT) = forwardVarTensor model inputT
      diff   = prim__sub predT zerosT
      sqDiff = prim__mul diff diff
      lossPtr = prim__sum sqDiff
      loss : Variable CPU
      loss = Var lossPtr Nothing (prim__item lossPtr)
      l = nativeTrainStep opt loss
  if k == 0 then pure (prim__item lossPtr) else benchPathA opt inputT zerosT model k


----------------------------------------------------------------------
-- Path B: forwardVar (Vect-pack at every applyVar call)
----------------------------------------------------------------------

benchPathB : {i, o : Nat} -> NativeOptimizer ->
             Vector i (Variable CPU) -> AnyPtr ->
             Network i [] o (Variable CPU) -> Nat -> IO Double
benchPathB _ _ _ _ Z = pure 0.0
benchPathB opt inputV zerosT model (S k) = do
  let (_, predV) = forwardVar model inputV
      (VTensor predElems) = predV
      predT = vecStackTensor predElems
      diff   = prim__sub predT zerosT
      sqDiff = prim__mul diff diff
      lossPtr = prim__sum sqDiff
      loss : Variable CPU
      loss = Var lossPtr Nothing (prim__item lossPtr)
      l = nativeTrainStep opt loss
  if k == 0 then pure (prim__item lossPtr) else benchPathB opt inputV zerosT model k


----------------------------------------------------------------------
-- Driver
----------------------------------------------------------------------

timeMs : IO () -> IO Double
timeMs action = do
  start <- clockTime Process
  action
  end <- clockTime Process
  pure (cast (toNano (timeDifference end start)) / 1.0e6)

%default partial

-- Fixed sizes — three points on the boundary-cost curve.
-- We pick distinct (i, o) pairs since boundary cost is i + o per cell.
runOne : (i : Nat) -> (o : Nat) -> (iters : Nat) -> IO ()
runOne i o iters = do
  putStrLn $ "--- LSTM(" ++ show i ++ " -> " ++ show o ++ "), " ++ show iters ++ " iters ---"
  inputT <- randomTensor1d i
  zerosT <- randomTensor1d o

  -- Build the LSTM via existing constructor
  lstm <- mkLstm {i} {o}
  let model = autoNamePrefix ("bench_" ++ show i ++ "_" ++ show o ++ "_") $ OutputLayer (MkAnyLayer LstmState lstm)

  let opt = nativeSgd 0.0001  -- tiny lr to avoid divergence

  -- Path A warmup + bench
  _ <- benchPathA opt inputT zerosT model 5
  msA <- timeMs (do _ <- benchPathA opt inputT zerosT model iters; pure ())

  -- Path B warmup + bench (use a fresh Vect-of-Variables input via tensorToScalars)
  let inputV : Vector i (Variable CPU)
      inputV = tensorToVarVect i inputT
  _ <- benchPathB opt inputV zerosT model 5
  msB <- timeMs (do _ <- benchPathB opt inputV zerosT model iters; pure ())

  let perIterA = msA / cast iters
      perIterB = msB / cast iters
      ratio = msB / msA
  putStrLn $ "  A (forwardVarTensor): " ++ show perIterA ++ " ms/iter"
  putStrLn $ "  B (forwardVar):       " ++ show perIterB ++ " ms/iter"
  putStrLn $ "  B/A:                  " ++ show ratio ++ "x"
  putStrLn ""

main : IO ()
main = do
  args <- getArgs
  let iters : Nat
      iters = case drop 1 args of
                (s :: _) => castNat s
                _ => 200

  putStrLn "=== Path C P3-1.5 boundary-cost bench ==="
  putStrLn $ "iters=" ++ show iters
  putStrLn ""

  -- Three sizes on the boundary cost curve
  runOne 1   4   iters
  runOne 64  64  iters
  runOne 256 256 iters

  putStrLn "Done."
