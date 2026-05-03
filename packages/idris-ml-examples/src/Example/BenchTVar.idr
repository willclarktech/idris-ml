module Example.BenchTVar

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
import Layer.CoreV2
import Layer.LinearV2
import Math
import Sampler
import Tensor
import Train
import Util
import Variable


-- Path C P3-1 spike: micro-benchmark.
--
-- Compares per-iteration cost of:
-- (a) the scalar Linear path (existing `Layer/Linear.idr`, with `applyVarTensor`
--     fast path active after `autoName`), driven via the existing `forwardVarTensor`
--     pipeline.
-- (b) the new `LinearV2` path (rank-aware `TVar`).
--
-- Both call the same backend `prim__mv` + `prim__add`. The diff is the Idris-side
-- packaging cost. The spike's claim is that the typed surface ≥ existing fast path.


----------------------------------------------------------------------
-- Helpers
----------------------------------------------------------------------

-- Build a Vector n Double of small random values (uniform [-0.1, 0.1])
-- packed into a persistent C buffer for input.
randomInputBuf : (n : Nat) -> IO AnyPtr
randomInputBuf n = do
  vals <- traverse (\_ => randomRIO (-0.1, 0.1)) (Vect.replicate n ())
  let nI = cast {to=Int} n
      buf = prim__allocDoubles nI
      buf' = packInto buf 0 vals
  pure (prim__createState1d nI buf')
  where
    packInto : AnyPtr -> Int -> Vect k Double -> AnyPtr
    packInto b _ [] = b
    packInto b o (x :: rest) = packInto (prim__setDouble b o x) (o + 1) rest


----------------------------------------------------------------------
-- TVar bench: LinearV2 forward + backward + optimizer step
----------------------------------------------------------------------

-- Run N iterations of (forward, MSE-vs-zero, train step) on a single
-- LinearV2 i o. The "loss against zeros" forces a backward through the
-- same path the supervised example uses.
benchTVar : {i, o : Nat} -> NativeOptimizer -> AnyPtr -> AnyPtr ->
            LinearStateV2 i o CPU -> Nat -> IO Double
benchTVar _ _ _ _ Z = pure 0.0
benchTVar opt inputT zerosT model (S k) = do
  let inputTV = tinput1d {n = i} inputT
      zerosTV = tinput1d {n = o} zerosT
      (_, predTV) = applyTVar model inputTV
      lossTV = tmseLoss predTV zerosTV
      l = nativeTrainStepTVar opt lossTV
  if k == 0 then pure l else benchTVar opt inputT zerosT model k


----------------------------------------------------------------------
-- Scalar bench: existing `Layer/Linear.idr` via forwardVarTensor + nllLoss
----------------------------------------------------------------------

scalarMseLoss : {n : Nat} -> AnyPtr -> AnyPtr -> Variable CPU
scalarMseLoss predT targetT =
  let diff   = prim__sub predT targetT
      sq     = prim__mul diff diff
      loss   = prim__sum sq
      val    = prim__item loss
  in Var loss Nothing val

benchScalar : {i, o : Nat} -> NativeOptimizer -> AnyPtr -> AnyPtr ->
              Network i [] o (Variable CPU) -> Nat -> IO Double
benchScalar _ _ _ _ Z = pure 0.0
benchScalar opt inputT zerosT model (S k) = do
  let (_, predT) = forwardVarTensor model inputT
      loss = scalarMseLoss {n = o} predT zerosT
      l = nativeTrainStep opt loss
  if k == 0 then pure l else benchScalar opt inputT zerosT model k


----------------------------------------------------------------------
-- Driver
----------------------------------------------------------------------

timeMs : IO () -> IO Double
timeMs action = do
  start <- clockTime Process
  action
  end <- clockTime Process
  let ns = timeDifference end start
  pure (cast (toNano ns) / 1.0e6)

runBench : (label : String) -> (iters : Nat) -> IO Double -> IO ()
runBench label iters action = do
  ms <- timeMs (do _ <- action; pure ())
  let perIter = ms / cast iters
  putStrLn $ "  " ++ label ++ ": " ++ show ms ++ " ms (" ++ show perIter ++ " ms/iter, " ++ show iters ++ " iters)"


-- Hard-coded 256x256 + 256 iterations (the spike's micro target).
--
-- We use {i = 256} {o = 256} so dims are statically known. Idris's
-- elaboration of `Vect 65536` for a Linear weight matrix may be slow
-- to compile; that's a one-time cost, not a runtime cost.

%default partial

main : IO ()
main = do
  args <- getArgs
  let iters : Nat
      iters = case drop 1 args of
                (s :: _) => castNat s
                _ => 200

  putStrLn "=== Path C P3-1 micro-bench (Linear 256x256) ==="
  putStrLn $ "iters=" ++ show iters
  putStrLn ""

  -- Shared persistent buffers
  inputT <- randomInputBuf 256
  zerosT <- randomInputBuf 256

  -- TVar path
  let lr = 0.001
  putStrLn "TVar path:"
  let optV2 = nativeSgd lr
  modelV2 <- linearLayerV2 {i = 256} {o = 256} "v2bench"
  -- Warm up (pay first-call FFI cost)
  _ <- benchTVar optV2 inputT zerosT modelV2 5
  msV2 <- timeMs (do _ <- benchTVar optV2 inputT zerosT modelV2 iters; pure ())
  putStrLn $ "  total:    " ++ show msV2 ++ " ms"
  putStrLn $ "  per iter: " ++ show (msV2 / cast iters) ++ " ms"
  putStrLn ""

  -- Scalar path
  putStrLn "Scalar path (existing Linear with applyVarTensor fast path):"
  let optScalar = nativeSgd lr
  ll <- linearLayer {i = 256} {o = 256} {ty = Variable CPU}
  let scalarModel = autoNamePrefix "scalar_" $ OutputLayer ll
  -- Warm up
  _ <- benchScalar optScalar inputT zerosT scalarModel 5
  msScalar <- timeMs (do _ <- benchScalar optScalar inputT zerosT scalarModel iters; pure ())
  putStrLn $ "  total:    " ++ show msScalar ++ " ms"
  putStrLn $ "  per iter: " ++ show (msScalar / cast iters) ++ " ms"
  putStrLn ""

  let speedup = msScalar / msV2
  putStrLn $ "TVar / Scalar speedup: " ++ show speedup ++ "x"
  putStrLn ""
  putStrLn $ formatResult [ ("iters", show iters)
                          , ("tvar_ms_per_iter", show (msV2 / cast iters))
                          , ("scalar_ms_per_iter", show (msScalar / cast iters))
                          , ("speedup", show speedup)
                          ]
