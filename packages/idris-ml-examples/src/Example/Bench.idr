-- | End-to-end training microbenchmark, on the v1 Nn/fit surface.
-- |
-- | Times warmup + a timed window of fwd+bwd+step epochs across the
-- | core model families (Linear, RNN, NTM copy/recall at several
-- | scales). Keeps manual warm/timed loops (not `fit`) for timing
-- | control. Loss values are not asserted — only wall time + peak RSS.

module Example.Bench

import Control.Linear.LIO
import Data.Linear.Notation
import Data.List
import Data.Vect
import System
import System.Clock

import BuildConfig
import Ml.Compat.Random
import Ml.Simple

----------------------------------------------------------------------
-- Timing
----------------------------------------------------------------------

elapsedMs : Clock Monotonic -> Clock Monotonic -> Double
elapsedMs t0 t1 =
  let s = cast {to=Double} (seconds t1 - seconds t0)
      ns = cast {to=Double} (nanoseconds t1 - nanoseconds t0)
  in s * 1000.0 + ns / 1000000.0

repeatEpoch : Nat -> (m -> IO (m, Double)) -> m -> Double -> IO (m, Double)
repeatEpoch Z _ m loss     = pure (m, loss)
repeatEpoch (S k) step m _ = do
  (m', loss') <- step m
  repeatEpoch k step m' loss'

----------------------------------------------------------------------
-- Shared random data helpers (raw values; tensors built fresh per epoch)
----------------------------------------------------------------------

randomInt : (lo, hi : Nat) -> IO Nat
randomInt lo hi = do
  n <- randomRIO (cast {to=Int32} (natToInteger lo), cast {to=Int32} (natToInteger hi))
  pure (fromInteger (cast {to=Integer} n))

randomBitVec : (w : Nat) -> IO (Vect w Double)
randomBitVec w = traverse (\_ => do b <- randomRIO (the Int32 0, 1)
                                    pure (if b == 1 then 1.0 else 0.0))
                          (Vect.replicate w ())

----------------------------------------------------------------------
-- Generic two-phase (NTM) loss — encode all input rows, decode targets.
----------------------------------------------------------------------

TwoPhaseSeq : (i, o : Nat) -> Type
TwoPhaseSeq i o = (List (Vect i Double), List (Vect o Double))

sumLosses : List (Tensor [] Ex F WithGrad) -> IO (Tensor [] Ex F WithGrad)
sumLosses []        = assert_total $ idris_crash "Bench.sumLosses: empty"
sumLosses (x :: xs) = go x xs
  where
    go : Tensor [] Ex F WithGrad -> List (Tensor [] Ex F WithGrad) -> IO (Tensor [] Ex F WithGrad)
    go acc []        = pure acc
    go acc (y :: ys) = do s <- tadd acc y; go s ys

-- Thread the (linear) cell through the encode phase, returning the advanced
-- cell beside a unit bang (encode outputs are unused).
encodeAll : {n, m, h, i, o : Nat} -> (1 _ : Ntm n m h i o Ex F WithGrad) ->
            List (Vect i Double) ->
            L IO {use = 1} (LPair (!* ()) (Ntm n m h i o Ex F WithGrad))
encodeAll cell []            = pure1 (MkBang () # cell)
encodeAll cell (row :: rest) = do
  x <- liftIO1 (retypeGrad <$> tensor {dims = [i]} (FromVect row))
  (MkBang _ # cell') <- recurStep cell x
  encodeAll cell' rest

-- Thread the (linear) cell through the decode phase, returning the per-step BCE
-- losses (ω tensors) beside the final cell.
decodeLosses : {n, m, h, i, o : Nat} -> (1 _ : Ntm n m h i o Ex F WithGrad) ->
               List (Vect o Double) ->
               L IO {use = 1} (LPair (!* (List (Tensor [] Ex F WithGrad)))
                                     (Ntm n m h i o Ex F WithGrad))
decodeLosses cell []             = pure1 (MkBang [] # cell)
decodeLosses cell (trow :: rest) = do
  z <- liftIO1 (retypeGrad <$> tensor {dims = [i]} (Const 0.0))
  (MkBang out # cell') <- recurStep cell z
  l <- liftIO1 $ do
         y <- retypeGrad <$> tensor {dims = [o]} (FromVect trow)
         tbceLoss out y
  (MkBang ls # cellF) <- decodeLosses cell' rest
  pure1 (MkBang (l :: ls) # cellF)

twoPhaseLoss : {n, m, h, i, o : Nat} -> Ntm n m h i o Ex F WithGrad ->
               TwoPhaseSeq i o -> IO (Tensor [] Ex F WithGrad)
twoPhaseLoss model (encIns, targs) = Control.Linear.LIO.run $ do
  (MkBang () # enc) <- encodeAll (recurReset model) encIns
  (MkBang ls # cellF) <- decodeLosses enc targs
  discard cellF
  liftIO1 $ do
    s <- sumLosses ls
    (1.0 / cast (length targs)) *: s

ntmEpoch : {n, m, h, i, o : Nat} -> Optimizer Ex -> List (TwoPhaseSeq i o) ->
           Ntm n m h i o Ex F WithGrad -> IO (Ntm n m h i o Ex F WithGrad, Double)
ntmEpoch opt batch model = do
  ls   <- traverse (twoPhaseLoss model) batch
  s    <- sumLosses ls
  mean <- (1.0 / cast (length batch)) *: s
  d    <- trainStep opt mean
  pure (model, d)

-- Copy-style two-phase sequence: input rows = data ++ [0] then a delimiter
-- row, target rows = the data rows. Generic over i = o + 1 isn't enforced;
-- we just emit `i`-wide input rows and `o`-wide target rows.
genCopyBatch : {i, o : Nat} -> (count, minLen, maxLen : Nat) -> IO (List (TwoPhaseSeq i o))
genCopyBatch Z _ _               = pure []
genCopyBatch (S k) minLen maxLen = do
  len <- randomInt minLen maxLen
  ins  <- sequence (List.replicate (S len) (randomBitVec i))
  outs <- sequence (List.replicate len (randomBitVec o))
  rest <- genCopyBatch k minLen maxLen
  pure ((ins, outs) :: rest)

----------------------------------------------------------------------
-- Supervised: Linear classifier, full-batch (b=5), batched NLL loss
----------------------------------------------------------------------

supIn : Vect 10 Double
supIn = [1.5, -2.7, -3.2, 4.1, 5.7, 0.0, -1.3, 8.8, 2.9, -1.4]

supTgt : Vect 15 Double
supTgt = [0,1,0, 0,1,0, 0,0,1, 0,1,0, 1,0,0]

supStep : Optimizer Ex -> Linear 2 3 Ex F WithGrad -> IO (Linear 2 3 Ex F WithGrad, Double)
supStep opt model = do
  x   <- tensor {dims=[5,2]} (FromVect supIn)
  tgt <- tensor {dims=[5,3]} (FromVect supTgt)
  out <- Control.Linear.LIO.run (do
           (MkBang o # m') <- forward {b=5} model (retypeGrad x)
           discard m'
           pure o)
  l   <- tnllLossMean {b=5} {n=3} out (retypeGrad tgt)
  d   <- trainStep opt l
  pure (model, d)

benchSupervised : IO ()
benchSupervised = do
  model <- runInit (linear {i=2} {o=3})
  opt <- sgd 0.03 defaultOpts
  (warmModel, _) <- repeatEpoch 100 (supStep opt) model 0.0
  t0 <- clockTime Monotonic
  (_, finalLoss) <- repeatEpoch 1000 (supStep opt) warmModel 0.0
  t1 <- clockTime Monotonic
  putStrLn $ "Supervised (1000 epochs): " ++ show (elapsedMs t0 t1) ++ " ms"
  putStrLn $ "  Final loss: " ++ show finalLoss
  putStrLn $ "  Peak RSS: " ++ show (getRssMB 0) ++ " MB"

----------------------------------------------------------------------
-- RNN (i=1, o=1, 8 fixed pattern sequences, BCE per timestep)
----------------------------------------------------------------------

patternSeq : Nat -> (List Double, List Double)
patternSeq len =
  let p = List.take (len + 1) (concat (List.replicate (len + 1) [0.0, 1.0, 0.0]))
  in (List.take len p, List.take len (List.drop 1 p))

rnnSeqs : Vect 8 (List Double, List Double)
rnnSeqs = map (patternSeq . (+ 3) . finToNat) (Data.Vect.Fin.range {len = 8})

rnnSeqLoss : Rnn 1 1 Ex F WithGrad -> (List Double, List Double) ->
             IO (Tensor [] Ex F WithGrad)
rnnSeqLoss cell0 (is, os) = Control.Linear.LIO.run $ do
  (MkBang (sumL, cnt) # cellF) <- go (recurReset cell0) Nothing 0 (zip is os)
  discard cellF
  liftIO1 (if cnt == 0 then pure sumL else (1.0 / cast cnt) *: sumL)
  where
    go : (1 _ : Rnn 1 1 Ex F WithGrad) -> Maybe (Tensor [] Ex F WithGrad) -> Nat ->
         List (Double, Double) ->
         L IO {use = 1} (LPair (!* (Tensor [] Ex F WithGrad, Nat)) (Rnn 1 1 Ex F WithGrad))
    go cell acc c [] = case acc of
      Just s  => pure1 (MkBang (s, c) # cell)
      Nothing => do discard cell
                    assert_total $ idris_crash "Bench.rnnSeqLoss: empty"
    go cell acc c ((xi, yi) :: rest) = do
      x          <- liftIO1 (retypeGrad <$> tensor {dims=[1]} (FromVect [xi]))
      (MkBang h # cell') <- recurStep cell x
      acc'       <- liftIO1 $ do
                      y <- retypeGrad <$> tensor {dims=[1]} (FromVect [yi])
                      l <- tbceLoss h y
                      case acc of Just s => Just <$> tadd s l; Nothing => pure (Just l)
      go cell' acc' (S c) rest

rnnEpoch : Optimizer Ex -> Rnn 1 1 Ex F WithGrad -> IO (Rnn 1 1 Ex F WithGrad, Double)
rnnEpoch opt model = do
  ls   <- traverse (rnnSeqLoss model) (toList rnnSeqs)
  s    <- sumLosses ls
  mean <- (1.0 / cast (the Nat 8)) *: s
  d    <- trainStep opt mean
  pure (model, d)

benchRnn : IO ()
benchRnn = do
  model <- runInit (rnn {i=1} {o=1} ttanh)
  opt <- sgd 0.03 defaultOpts
  (warmModel, _) <- repeatEpoch 100 (rnnEpoch opt) model 0.0
  t0 <- clockTime Monotonic
  (_, finalLoss) <- repeatEpoch 1000 (rnnEpoch opt) warmModel 0.0
  t1 <- clockTime Monotonic
  putStrLn $ "RNN (1000 epochs):        " ++ show (elapsedMs t0 t1) ++ " ms"
  putStrLn $ "  Final loss: " ++ show finalLoss
  putStrLn $ "  Peak RSS: " ++ show (getRssMB 1) ++ " MB"

----------------------------------------------------------------------
-- NTM small (n=10 m=5 h=20, copy, batch=5)
----------------------------------------------------------------------

benchNtm : IO ()
benchNtm = do
  model <- runInit (ntm {n=10} {m=5} {h=20} {i=4} {o=3})
  opt <- rmsprop 0.0001 {alpha=0.95} {momentum=0.0} ({ clip := NormClip 10.0 } defaultOpts)
  batch <- genCopyBatch {i=4} {o=3} 5 2 4
  (warmModel, _) <- repeatEpoch 10 (ntmEpoch opt batch) model 0.0
  t0 <- clockTime Monotonic
  (_, benchLoss) <- repeatEpoch 100 (ntmEpoch opt batch) warmModel 0.0
  t1 <- clockTime Monotonic
  putStrLn $ "NTM (100 epochs):         " ++ show (elapsedMs t0 t1) ++ " ms"
  putStrLn $ "  Final loss: " ++ show benchLoss
  putStrLn $ "  Peak RSS: " ++ show (getRssMB 2) ++ " MB"

----------------------------------------------------------------------
-- NTM copy production scale (n=128 m=20 h=100, batch=16)
----------------------------------------------------------------------

benchNtmCopy : IO ()
benchNtmCopy = do
  model <- runInit (ntm {n=128} {m=20} {h=100} {i=9} {o=8})
  opt <- rmsprop 0.0001 {alpha=0.95} {momentum=0.0} ({ clip := NormClip 10.0 } defaultOpts)
  batch <- genCopyBatch {i=9} {o=8} 16 1 20
  (warmModel, _) <- repeatEpoch 10 (ntmEpoch opt batch) model 0.0
  t0 <- clockTime Monotonic
  (_, benchLoss) <- repeatEpoch 100 (ntmEpoch opt batch) warmModel 0.0
  t1 <- clockTime Monotonic
  putStrLn $ "NTM-copy (100 epochs):    " ++ show (elapsedMs t0 t1) ++ " ms"
  putStrLn $ "  Final loss: " ++ show benchLoss
  putStrLn $ "  Peak RSS: " ++ show (getRssMB 3) ++ " MB"

----------------------------------------------------------------------
-- NTM copy 1k (fresh data + GC every 10 epochs, matching real training)
----------------------------------------------------------------------

copy1kLoop : Optimizer Ex -> Nat -> Nat -> Ntm 128 20 100 9 8 Ex F WithGrad -> Double ->
             IO (Ntm 128 20 100 9 8 Ex F WithGrad, Double)
copy1kLoop opt numEpochs remaining m loss =
  if remaining == 0 then pure (m, loss)
  else do
    batch <- genCopyBatch {i=9} {o=8} 16 1 20
    (m', loss') <- ntmEpoch opt batch m
    let idx = minus numEpochs remaining
    when (modNatNZ idx 10 ItIsSucc == 0) forceGC
    copy1kLoop opt numEpochs (minus remaining 1) m' loss'

benchNtmCopy1k : IO ()
benchNtmCopy1k = do
  model <- runInit (ntm {n=128} {m=20} {h=100} {i=9} {o=8})
  opt <- rmsprop 0.0001 {alpha=0.95} {momentum=0.9} ({ clip := NormClip 10.0 } defaultOpts)
  (warmModel, _) <- copy1kLoop opt 10 10 model 0.0
  forceGC
  t0 <- clockTime Monotonic
  (_, finalLoss) <- copy1kLoop opt 1000 1000 warmModel 0.0
  t1 <- clockTime Monotonic
  putStrLn $ "NTM-copy-1k (1000 epochs): " ++ show (elapsedMs t0 t1) ++ " ms"
  putStrLn $ "  Final loss: " ++ show finalLoss
  putStrLn $ "  Peak RSS: " ++ show (getRssMB 4) ++ " MB"

----------------------------------------------------------------------
-- NTM recall (n=128 m=20 h=100, i=8 o=6, batch=16)
----------------------------------------------------------------------

benchNtmRecall : IO ()
benchNtmRecall = do
  model <- runInit (ntm {n=128} {m=20} {h=100} {i=8} {o=6})
  opt <- rmsprop 0.0001 {alpha=0.95} {momentum=0.9} ({ clip := NormClip 10.0 } defaultOpts)
  batch <- genCopyBatch {i=8} {o=6} 16 2 6
  (warmModel, _) <- repeatEpoch 10 (ntmEpoch opt batch) model 0.0
  t0 <- clockTime Monotonic
  (_, benchLoss) <- repeatEpoch 100 (ntmEpoch opt batch) warmModel 0.0
  t1 <- clockTime Monotonic
  putStrLn $ "NTM-recall (100 epochs):  " ++ show (elapsedMs t0 t1) ++ " ms"
  putStrLn $ "  Final loss: " ++ show benchLoss
  putStrLn $ "  Peak RSS: " ++ show (getRssMB 5) ++ " MB"

----------------------------------------------------------------------
-- Main
----------------------------------------------------------------------

%default partial

main : IO ()
main = do
  requireMachine {m = ChosenMachine}
  srand 123456
  tsetInitSeed {ex = Ex} 123456
  args <- getArgs
  case drop 1 args of
    [] => do
      benchSupervised
      benchRnn
      benchNtm
      benchNtmCopy
      benchNtmCopy1k
      benchNtmRecall
    ["supervised"]  => benchSupervised
    ["rnn"]         => benchRnn
    ["ntm"]         => benchNtm
    ["ntm-copy"]    => benchNtmCopy
    ["ntm-copy-1k"] => benchNtmCopy1k
    ["ntm-recall"]  => benchNtmRecall
    other           => do
      putStrLn $ "unknown bench selector: " ++ show other
      putStrLn "valid: supervised | rnn | ntm | ntm-copy | ntm-copy-1k | ntm-recall"
      exitWith (ExitFailure 2)
