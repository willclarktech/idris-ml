-- | LayersBench: single-layer forward+backward microbenchmark (Axis B).
-- |
-- | Companion to packages/backends/bench_ops.c (Axis A, op-kernel level).
-- | Axis B sits one rung up — instead of pure C-kernel timing, it
-- | exercises the FFI + tape wrap + autograd graph at the *Idris-side
-- | layer* granularity. Each section runs N fwd+bwd+step cycles on a
-- | single Nn layer and emits a one-line `<label>:\t<ms> ms (<iters>
-- | iters)` row that scripts/perf-fast.sh parses into a `kind: "op_bench"
-- | axis="B"` JSONL entry.
-- |
-- | Run:
-- |   make bench-layers
-- | The Tier-1 driver `scripts/perf-fast.sh` invokes this alongside the
-- | C-level Axis A benches.
-- |
-- | Selection (see docs/develop/testing-taxonomy.md): one entry per
-- | distinct compute pattern — Linear, LstmCell, Conv2dBlock, NtmHead,
-- | TransformerBlock.

module Example.LayersBench

import Control.Linear.LIO
import Data.Linear.Notation
import Data.List
import Data.String
import Data.Vect
import System
import System.Clock

import BuildConfig
import Ml.Compat.Random
import Ml.Simple

%default partial

----------------------------------------------------------------------
-- Timing + formatting helpers (copied from MatmulBench; intentional
-- example-level duplication per feedback_example_duplication.md)
----------------------------------------------------------------------

elapsedMs : Clock Monotonic -> Clock Monotonic -> Double
elapsedMs t0 t1 =
  let s  = cast {to=Double} (seconds t1 - seconds t0)
      ns = cast {to=Double} (nanoseconds t1 - nanoseconds t0)
  in s * 1000.0 + ns / 1000000.0

-- Three-decimal fixed-point so the regex in scripts/perf-fast.sh
-- (`([0-9.]+)\s*ms`) reads back a stable wall-clock figure.
fmt3 : Double -> String
fmt3 x =
  let scaled = the Integer (cast (x * 1000.0))
      whole = scaled `div` 1000
      frac  = abs (scaled `mod` 1000)
      fracS = if frac < 10 then "00" ++ show frac
                else if frac < 100 then "0" ++ show frac
                else show frac
  in show whole ++ "." ++ fracS

repeatEpoch : Nat -> (m -> IO (m, Double)) -> m -> Double -> IO (m, Double)
repeatEpoch Z _ m loss     = pure (m, loss)
repeatEpoch (S k) step m _ = do
  (m', loss') <- step m
  repeatEpoch k step m' loss'

----------------------------------------------------------------------
-- Linear (batch=32, in=512, out=512)
----------------------------------------------------------------------

LinearI : Nat
LinearI = 512

LinearO : Nat
LinearO = 512

LinearBatch : Nat
LinearBatch = 32

LinearIters : Nat
LinearIters = 100

LinearWarmup : Nat
LinearWarmup = 10

benchLinear : IO ()
benchLinear = do
  model <- runInit (linear {ex=Ex} {dt=F} {i=LinearI} {o=LinearO})
  opt <- sgd 0.01 defaultOpts
  let step = \m => do
        x   <- tensor {dims=[LinearBatch, LinearI]} (Const 0.1)
        tgt <- tensor {dims=[LinearBatch, LinearO]} (Const 0.1)
        out <- Control.Linear.LIO.run (do
                 (MkBang o # m') <- forward {b=LinearBatch} m (retypeGrad x)
                 discard m'
                 pure o)
        l   <- tnllLossMean {b=LinearBatch} {n=LinearO} out (retypeGrad tgt)
        d   <- trainStep opt l
        pure (m, d)
  (warmModel, _) <- repeatEpoch LinearWarmup step model 0.0
  t0 <- clockTime Monotonic
  _ <- repeatEpoch LinearIters step warmModel 0.0
  t1 <- clockTime Monotonic
  putStrLn $ "linear bs=32 i=512 o=512:\t" ++ fmt3 (elapsedMs t0 t1) ++ " ms\t("
          ++ show LinearIters ++ " iters)"

----------------------------------------------------------------------
-- LstmCell (hidden=256, unbatched, single timestep per iter)
----------------------------------------------------------------------

LstmI : Nat
LstmI = 256

LstmO : Nat
LstmO = 256

LstmIters : Nat
LstmIters = 100

LstmWarmup : Nat
LstmWarmup = 10

-- The LstmCell carries state in its record, so it must be threaded linearly
-- across timesteps. A local linear repeat loop consumes the model each step
-- and returns the advanced model beside the loss.
lstmStep : Optimizer Ex -> (1 _ : Lstm LstmI LstmO Ex F WithGrad) ->
           L IO {use = 1} (LPair (!* Double) (Lstm LstmI LstmO Ex F WithGrad))
lstmStep opt m = do
  inp        <- liftIO1 (retypeGrad <$> tensor {dims=[LstmI]} (Const 0.1))
  (MkBang h # m') <- recurStep m inp
  d          <- liftIO1 $ do
                  tgt <- retypeGrad <$> tensor {dims=[LstmO]} (Const 0.1)
                  l   <- tmseLoss h tgt
                  trainStep opt l
  pure1 (MkBang d # m')

repeatLstm : Optimizer Ex -> Nat -> (1 _ : Lstm LstmI LstmO Ex F WithGrad) ->
             L IO {use = 1} (Lstm LstmI LstmO Ex F WithGrad)
repeatLstm _   Z     m = pure1 m
repeatLstm opt (S k) m = do
  (MkBang _ # m') <- lstmStep opt m
  repeatLstm opt k m'

benchLstmCell : IO ()
benchLstmCell = Control.Linear.LIO.run $ do
  model <- runInitL (lstm {ex=Ex} {dt=F} {i=LstmI} {o=LstmO})
  opt <- liftIO1 (sgd 0.01 defaultOpts)
  warmModel <- repeatLstm opt LstmWarmup model
  t0 <- liftIO1 (clockTime Monotonic)
  timedModel <- repeatLstm opt LstmIters warmModel
  t1 <- liftIO1 (clockTime Monotonic)
  discard timedModel
  liftIO1 $ putStrLn $ "lstm_cell hidden=" ++ show LstmO ++ ":\t" ++ fmt3 (elapsedMs t0 t1) ++ " ms\t("
          ++ show LstmIters ++ " iters)"

----------------------------------------------------------------------
-- Conv2dBlock (batch=8, c_in=3, h=w=16, c_out=16, k=3)
----------------------------------------------------------------------

ConvInC : Nat
ConvInC = 3

ConvOutC : Nat
ConvOutC = 16

ConvH : Nat
ConvH = 16

ConvW : Nat
ConvW = 16

ConvKH : Nat
ConvKH = 3

ConvKW : Nat
ConvKW = 3

ConvInputDim : Nat
ConvInputDim = ConvInC * (ConvH * ConvW)        -- 3 * 16 * 16 = 768

ConvOH : Nat
ConvOH = ConvOutDim ConvH ConvKH 0              -- 14

ConvOW : Nat
ConvOW = ConvOutDim ConvW ConvKW 0              -- 14

ConvOutputDim : Nat
ConvOutputDim = ConvOutC * (ConvOH * ConvOW)    -- 16 * 14 * 14 = 3136

ConvBatch : Nat
ConvBatch = 8

ConvIters : Nat
ConvIters = 100

ConvWarmup : Nat
ConvWarmup = 10

benchConv2dBlock : IO ()
benchConv2dBlock = do
  model <- runInit (conv2d {ex=Ex} {dt=F} {inC=ConvInC} {outC=ConvOutC} {h=ConvH} {w=ConvW}
                           {kH=ConvKH} {kW=ConvKW} {padH=0} {padW=0})
  opt <- sgd 0.01 defaultOpts
  let step = \m => do
        x   <- tensor {dims=[ConvBatch, ConvInputDim]} (Const 0.1)
        tgt <- tensor {dims=[ConvBatch, ConvOutputDim]} (Const 0.1)
        out <- Control.Linear.LIO.run (do
                 (MkBang o # m') <- forward {b=ConvBatch} m (retypeGrad x)
                 discard m'
                 pure o)
        l   <- tnllLossMean {b=ConvBatch} {n=ConvOutputDim} out (retypeGrad tgt)
        d   <- trainStep opt l
        pure (m, d)
  (warmModel, _) <- repeatEpoch ConvWarmup step model 0.0
  t0 <- clockTime Monotonic
  _ <- repeatEpoch ConvIters step warmModel 0.0
  t1 <- clockTime Monotonic
  putStrLn $ "conv2d_block bs=" ++ show ConvBatch
          ++ " " ++ show ConvInC ++ "x" ++ show ConvH ++ "x" ++ show ConvW
          ++ "->" ++ show ConvOutC
          ++ " k=" ++ show ConvKH ++ "x" ++ show ConvKW
          ++ ":\t" ++ fmt3 (elapsedMs t0 t1) ++ " ms\t(" ++ show ConvIters ++ " iters)"

----------------------------------------------------------------------
-- Ntm (head + controller, tiny dims, two-phase copy task)
----------------------------------------------------------------------

NtmW : Nat
NtmW = 3

NtmInputW : Nat
NtmInputW = S NtmW

NtmOutputW : Nat
NtmOutputW = NtmW

NtmN : Nat
NtmN = 8

NtmM : Nat
NtmM = 4

NtmH : Nat
NtmH = 20

NtmIters : Nat
NtmIters = 30

NtmWarmup : Nat
NtmWarmup = 5

sumLosses : List (Tensor [] Ex F WithGrad) -> IO (Tensor [] Ex F WithGrad)
sumLosses []        = assert_total $ idris_crash "LayersBench.sumLosses: empty"
sumLosses (x :: xs) = go x xs
  where
    go : Tensor [] Ex F WithGrad -> List (Tensor [] Ex F WithGrad) -> IO (Tensor [] Ex F WithGrad)
    go acc []        = pure acc
    go acc (y :: ys) = do s <- tadd acc y; go s ys

ntmTwoPhaseStep : Optimizer Ex -> List (Vect NtmInputW Double) -> List (Vect NtmOutputW Double) ->
                  Ntm NtmN NtmM NtmH NtmInputW NtmOutputW Ex F WithGrad ->
                  IO (Ntm NtmN NtmM NtmH NtmInputW NtmOutputW Ex F WithGrad, Double)
ntmTwoPhaseStep opt encIns targs model = do
  mean <- Control.Linear.LIO.run $ do
            (MkBang () # enc) <- encodeAll (recurReset model) encIns
            (MkBang ls # cellF) <- decodeLosses enc targs
            discard cellF
            liftIO1 $ do
              s <- sumLosses ls
              (1.0 / cast (length targs)) *: s
  d <- trainStep opt mean
  pure (model, d)
  where
    encodeAll : (1 _ : Ntm NtmN NtmM NtmH NtmInputW NtmOutputW Ex F WithGrad) ->
                List (Vect NtmInputW Double) ->
                L IO {use = 1} (LPair (!* ())
                                      (Ntm NtmN NtmM NtmH NtmInputW NtmOutputW Ex F WithGrad))
    encodeAll cell []            = pure1 (MkBang () # cell)
    encodeAll cell (row :: rest) = do
      x <- liftIO1 (retypeGrad <$> tensor {dims=[NtmInputW]} (FromVect row))
      (MkBang _ # cell') <- recurStep cell x
      encodeAll cell' rest
    decodeLosses : (1 _ : Ntm NtmN NtmM NtmH NtmInputW NtmOutputW Ex F WithGrad) ->
                   List (Vect NtmOutputW Double) ->
                   L IO {use = 1} (LPair (!* (List (Tensor [] Ex F WithGrad)))
                                         (Ntm NtmN NtmM NtmH NtmInputW NtmOutputW Ex F WithGrad))
    decodeLosses cell []             = pure1 (MkBang [] # cell)
    decodeLosses cell (trow :: rest) = do
      z <- liftIO1 (retypeGrad <$> tensor {dims=[NtmInputW]} (Const 0.0))
      (MkBang out # cell') <- recurStep cell z
      l <- liftIO1 $ do
             y <- retypeGrad <$> tensor {dims=[NtmOutputW]} (FromVect trow)
             tbceLoss out y
      (MkBang ls # cellF) <- decodeLosses cell' rest
      pure1 (MkBang (l :: ls) # cellF)

benchNtmHead : IO ()
benchNtmHead = do
  model <- runInit (ntm {n=NtmN} {m=NtmM} {h=NtmH} {i=NtmInputW} {o=NtmOutputW})
  opt <- rmsprop 0.0001 {alpha=0.95} {momentum=0.0} ({ clip := NormClip 10.0 } defaultOpts)
  -- Fixed single-sequence copy task: 3 data rows + delimiter, 3 targets.
  let dataRows : List (Vect NtmW Double) = [[1,0,1], [0,1,1], [1,1,0]]
      encIns : List (Vect NtmInputW Double)
        = map (\r => r ++ [0.0]) dataRows ++ [Vect.replicate NtmW 0.0 ++ [1.0]]
      targs  : List (Vect NtmOutputW Double) = dataRows
  let step = ntmTwoPhaseStep opt encIns targs
  (warmModel, _) <- repeatEpoch NtmWarmup step model 0.0
  t0 <- clockTime Monotonic
  _ <- repeatEpoch NtmIters step warmModel 0.0
  t1 <- clockTime Monotonic
  putStrLn $ "ntm n=" ++ show NtmN ++ " m=" ++ show NtmM
          ++ " h=" ++ show NtmH ++ " batch=1"
          ++ ":\t" ++ fmt3 (elapsedMs t0 t1) ++ " ms\t(" ++ show NtmIters ++ " iters)"

----------------------------------------------------------------------
-- TransformerBlock (batch=2, dModel=64, heads=4, headDim=16)
--
-- The bare Nn.transformerBlock — the embedding + final-norm + vocab
-- projection are insignificant at these dims (and live alongside the
-- block in the decomposed surface, not inside it).
-- Captures the attention + FFN + 2× LayerNorm + 2× residual pattern.
----------------------------------------------------------------------

TxDModel : Nat
TxDModel = 64

TxHeads : Nat
TxHeads = 4

TxHeadDim : Nat
TxHeadDim = 16

TxBatch : Nat
TxBatch = 2

TxIters : Nat
TxIters = 50

TxWarmup : Nat
TxWarmup = 5

benchTransformerBlock : IO ()
benchTransformerBlock = do
  model <- runInit (transformerBlock {ex=Ex} {dt=F} {dModel=TxDModel} {numHeads=TxHeads} {headDim=TxHeadDim})
  opt <- sgd 0.01 defaultOpts
  let step = \m => do
        x   <- tensor {dims=[TxBatch, TxDModel]} (Const 0.1)
        tgt <- tensor {dims=[TxBatch, TxDModel]} (Const 0.1)
        out <- Control.Linear.LIO.run (do
                 (MkBang o # m') <- forward {b=TxBatch} m (retypeGrad x)
                 discard m'
                 pure o)
        l   <- tnllLossMean {b=TxBatch} {n=TxDModel} out (retypeGrad tgt)
        d   <- trainStep opt l
        pure (m, d)
  (warmModel, _) <- repeatEpoch TxWarmup step model 0.0
  t0 <- clockTime Monotonic
  _ <- repeatEpoch TxIters step warmModel 0.0
  t1 <- clockTime Monotonic
  putStrLn $ "transformer_block bs=" ++ show TxBatch
          ++ " d=" ++ show TxDModel
          ++ " heads=" ++ show TxHeads
          ++ ":\t" ++ fmt3 (elapsedMs t0 t1) ++ " ms\t(" ++ show TxIters ++ " iters)"

----------------------------------------------------------------------
-- Main
----------------------------------------------------------------------

main : IO ()
main = do
  requireMachine {m = ChosenMachine}
  putStrLn "--- Linear ---"
  benchLinear
  putStrLn ""
  putStrLn "--- LstmCell ---"
  benchLstmCell
  putStrLn ""
  putStrLn "--- Conv2dBlock ---"
  benchConv2dBlock
  putStrLn ""
  putStrLn "--- Ntm ---"
  benchNtmHead
  putStrLn ""
  putStrLn "--- TransformerBlock ---"
  benchTransformerBlock
  putStrLn ""
  putStrLn "=== Done ==="
