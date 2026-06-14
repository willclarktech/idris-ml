module Example.Lstm

import Data.List
import Data.Vect
import System

import BuildConfig
import Checkpoint
import Compat.Random
import ML.Simple
import Train

-- LSTM pattern-prediction example. Single LSTM(1 -> 4) -> Linear(4 -> 1)
-- network with BCE-with-logits loss, on the v1 Nn/fit surface.
--
-- The model is a plain record of an `Nn.Lstm` cell (Recurrent) + an
-- `Nn.Linear` head (Module). Recurrent layers aren't batched Modules, so
-- the per-sequence loss folds `recurStep` over the timesteps by hand and
-- applies the head with the 1-D `tlinear`, exactly as `Nn.Recurrent`'s
-- `recurStep` does internally.

record Model where
  constructor MkModel
  cell : Lstm 1 4 Ex F WithGrad
  head : Linear 4 1 Ex F WithGrad

----------------------------------------------------------------------
-- Pattern data (was Generate.patternData / RecurrentDataPoint)
----------------------------------------------------------------------

-- Deterministic [0,1,0,0,1,0,...] cycle. Sequence k has length k+3;
-- input = pattern, target = next element. Kept as a pure
-- (List Double, List Double): fresh device tensors are built per epoch
-- inside `seqLoss`, sidestepping the persistent-handle use-after-free
-- (collation/backward frees a batch's source tensors each epoch).
patternSeq : Nat -> (List Double, List Double)
patternSeq len =
  let p = List.take (len + 1) (concat (List.replicate (len + 1) [0.0, 1.0, 0.0]))
  in (List.take len p, List.take len (List.drop 1 p))

NumSeqs : Nat
NumSeqs = 8

patternSeqs : Vect NumSeqs (List Double, List Double)
patternSeqs = map (patternSeq . (+ 3) . finToNat) (Data.Vect.Fin.range {len = NumSeqs})

----------------------------------------------------------------------
-- Loss
----------------------------------------------------------------------

scalar1 : Double -> IO (Tensor [1] Ex F WithGrad)
scalar1 x = retypeGrad <$> tensor {dims = [1]} (FromVect [x])

-- Sum a non-empty list of scalar losses (the empty arm is unreachable —
-- every sequence and every epoch has at least one element).
sumLosses : List (Tensor [] Ex F WithGrad) -> IO (Tensor [] Ex F WithGrad)
sumLosses [] = assert_total $ idris_crash "Lstm.sumLosses: empty"
sumLosses (x :: xs) = go x xs
  where
    go : Tensor [] Ex F WithGrad -> List (Tensor [] Ex F WithGrad) ->
         IO (Tensor [] Ex F WithGrad)
    go acc []        = pure acc
    go acc (y :: ys) = do s <- tadd acc y; go s ys

-- Per-sequence loss: reset state, fold timesteps threading the cell,
-- apply the head with 1-D `tlinear`, BCE-with-logits per step, mean.
seqLoss : Model -> (List Double, List Double) -> IO (Tensor [] Ex F WithGrad)
seqLoss (MkModel cell0 head) (is, os) = do
  (sumL, n) <- go (recurReset cell0) Nothing 0 (zip is os)
  if n == 0 then pure sumL else (1.0 / cast n) *: sumL
  where
    go : Lstm 1 4 Ex F WithGrad -> Maybe (Tensor [] Ex F WithGrad) -> Nat ->
         List (Double, Double) -> IO (Tensor [] Ex F WithGrad, Nat)
    go _ acc cnt [] = case acc of
      Just s  => pure (s, cnt)
      Nothing => assert_total $ idris_crash "Lstm.seqLoss: empty sequence"
    go cell acc cnt ((xi, yi) :: rest) = do
      x          <- scalar1 xi
      (cell', h) <- recurStep cell x
      out        <- tlinear head.weightT h head.biasT
      y          <- scalar1 yi
      l          <- tbceLoss out y
      acc'       <- case acc of
                      Just s  => Just <$> tadd s l
                      Nothing => pure (Just l)
      go cell' acc' (S cnt) rest

-- One epoch = one pass over all NumSeqs sequences: mean per-seq loss, one
-- optimizer step (the legacy `epochRecurrentVar` shape, as an EpochStep).
recurEpoch : Optimizer Ex -> Model -> Vect NumSeqs (List Double, List Double) -> IO (Model, Double)
recurEpoch opt model seqs = do
  seqLs <- traverse (seqLoss model) (toList seqs)
  totalL <- sumLosses seqLs
  mean <- (1.0 / cast NumSeqs) *: totalL
  d <- nativeTrainStep opt mean
  pure (model, d)

----------------------------------------------------------------------
-- Config & Main
----------------------------------------------------------------------

record Config where
  constructor MkConfig
  lr : Double
  epochs : Nat
  patience : Nat
  seed : Bits64
  checkpointDir : String
  checkpointEvery : Nat

defaultConfig : Config
defaultConfig = MkConfig 0.5 2000 500 42 "" 200

specs : List (ArgSpec Config)
specs = [ Arg "--lr" (\v, c => { lr := cast v } c)
        , Arg "--epochs" (\v, c => { epochs := castNat v } c)
        , Arg "--patience" (\v, c => { patience := castNat v } c)
        , Arg "--seed" (\v, c => { seed := castBits64 v } c)
        -- Checkpointing: save to / auto-resume from DIR. `--resume` is an
        -- alias for `--checkpoint-dir` (resumes if DIR/last.* is present).
        , Arg "--checkpoint-dir" (\v, c => { checkpointDir := v } c)
        , Arg "--resume" (\v, c => { checkpointDir := v } c)
        , Arg "--checkpoint-every" (\v, c => { checkpointEvery := castNat v } c)
        ]

%default partial

main : IO ()
main = do
  requireMachine {m = ChosenMachine}
  args <- getArgs
  let cfg = parseArgs defaultConfig specs (drop 1 args)

  srand cfg.seed
  tsetInitSeed {ex = Ex} cfg.seed

  opt <- sgd cfg.lr defaultOpts

  putStrLn "=== LSTM Pattern Prediction ==="
  putStrLn $ "Config: lr=" ++ show cfg.lr ++ " epochs=" ++ show cfg.epochs
           ++ " patience=" ++ show cfg.patience ++ " seed=" ++ show cfg.seed

  model <- runInit $ do
    cell <- lstm {i = 1} {o = 4}
    head <- linear {i = 4} {o = 1}
    pure (MkModel cell head)
  putStrLn ""

  let trainCfgBase = patienceConfig cfg.epochs cfg.patience
      trainCfg = case cfg.checkpointDir of
                   "" => trainCfgBase
                   dir => withCheckpoint
                            (fileCheckpoint dir cfg.checkpointEvery True opt)
                            trainCfgBase

  (_, epochsDone, finalLoss) <-
    fit (recurEpoch opt) opt (generate (pure patternSeqs)) trainCfg model

  putStrLn ""
  putStrLn $ formatResult [ ("epochs", show epochsDone)
                          , ("loss", show finalLoss)
                          , ("seed", show cfg.seed)
                          ]
