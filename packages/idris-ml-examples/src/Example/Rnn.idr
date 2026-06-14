module Example.Rnn

import Control.Linear.LIO
import Data.Linear.Notation
import Data.List
import Data.Vect
import System

import BuildConfig
import Checkpoint
import Compat.Random
import FitL
import ML.Simple
import Train

-- RNN pattern-prediction example. Single RNN(1 -> 4, tanh) -> Linear(4 -> 1)
-- network with BCE-with-logits loss, on the v1 Nn/fit surface. See
-- Example.Lstm for the recurrent migration shape (record of cell + head,
-- per-sequence hand-folded `recurStep` + 1-D `tlinear` head).

record Model where
  constructor MkModel
  cell : Rnn 1 4 Ex F WithGrad
  head : Linear 4 1 Ex F WithGrad

-- Top-level `Init` derivation (not an inline `do` inside `runInitL`): a
-- nested do-block under the linear `run $ do …` blows past the elaborator's
-- ambiguity-depth limit, so the model derivation lives here as a plain value.
mkModel : Init Model
mkModel = do
  cell <- rnn {i = 1} {o = 4} ttanh
  head <- linear {i = 4} {o = 1}
  pure (MkModel cell head)

----------------------------------------------------------------------
-- Pattern data
----------------------------------------------------------------------

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

sumLosses : List (Tensor [] Ex F WithGrad) -> IO (Tensor [] Ex F WithGrad)
sumLosses []        = assert_total $ idris_crash "Rnn.sumLosses: empty"
sumLosses (x :: xs) = go x xs
  where
    go : Tensor [] Ex F WithGrad -> List (Tensor [] Ex F WithGrad) ->
         IO (Tensor [] Ex F WithGrad)
    go acc []        = pure acc
    go acc (y :: ys) = do s <- tadd acc y; go s ys

seqLoss : Model -> (List Double, List Double) -> IO (Tensor [] Ex F WithGrad)
seqLoss (MkModel cell0 head) (is, os) = do
  (sumL, n) <- go (recurReset cell0) Nothing 0 (zip is os)
  if n == 0 then pure sumL else (1.0 / cast n) *: sumL
  where
    go : Rnn 1 4 Ex F WithGrad -> Maybe (Tensor [] Ex F WithGrad) -> Nat ->
         List (Double, Double) -> IO (Tensor [] Ex F WithGrad, Nat)
    go _ acc cnt [] = case acc of
      Just s  => pure (s, cnt)
      Nothing => assert_total $ idris_crash "Rnn.seqLoss: empty sequence"
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

-- Linear-resource epoch step (consume-match-rebuild-delegate): the model is
-- read once per sequence (8×), so match `MkModel cell head` to bind the
-- fields at their ω constructor quantity, reuse them freely in the IO loss
-- computation, then rebuild the model beside the banged loss. The
-- single-owner obligation is discharged by the one pattern-match.
recurEpochL : Optimizer Ex -> (1 _ : Model) -> Vect NumSeqs (List Double, List Double) ->
              L IO {use = 1} (LPair (!* Double) Model)
recurEpochL opt (MkModel cell head) seqs = do
  d <- liftIO1 $ do
         seqLs <- traverse (seqLoss (MkModel cell head)) (toList seqs)
         totalL <- sumLosses seqLs
         mean <- (1.0 / cast NumSeqs) *: totalL
         nativeTrainStep opt mean
  pure1 (MkBang d # MkModel cell head)

-- Consume the final (linear) model: it holds only C-managed param handles,
-- so dropping the matched ω fields is a no-op discharge (leaf-`discardL` shape).
discardModel : (1 _ : Model) -> L IO ()
discardModel (MkModel _ _) = pure ()

----------------------------------------------------------------------
-- Config & Main
----------------------------------------------------------------------

record Config where
  constructor MkConfig
  lr              : Double
  epochs          : Nat
  patience        : Nat
  seed            : Bits64
  checkpointDir   : String
  checkpointEvery : Nat

defaultConfig : Config
defaultConfig = MkConfig 0.5 2000 500 42 "" 200

specs : List (ArgSpec Config)
specs = [ Arg "--lr" (\v, c => { lr := cast v } c)
        , Arg "--epochs" (\v, c => { epochs := castNat v } c)
        , Arg "--patience" (\v, c => { patience := castNat v } c)
        , Arg "--seed" (\v, c => { seed := castBits64 v } c)
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

  putStrLn "=== RNN Pattern Prediction ==="
  putStrLn $ "Config: lr=" ++ show cfg.lr ++ " epochs=" ++ show cfg.epochs
           ++ " patience=" ++ show cfg.patience ++ " seed=" ++ show cfg.seed

  let trainCfgBase = patienceConfig cfg.epochs cfg.patience
      trainCfg = case cfg.checkpointDir of
                   ""  => trainCfgBase
                   dir => withCheckpoint
                            (fileCheckpoint dir cfg.checkpointEvery True opt)
                            trainCfgBase

  -- Linear surface end to end: model born linear (runInitL), threaded through
  -- fitL (recurEpochL consumes-and-returns it each epoch), final handle
  -- discarded. main : IO re-enters via run. `run` is fully qualified —
  -- `import System` (getArgs) brings other `run`s that otherwise blow the
  -- elaborator's ambiguity-depth limit in this do-block.
  Control.Linear.LIO.run $ do
    model <- runInitL mkModel
    liftIO1 (putStrLn "")
    (MkBang (epochsDone, finalLoss) # trained) <-
      fitL (recurEpochL opt) opt (generate (pure patternSeqs)) trainCfg model
    discardModel trained
    liftIO1 $ putStrLn ""
    liftIO1 $ putStrLn $ formatResult [ ("epochs", show epochsDone)
                                      , ("loss", show finalLoss)
                                      , ("seed", show cfg.seed)
                                      ]
