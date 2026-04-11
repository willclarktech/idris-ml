-- | Multi-Head Transformer Sequence Reversal Example
-- |
-- | Sequence reversal with learned embeddings, sinusoidal positional
-- | encoding, multi-head causal self-attention, and layer normalization.
-- |
-- | Input (teacher-forced): [t0, t1, ..., t4, SEP, t4, ..., t0, EOS]
-- | Target: predict next token at each position.

module Example.MultiHeadTransformer

import Data.List
import Data.Vect
import Decidable.Equality
import System
import System.Random

import Backprop
import DataPoint
import Endofunctor
import Floating
import Layer
import Layer.Core
import Layer.MultiHeadTransformer
import Math
import Optimizer
import Tensor
import Train
import Util
import Variable


----------------------------------------------------------------------
-- Configuration
----------------------------------------------------------------------

VocabSize : Nat
VocabSize = 10         -- 8 content tokens (A-H) + SEP + EOS

InputLen : Nat
InputLen = 5           -- tokens before separator

-- Full sequence: [t0..t4, SEP, t4..t0, EOS] = 12 tokens
-- Teacher forcing: input = first 11, target = last 11
-- So seqLen for the model = 11
SeqLen : Nat
SeqLen = 2 * InputLen + 1

DModel : Nat
DModel = 32

NumHeads : Nat
NumHeads = 4

HeadDim : Nat
HeadDim = 8

SepToken : Nat
SepToken = 8

EosToken : Nat
EosToken = 9

InputDim : Nat
InputDim = SeqLen * VocabSize

OutputDim : Nat
OutputDim = SeqLen * VocabSize


----------------------------------------------------------------------
-- Data Generation
----------------------------------------------------------------------

||| Generate one reversal training example.
||| Sequence: [t0, t1, t2, t3, t4, SEP, t4, t3, t2, t1, t0, EOS]
||| Input (teacher-forced): first 11 tokens, one-hot encoded.
||| Target: tokens 1..11, one-hot encoded.
makeReversalExample : IO (DataPoint InputDim OutputDim Double)
makeReversalExample = do
  -- Generate random content tokens (0..7)
  tokens <- sequence (replicate InputLen (randomRIO (the Int32 0, 7)))
  let tokNats = map (cast {to=Nat}) tokens
      revToks = reverse tokNats
      -- Full sequence: tokens ++ [SEP] ++ reversed ++ [EOS]
      fullSeq = tokNats ++ [SepToken] ++ revToks ++ [EosToken]
      -- Input: first SeqLen tokens (drop last)
      inputToks = Data.List.take SeqLen fullSeq
      -- Target: last SeqLen tokens (drop first)
      targetToks = Data.List.take SeqLen (drop 1 fullSeq)
      -- One-hot encode
      oneHot : Nat -> List Double
      oneHot tok = map (\i => if i == tok then 1.0 else 0.0)
                       (map finToNat (toList (Data.Vect.Fin.range {len=VocabSize})))
      inputFlat = concatMap oneHot inputToks
      targetFlat = concatMap oneHot targetToks
      toVect : (n : Nat) -> List Double -> Vect n (Scalar Double)
      toVect Z _ = []
      toVect (S k) [] = STensor 0.0 :: toVect k []
      toVect (S k) (x :: xs) = STensor x :: toVect k xs
  pure $ MkDataPoint (VTensor (toVect InputDim inputFlat))
                     (VTensor (toVect OutputDim targetFlat))


----------------------------------------------------------------------
-- Per-position categorical cross-entropy loss
----------------------------------------------------------------------

||| Categorical cross-entropy applied per position.
perPositionCE : {seqLen, vocabSize : Nat} ->
                Vector (seqLen * vocabSize) Variable -> Vector (seqLen * vocabSize) Variable -> Variable
perPositionCE {seqLen} {vocabSize} (VTensor preds) (VTensor targets) =
  let vsI = cast {to=Int} vocabSize
      logitsTensor = prim__reshape2d (vecStackTensor preds) (cast {to=Int} seqLen) vsI
      logProbs = prim__logSoftmax2d logitsTensor
      targetTensor = prim__reshape2d (vecStackTensor targets) (cast {to=Int} seqLen) vsI
      product = prim__mul logProbs targetTensor
      totalSum = prim__sum product
      loss = prim__mulScalar (prim__neg totalSum) (1.0 / cast {to=Double} seqLen)
      val = prim__item loss
  in Var loss Nothing val

catCELoss : LossFunction Variable
catCELoss {n} preds targets =
  case decEq n (SeqLen * VocabSize) of
    Yes Refl => perPositionCE {seqLen=SeqLen, vocabSize=VocabSize} preds targets
    No _ => fromDouble 0.0  -- unreachable


----------------------------------------------------------------------
-- CLI
----------------------------------------------------------------------

record Config where
  constructor MkConfig
  lr : Double
  epochs : Nat
  seed : Bits64

defaultConfig : Config
defaultConfig = MkConfig 0.001 3000 42

specs : List (ArgSpec Config)
specs = [ Arg "--lr" (\v, c => { lr := cast v } c)
        , Arg "--epochs" (\v, c => { epochs := castNat v } c)
        , Arg "--seed" (\v, c => { seed := castBits64 v } c) ]


----------------------------------------------------------------------
-- Main
----------------------------------------------------------------------

tokenName : Nat -> String
tokenName n = if n < 8
  then strCons (cast (cast {to=Int} n + 65)) ""  -- A=0, B=1, ...
  else if n == SepToken then "|"
  else if n == EosToken then "$"
  else "?"

main : IO ()
main = do
  args <- getArgs
  let cfg = parseArgs defaultConfig specs (drop 1 args)

  srand cfg.seed

  let opt = nativeAdamGlobalClip cfg.lr 0.9 0.999 1.0e-8 1.0

  putStrLn "=== Multi-Head Transformer: Sequence Reversal ==="
  putStrLn $ "Config: lr=" ++ show cfg.lr ++ " epochs=" ++ show cfg.epochs
           ++ " seed=" ++ show cfg.seed
  putStrLn $ "Architecture: seqLen=" ++ show SeqLen ++ " dModel=" ++ show DModel
           ++ " heads=" ++ show NumHeads ++ " headDim=" ++ show HeadDim
           ++ " vocab=" ++ show VocabSize

  -- Generate training data
  trainData <- sequence (replicate 16 makeReversalExample)
  let prepared = map (map fromDouble) trainData

  mht <- mhTransformerLayer {seqLen=SeqLen, dModel=DModel, numHeads=NumHeads,
                              headDim=HeadDim, vocabSize=VocabSize}
  let model = autoName $ OutputLayer mht
  putStrLn $ "Model: " ++ show model
  putStrLn ""

  -- Helpers for evaluation
  let listAt : Nat -> List Double -> Double
      listAt _ [] = 0.0
      listAt Z (xx :: _) = xx
      listAt (S k) (_ :: xs) = listAt k xs
  let positions = map finToNat (toList (Data.Vect.Fin.range {len=SeqLen}))
  let tensorVals : {n : Nat} -> Vector n Variable -> List Double
      tensorVals (VTensor xs) =
        let vectToList : Vect k (Scalar Variable) -> List Double
            vectToList [] = []
            vectToList (STensor v :: rest) = prim__item v.tensorPtr :: vectToList rest
        in vectToList xs
  let argmaxAt : List Double -> Nat -> Nat
      argmaxAt vals pos =
        let probs = map (\j => listAt (pos * VocabSize + j) vals)
                        (map finToNat (toList (Data.Vect.Fin.range {len=VocabSize})))
            best = foldl (\(bi,bv), (i,v) => if v > bv then (i,v) else (bi,bv))
                         (the (Nat, Double) (0, -1.0e10))
                         (zip (map finToNat (toList (Data.Vect.Fin.range {len=VocabSize}))) probs)
        in fst best

  -- Training
  (trained, epochsDone, _) <- runTraining
    (\m, d => epochNative opt d catCELoss m) (pure prepared) (simpleConfig cfg.epochs) model

  -- Evaluate on first example
  putStrLn ""
  putStrLn "Evaluation on first example:"
  let firstInput = x (index FZ prepared)
  let (_, firstPred) = forwardVar trained firstInput
  let predVals = tensorVals firstPred
      inputVals = tensorVals (x (index FZ prepared))
      targetVals = tensorVals (y (index FZ prepared))
      inputDecoded = map (argmaxAt inputVals) positions
      targetDecoded = map (argmaxAt targetVals) positions
      predicted = map (argmaxAt predVals) positions
      correct = foldl (\acc, (a,b) => if a == b then acc + 1 else acc) (the Nat 0) (zip predicted targetDecoded)
      -- Reversal portion accuracy (positions InputLen onwards in target)
      revPredicted = drop InputLen predicted
      revTarget = drop InputLen targetDecoded
      revCorrect = foldl (\acc, (a,b) => if a == b then acc + 1 else acc) (the Nat 0) (zip revPredicted revTarget)
      revTotal = length revPredicted

  putStr "  Input:      "
  putStrLn $ concatMap tokenName inputDecoded
  putStr "  Target:     "
  putStrLn $ concatMap tokenName targetDecoded
  putStr "  Predicted:  "
  putStrLn $ concatMap tokenName predicted
  putStrLn $ "  Full acc:   " ++ show correct ++ "/" ++ show SeqLen
  putStrLn $ "  Rev acc:    " ++ show revCorrect ++ "/" ++ show revTotal

  putStrLn ""
  putStrLn $ formatResult [("epochs", show epochsDone),
                            ("full_acc", show correct ++ "/" ++ show SeqLen),
                            ("rev_acc", show revCorrect ++ "/" ++ show revTotal),
                            ("seed", show cfg.seed)]
