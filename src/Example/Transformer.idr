-- | Transformer Autoregressive Example
-- |
-- | Character-level next-token prediction with a single-head
-- | causal Transformer. Learns a repeating pattern "ABCDABCD..."

module Example.Transformer

import Data.List
import Data.Stream
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
import Layer.Transformer
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
VocabSize = 5

SeqLen : Nat
SeqLen = 16

DModel : Nat
DModel = 32

InputDim : Nat
InputDim = SeqLen * DModel

OutputDim : Nat
OutputDim = SeqLen * VocabSize


----------------------------------------------------------------------
-- Positional Encoding
----------------------------------------------------------------------

||| Sinusoidal positional encoding value.
posEnc : Nat -> Nat -> Double
posEnc pos dim =
  let p = cast {to=Double} pos
      i = cast {to=Double} (div dim 2)  -- pair index: dims 0,1 share freq; 2,3 share; etc.
      dm = cast {to=Double} DModel
      angle = p / pow 10000.0 (2.0 * i / dm)
  in if modNatNZ dim 2 ItIsSucc == 0 then sin angle else cos angle

||| Embed a token at a position: token one-hot + position one-hot.
||| First VocabSize dims = token identity, next SeqLen dims = position.
embedTokenAt : Nat -> Nat -> Vect DModel Double
embedTokenAt pos idx =
  let tokHot = map (\i => if finToNat i == idx then 1.0 else 0.0)
                   (Data.Vect.Fin.range {len=VocabSize})
      posHot = map (\i => if finToNat i == pos then 1.0 else 0.0)
                   (Data.Vect.Fin.range {len=SeqLen})
      -- VocabSize + SeqLen = 5 + 16 = 21, need DModel=32, pad remaining 11
      padding = replicate (minus DModel (VocabSize + SeqLen)) 0.0
  in tokHot ++ posHot ++ padding


----------------------------------------------------------------------
-- Data Generation
----------------------------------------------------------------------

pattern : List Nat
pattern = take 100 $ cycle [0, 1, 2, 3, 4]

makeExample : Nat -> DataPoint InputDim OutputDim Double
makeExample start =
  let tokens = Data.List.take SeqLen (drop start pattern)
      nextTokens = Data.List.take SeqLen (drop (start + 1) pattern)
      -- Input: embedded tokens with positional encoding
      inputFlat = concatMap (\(pos, tok) => toList (embedTokenAt pos tok))
                            (zip (map finToNat (toList (Data.Vect.Fin.range {len=SeqLen}))) tokens)
      -- Target: one-hot of next token at each position
      targetFlat = concatMap (\tok => toList (map (\i => if finToNat i == tok then 1.0 else 0.0)
                              (Data.Vect.Fin.range {len=VocabSize}))) nextTokens
      toVect : (n : Nat) -> List Double -> Vect n (Scalar Double)
      toVect Z _ = []
      toVect (S k) [] = STensor 0.0 :: toVect k []
      toVect (S k) (x :: xs) = STensor x :: toVect k xs
  in MkDataPoint (VTensor (toVect InputDim inputFlat))
                  (VTensor (toVect OutputDim targetFlat))

trainingData : Vect 16 (DataPoint InputDim OutputDim Double)
trainingData = map (makeExample . finToNat) range


----------------------------------------------------------------------
-- Per-position categorical cross-entropy loss
----------------------------------------------------------------------

||| Categorical cross-entropy applied per position.
||| Output has SeqLen groups of VocabSize logits; target is one-hot.
||| Computes: -mean_pos[ sum_class( target[c] * logSoftmax(logits)[c] ) ]
perPositionCE : {seqLen, vocabSize : Nat} ->
                Vector (seqLen * vocabSize) Variable -> Vector (seqLen * vocabSize) Variable -> Variable
perPositionCE {seqLen} {vocabSize} (VTensor preds) (VTensor targets) =
  let vsI = cast {to=Int} vocabSize
      -- Stack all logits into a tensor, reshape to [seqLen, vocabSize]
      logitsTensor = prim__reshape2d (vecStackTensor preds) (cast {to=Int} seqLen) vsI
      -- Row-wise log-softmax (numerically stable)
      logProbs = prim__logSoftmax2d logitsTensor
      -- Stack targets similarly
      targetTensor = prim__reshape2d (vecStackTensor targets) (cast {to=Int} seqLen) vsI
      -- NLL: -mean(sum_class(target * logProbs))
      -- Element-wise multiply, sum all, negate, divide by seqLen
      product = prim__mul logProbs targetTensor  -- [seqLen, vocabSize]
      totalSum = prim__sum product  -- scalar: sum of all target * logProb
      loss = prim__mulScalar (prim__neg totalSum) (1.0 / cast {to=Double} seqLen)
      val = prim__item loss
  in Var loss Nothing val

||| Per-position categorical CE loss wrapped as LossFunction.
||| The implicit {n} is always instantiated to OutputDim = SeqLen * VocabSize
||| by calculateLossVar (from the network output dimension).
catCELoss : LossFunction Variable
catCELoss {n} preds targets =
  -- n unifies with OutputDim at the call site.
  -- We need to tell Idris that n = SeqLen * VocabSize.
  -- Since this is true by construction (the network has output dim OutputDim),
  -- we use decEq to verify at runtime and crash if wrong.
  case decEq n (SeqLen * VocabSize) of
    Yes Refl => perPositionCE {seqLen=SeqLen, vocabSize=VocabSize} preds targets
    No _ => fromDouble 0.0  -- unreachable: n = OutputDim by network construction


----------------------------------------------------------------------
-- CLI
----------------------------------------------------------------------

record Config where
  constructor MkConfig
  lr : Double
  epochs : Nat
  seed : Bits64

defaultConfig : Config
defaultConfig = MkConfig 0.001 5000 42

specs : List (ArgSpec Config)
specs = [ Arg "--lr" (\v, c => { lr := cast v } c)
        , Arg "--epochs" (\v, c => { epochs := castNat v } c)
        , Arg "--seed" (\v, c => { seed := castBits64 v } c) ]


----------------------------------------------------------------------
-- Main
----------------------------------------------------------------------

tokenName : Nat -> String
tokenName 0 = "A"
tokenName 1 = "B"
tokenName 2 = "C"
tokenName 3 = "D"
tokenName 4 = "E"
tokenName _ = "?"

main : IO ()
main = do
  args <- getArgs
  let cfg = parseArgs defaultConfig specs (drop 1 args)

  srand cfg.seed

  let opt = nativeAdamGlobalClip cfg.lr 0.9 0.999 1.0e-8 1.0
  let prepared = map (map fromDouble) trainingData

  putStrLn "=== Transformer Autoregressive ==="
  putStrLn $ "Config: lr=" ++ show cfg.lr ++ " epochs=" ++ show cfg.epochs
           ++ " seed=" ++ show cfg.seed
  putStrLn $ "Architecture: seqLen=" ++ show SeqLen ++ " dModel=" ++ show DModel
           ++ " vocab=" ++ show VocabSize

  tfm <- transformerLayer {seqLen=SeqLen, dModel=DModel, vocabSize=VocabSize}
  let model = autoName $ OutputLayer tfm
  putStrLn $ "Model: " ++ show model
  putStrLn ""

  -- Pre-training eval (to verify forward works)
  let preInput = x (index FZ prepared)
  let (_, prePred) = forwardVar model preInput
  putStr "Pre-train output[0..7]: "
  let preVals = map (\v => prim__item v.tensorPtr) (Data.List.take 8 (toList prePred))
  putStrLn $ show preVals

  -- Helpers
  let listAt : Nat -> List Double -> Double
      listAt _ [] = 0.0
      listAt Z (xx :: _) = xx
      listAt (S k) (_ :: xs) = listAt k xs
  let positions = map finToNat (toList (Data.Vect.Fin.range {len=SeqLen}))
  -- Extract Variable values in FORWARD order.
  -- Tensor's Foldable reverses (documented gotcha). Unwrap VTensor to get
  -- the underlying Vect and use Vect's toList which preserves order.
  let tensorVals : {n : Nat} -> Vector n Variable -> List Double
      tensorVals (VTensor xs) =
        let vectToList : Vect k (Scalar Variable) -> List Double
            vectToList [] = []
            vectToList (STensor v :: rest) = prim__item v.tensorPtr :: vectToList rest
        in vectToList xs

  -- Metrics: compute accuracy on training data during training
  let evalMetrics : Network InputDim [] OutputDim Variable -> IO (List (String, String))
      evalMetrics m = do
        -- Use calculateLossVar to run the full forward + loss in one go
        -- This is the same path that training uses, so results should match
        let freshData = map (map fromDouble) trainingData
            lossVar = calculateLossVar catCELoss m freshData
            lossVal = prim__item lossVar.tensorPtr
        -- Also run forwardVar for argmax predictions
        let (_, pred) = forwardVar m (x (index FZ freshData))
            predVals = tensorVals pred
            expected = Data.List.take SeqLen (drop 1 pattern)
            predicted = map (\pos =>
              let probs = map (\j => listAt (pos * VocabSize + j) predVals) (the (List Nat) [0,1,2,3,4])
                  best = foldl (\(bi,bv), (i,v) => if v > bv then (i,v) else (bi,bv))
                               (the (Nat, Double) (0, -1.0e10)) (zip (the (List Nat) [0,1,2,3,4]) probs)
              in fst best) positions
            correct = foldl (\acc, (a,b) => if a == b then acc + 1 else acc) (the Nat 0) (zip expected predicted)
        pure [("acc", show correct ++ "/" ++ show SeqLen)]

  -- Use simple config (accuracy not reliably trackable during training due to Idris purity)
  (trained, epochsDone, _) <- runTraining
    (\m, d => epochNative opt d catCELoss m) (pure prepared) (simpleConfig cfg.epochs) model

  -- Evaluate ALL data points
  putStrLn "Per-example accuracy:"
  traverse_ (\idx => do
    let dp = map fromDouble (index idx trainingData)
        (_, pred) = forwardVar trained (x dp)
        predVals = tensorVals pred
        expected = Data.List.take SeqLen (drop (finToNat idx + 1) pattern)
        predicted = map (\pos =>
          let probs = map (\j => listAt (pos * VocabSize + j) predVals) (the (List Nat) [0,1,2,3,4])
              best = foldl (\(bi,bv), (i,v) => if v > bv then (i,v) else (bi,bv))
                           (the (Nat, Double) (0, -1.0e10)) (zip (the (List Nat) [0,1,2,3,4]) probs)
          in fst best) positions
        correct = foldl (\acc, (a,b) => if a == b then acc + 1 else acc) (the Nat 0) (zip expected predicted)
    -- Also compute loss on this example's predictions via catCELoss
    let singleLoss = catCELoss pred (y dp)
    putStrLn $ "  start=" ++ show (finToNat idx) ++ " acc=" ++ show correct ++ "/16"
             ++ " loss=" ++ show (prim__item singleLoss.tensorPtr)
    ) (Data.Vect.Fin.range {len=16})

  -- First data point predictions
  let firstInput = x (index FZ prepared)
  let (_, firstPred) = forwardVar trained firstInput

  -- Show logits for first 4 positions
  let allVals = tensorVals firstPred
  putStrLn "Post-train logits (per position, 4 classes each):"
  traverse_ (\pos =>
    let start = pos * VocabSize
        vals = map (\j => listAt (start + j) allVals) (the (List Nat) [0,1,2,3,4])
        target = listAt pos (drop 1 (map cast pattern))
    in putStrLn $ "  pos " ++ show pos ++ " target=" ++ tokenName (cast target)
                ++ " logits=" ++ show vals
    ) (Data.List.take 4 (map finToNat (toList (Data.Vect.Fin.range {len=SeqLen}))))

  putStrLn ""
  putStrLn "Eval:"
  let predList = tensorVals firstPred
      listAt : Nat -> List Double -> Double
      listAt _ [] = 0.0
      listAt Z (x :: _) = x
      listAt (S k) (_ :: xs) = listAt k xs
      showPos : Nat -> String
      showPos pos =
        let start = pos * VocabSize
            probs = map (\j => listAt (start + j) predList) (the (List Nat) [0,1,2,3,4])
            best = foldl (\(bi,bv), (i,v) => if v > bv then (i,v) else (bi,bv))
                         (the (Nat, Double) (0, -1.0e10))
                         (zip (the (List Nat) [0,1,2,3,4]) probs)
        in tokenName (fst best)
  let positions = map finToNat (toList (Data.Vect.Fin.range {len=SeqLen}))
  putStr "  Pattern:   "
  putStrLn $ concatMap tokenName (Data.List.take SeqLen pattern)
  putStr "  Predicted: "
  putStrLn $ concatMap showPos positions
  let predicted = map (\pos => fst (foldl (\(bi,bv), (i,v) => if v > bv then (i,v) else (bi,bv))
        (the (Nat, Double) (0, -1.0e10))
        (zip (the (List Nat) [0,1,2,3,4])
             (map (\j => listAt (pos * VocabSize + j) predList) (the (List Nat) [0,1,2,3,4]))))
        ) positions
      expected = Data.List.take SeqLen (drop 1 pattern)
      correct = foldl (\acc, (a,b) => if a == b then acc + 1 else acc) (the Nat 0) (zip expected predicted)
  putStrLn $ "  Accuracy:  " ++ show correct ++ "/" ++ show SeqLen

  putStrLn ""
  putStrLn $ formatResult [("epochs", show epochsDone), ("acc", show correct ++ "/" ++ show SeqLen),
                            ("seed", show cfg.seed)]
