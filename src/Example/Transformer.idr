-- | Transformer Autoregressive Example
-- |
-- | Character-level next-token prediction with a single-head
-- | causal Transformer. Learns a repeating pattern "ABCDABCD..."
-- | and generates continuations autoregressively.

module Example.Transformer

import Data.List
import Data.Stream
import Data.Vect
import System
import System.Clock
import System.Random

import Backprop
import DataPoint
import Endofunctor
import Floating
import Layer
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

-- Vocabulary: 4 symbols (A=0, B=1, C=2, D=3)
VocabSize : Nat
VocabSize = 4

-- Sequence length (context window)
SeqLen : Nat
SeqLen = 16

-- Model dimension
DModel : Nat
DModel = 32

-- Output: predict next token at each position -> SeqLen * VocabSize logits
-- But we use a simpler approach: Transformer output is SeqLen * DModel,
-- then a final linear projects each position to VocabSize.

-- Total network: Transformer(SeqLen*DModel -> SeqLen*DModel) -> Linear(SeqLen*DModel -> SeqLen*VocabSize)
InputDim : Nat
InputDim = SeqLen * DModel

OutputDim : Nat
OutputDim = SeqLen * VocabSize


----------------------------------------------------------------------
-- Data Generation
----------------------------------------------------------------------

-- Pattern: ABCDABCD... (repeating)
pattern : List Nat
pattern = take 100 $ cycle [0, 1, 2, 3]

-- One-hot encode a token (index -> VocabSize vector)
oneHot : Nat -> Vect VocabSize Double
oneHot idx = map (\i => if finToNat i == idx then 1.0 else 0.0) range

-- Create training data: input = one-hot encoded sequence, target = shifted by 1
-- Input: tokens 0..SeqLen-1 one-hot encoded -> Vector (SeqLen * DModel) Double
-- For simplicity, we embed tokens as one-hot padded to DModel
-- (first VocabSize dims are one-hot, rest are 0)
embedToken : Nat -> Vect DModel Double
embedToken idx =
  let hot = oneHot idx
      padding = replicate (minus DModel VocabSize) 0.0
  in hot ++ padding

-- Create one training example: input tokens at positions [start..start+SeqLen-1],
-- target is the next token at each position (teacher forcing)
makeExample : Nat -> DataPoint InputDim OutputDim Double
makeExample start =
  let tokens = take SeqLen (drop start pattern)
      nextTokens = take SeqLen (drop (start + 1) pattern)
      -- Input: concatenation of embedded tokens
      inputVecs = map embedToken tokens
      inputFlat = concatMap toList inputVecs
      -- Target: one-hot of next token at each position, flattened
      targetVecs = map oneHot nextTokens
      targetFlat = concatMap toList targetVecs
      toVect : (n : Nat) -> List Double -> Vect n (Scalar Double)
      toVect Z _ = []
      toVect (S k) [] = STensor 0.0 :: toVect k []
      toVect (S k) (x :: xs) = STensor x :: toVect k xs
  in MkDataPoint (VTensor (toVect InputDim inputFlat))
                  (VTensor (toVect OutputDim targetFlat))

-- Generate batch of examples from different starting positions
trainingData : Vect 8 (DataPoint InputDim OutputDim Double)
trainingData = map (makeExample . finToNat) range


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
tokenName _ = "?"

main : IO ()
main = do
  args <- getArgs
  let cfg = parseArgs defaultConfig specs (drop 1 args)

  srand cfg.seed

  let lossFn = crossEntropy
  let opt = nativeSgd cfg.lr
  let prepared = map (map fromDouble) trainingData

  putStrLn "=== Transformer Autoregressive ==="
  putStrLn $ "Config: lr=" ++ show cfg.lr ++ " epochs=" ++ show cfg.epochs
           ++ " seed=" ++ show cfg.seed
  putStrLn $ "Architecture: seqLen=" ++ show SeqLen ++ " dModel=" ++ show DModel
           ++ " vocab=" ++ show VocabSize

  -- Build: Transformer -> Linear output projection
  tfm <- transformerLayer {seqLen=SeqLen, dModel=DModel}
  ll <- linearLayer {i=InputDim, o=OutputDim}
  let model = autoName $ tfm ~> OutputLayer ll
  putStrLn $ "Model: " ++ show model
  putStrLn ""

  (trained, epochsDone, _) <- runTraining
    (\m, d => epochNative opt d lossFn m) (pure prepared)
    (simpleConfig cfg.epochs) model

  -- Eval: show predictions for one sequence
  let dblModel = toDoubleNetwork (emap refreshValue trained)
  let dblData = map (map fromDouble) trainingData
  let preds = evaluate dblModel dblData
  let loss = calculateLoss lossFn dblModel dblData

  putStrLn ""
  putStrLn "Eval:"
  putStrLn $ "  Loss: " ++ show loss

  -- Show first sequence: at each position, show predicted token
  let firstPred = index FZ preds
  putStr "  Pattern:   "
  putStrLn $ concatMap tokenName (Data.List.take SeqLen pattern)
  putStr "  Predicted: "
  -- Extract argmax at each position from the flattened output
  let predList = toList firstPred
      listAt : Nat -> List Double -> Double
      listAt _ [] = 0.0
      listAt Z (x :: _) = x
      listAt (S k) (_ :: xs) = listAt k xs
      showPos : Nat -> String
      showPos pos =
        let start = pos * VocabSize
            probs = map (\j => listAt (start + j) predList) (the (List Nat) [0,1,2,3])
            best = foldl (\(bi,bv), (i,v) => if v > bv then (i,v) else (bi,bv))
                         (the (Nat, Double) (0, -1.0e10))
                         (zip (the (List Nat) [0,1,2,3]) probs)
        in tokenName (fst best)
  let positions : List Nat
      positions = map finToNat (toList (Data.Vect.Fin.range {len=SeqLen}))
  putStrLn $ concatMap showPos positions

  putStrLn ""
  putStrLn $ formatResult [("epochs", show epochsDone), ("loss", show loss),
                            ("seed", show cfg.seed)]
