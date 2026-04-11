module Example.Rnn

import Data.List
import Data.Vect
import System
import System.Clock
import System.Random

import Backprop
import DataPoint
import Endofunctor
import Floating
import Generate
import Layer
import Math
import Optimizer
import Tensor
import Train
import Util
import Variable


record Config where
  constructor MkConfig
  lr : Double
  epochs : Nat
  seed : Bits64

defaultConfig : Config
defaultConfig = MkConfig 0.03 1000 123456

specs : List (ArgSpec Config)
specs = [ Arg "--lr" (\v, c => { lr := cast v } c)
        , Arg "--epochs" (\v, c => { epochs := castNat v } c)
        , Arg "--seed" (\v, c => { seed := castBits64 v } c) ]

showSeq : List (Vector 1 Double) -> String
showSeq xs = concatMap (\(VTensor [STensor v]) => if v >= 0.5 then "1" else "0") xs

main : IO ()
main = do
  args <- getArgs
  let cfg = parseArgs defaultConfig specs (drop 1 args)

  srand cfg.seed

  let lossFn = binaryCrossEntropyWithLogits
  let opt = nativeSgd cfg.lr
  let dataPoints = map (map fromDouble) (patternData 8)

  putStrLn "=== RNN Pattern Prediction ==="
  putStrLn $ "Config: lr=" ++ show cfg.lr ++ " epochs=" ++ show cfg.epochs
           ++ " seed=" ++ show cfg.seed

  rnn <- rnnLayer
  let model = autoName $ OutputLayer rnn
  putStrLn $ "Architecture: " ++ show model
  putStrLn ""

  (trained, epochsDone, _) <- runTraining
    (\m, d => epochRecurrentNative opt d lossFn m) (pure dataPoints) (simpleConfig cfg.epochs) model

  let dblModel = toDoubleNetwork (emap refreshValue trained)
  let predictions : Vect 8 (List (Vector 1 Double))
      predictions = map (map (map (\x => cast (0 < x)))) (evaluateRecurrent dblModel (patternData 8))
  let finalLoss = calculateLossRecurrent lossFn dblModel (patternData 8)

  putStrLn ""
  putStrLn "Eval:"
  putStrLn $ "  Loss: " ++ show finalLoss
  let tgts = map ys (patternData 8)
  putStrLn "  Seq  Target     Predicted"
  traverse_ (\(i, (t, p)) =>
    let ts = showSeq (toList t)
        ps = showSeq (toList p)
    in putStrLn $ "  " ++ show (finToNat i + 1) ++ ".   " ++ ts ++ "  ->  " ++ ps
                ++ (if ts == ps then " ok" else ""))
    (zip Fin.range (zip tgts predictions))
  putStrLn ""
  putStrLn $ formatResult [("epochs", show epochsDone), ("loss", show finalLoss),
                            ("seed", show cfg.seed)]
