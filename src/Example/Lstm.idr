module Example.Lstm

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
  patience : Nat
  seed : Bits64

defaultConfig : Config
defaultConfig = MkConfig 0.1 2000 500 123456

specs : List (ArgSpec Config)
specs = [ Arg "--lr" (\v, c => { lr := cast v } c)
        , Arg "--epochs" (\v, c => { epochs := castNat v } c)
        , Arg "--patience" (\v, c => { patience := castNat v } c)
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

  putStrLn "=== LSTM Pattern Prediction ==="
  putStrLn $ "Config: lr=" ++ show cfg.lr ++ " epochs=" ++ show cfg.epochs
           ++ " patience=" ++ show cfg.patience ++ " seed=" ++ show cfg.seed

  lstm <- lstmLayer
  ll <- linearLayer {i=1, o=1}
  let model = autoName $ lstm ~> OutputLayer ll
  putStrLn $ "Architecture: " ++ show model
  putStrLn ""

  (trained, epochsDone, _) <- runTraining
    (\m, d => epochRecurrentNative opt d lossFn m) (pure dataPoints)
    (patienceConfig cfg.epochs cfg.patience) model

  let dblModel = toDoubleNetwork (emap refreshValue trained)
  let predictions : Vect 8 (List (Vector 1 Double))
      predictions = map (map (map (\x => cast (0 < x)))) (evaluateRecurrent dblModel (patternData 8))
  let loss = calculateLossRecurrent lossFn dblModel (patternData 8)

  putStrLn ""
  putStrLn "Eval:"
  putStrLn $ "  Loss: " ++ show loss
  let tgts = map ys (patternData 8)
  putStrLn "  Seq  Target     Predicted"
  traverse_ (\(i, (t, p)) =>
    let ts = showSeq (toList t)
        ps = showSeq (toList p)
    in putStrLn $ "  " ++ show (finToNat i + 1) ++ ".   " ++ ts ++ "  ->  " ++ ps
                ++ (if ts == ps then " ok" else ""))
    (zip Fin.range (zip tgts predictions))
  putStrLn ""
  putStrLn $ formatResult [("epochs", show epochsDone), ("loss", show loss),
                            ("seed", show cfg.seed)]
