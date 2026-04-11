module Example.Lstm

import Data.List
import Data.Stream
import Data.String
import Data.Vect
import System
import System.Clock
import System.Random

import Backprop
import DataPoint
import Endofunctor
import Floating
import Layer
import Math
import Optimizer
import Tensor
import Train
import Util
import Variable


generateData : Nat -> (List Double, List Double)
generateData n =
  let infinitePattern = cycle [0, 1, 0]
  in (take n infinitePattern, take n (drop 1 infinitePattern))

generateDataSet : {n : Nat} -> Vect n (List Double, List Double)
generateDataSet = map (generateData . (+3) . finToNat) Data.Vect.Fin.range

rawData : (n : Nat) -> Vect n (RecurrentDataPoint 1 1 Double)
rawData n = map (\(is, os) => MkRecurrentDataPoint (prep is) (prep os)) $ generateDataSet {n}
  where
    prep : (ns : List Double) -> List (Vector 1 Double)
    prep ns = map (flatten . STensor) ns


----------------------------------------------------------------------
-- CLI Argument Parsing
----------------------------------------------------------------------

record Config where
  constructor MkConfig
  lr : Double
  epochs : Nat
  patience : Nat
  seed : Bits64

defaultConfig : Config
defaultConfig = MkConfig 0.1 2000 500 123456

parseConfig : List String -> Config
parseConfig args = go args defaultConfig
  where
    go : List String -> Config -> Config
    go [] c = c
    go ("--lr" :: v :: rest) c = go rest ({ lr := cast v } c)
    go ("--epochs" :: v :: rest) c = go rest ({ epochs := cast (cast {to=Integer} v) } c)
    go ("--patience" :: v :: rest) c = go rest ({ patience := cast (cast {to=Integer} v) } c)
    go ("--seed" :: v :: rest) c = go rest ({ seed := cast (cast {to=Integer} v) } c)
    go (_ :: rest) c = go rest c


----------------------------------------------------------------------
-- Main
----------------------------------------------------------------------

main : IO ()
main = do
  tStart <- clockTime Monotonic
  args <- getArgs
  let cfg = parseConfig (drop 1 args)

  srand cfg.seed

  let lossFn = binaryCrossEntropyWithLogits
  let opt = nativeSgd cfg.lr
  let dataPoints = map (map fromDouble) (rawData 8)

  putStrLn "=== LSTM Pattern Prediction ==="
  putStrLn $ "Config: lr=" ++ show cfg.lr
           ++ " epochs=" ++ show cfg.epochs
           ++ " patience=" ++ show cfg.patience
           ++ " seed=" ++ show cfg.seed

  lstm <- lstmLayer
  ll <- linearLayer {i=1, o=1}
  let model = autoName $ lstm ~> OutputLayer ll
  putStrLn $ "Architecture: " ++ show model
  putStrLn ""

  (trained, epochsDone, _) <- runTraining
    (\m, d => epochRecurrentNative opt d lossFn m)
    (pure dataPoints)
    (patienceConfig cfg.epochs cfg.patience)
    model
  t1 <- clockTime Monotonic

  let dblModel = toDoubleNetwork (emap refreshValue trained)
  let predictions : Vect 8 (List (Vector 1 Double))
      predictions = map (map (map (\x => cast (0 < x)))) (evaluateRecurrent dblModel (rawData 8))
  let loss = calculateLossRecurrent lossFn dblModel (rawData 8)

  putStrLn ""
  putStrLn "Eval:"
  putStrLn $ "  Loss: " ++ show loss
  let showSeq : List (Vector 1 Double) -> String
      showSeq xs = concatMap (\(VTensor [STensor v]) => if v >= 0.5 then "1" else "0") xs
  let tgts = map ys (rawData 8)
  putStrLn "  Seq  Target     Predicted"
  traverse_ (\(i, (t, p)) =>
    let ts = showSeq (toList t)
        ps = showSeq (toList p)
        mark = if ts == ps then " ok" else ""
    in putStrLn $ "  " ++ show (finToNat i + 1) ++ ".   " ++ ts ++ "  ->  " ++ ps ++ mark)
    (zip Fin.range (zip tgts predictions))
  putStrLn ""
  putStrLn $ formatTimingSummary tStart t1 epochsDone
  putStrLn $ "RESULT\tepochs=" ++ show epochsDone ++ "\tloss=" ++ show loss
           ++ "\ttime=" ++ show (seconds t1 - seconds tStart) ++ "s"
           ++ "\tseed=" ++ show cfg.seed
