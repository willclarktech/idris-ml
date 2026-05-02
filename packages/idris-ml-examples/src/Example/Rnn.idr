module Example.Rnn

import Data.List
import Data.Vect
import System
import Compat.Random

import Backprop
import DataPoint
import Device
import Endofunctor
import Floating
import Generate
import Hpo.LrFinder
import Layer
import Math
import Optimizer
import Tensor
import Train
import Util
import Variable


showSeq : List (Vector 1 Double) -> String
showSeq xs = "[" ++ go xs ++ "]"
  where
    go : List (Vector 1 Double) -> String
    go [] = ""
    go [VTensor [STensor v]] = show (cast {to=Int} v)
    go (VTensor [STensor v] :: rest) = show (cast {to=Int} v) ++ "," ++ go rest

-- Tensor-level BCE with logits: max(x,0) - x*y + log(1+exp(-|x|))
bceLossTensor : LossFnTensor CPU
bceLossTensor predT targetT =
  let relu_x = prim__clampMin predT 0.0
      xy = prim__mul predT targetT
      abs_x = prim__abs predT
      neg_abs_x = prim__neg abs_x
      exp_neg = prim__exp neg_abs_x
      one_plus_exp = tensorAdd exp_neg (prim__createScalar 1.0 0)
      log_term = prim__log one_plus_exp
      -- max(x,0) - x*y + log(1+exp(-|x|))
      loss = tensorAdd (prim__sub relu_x xy) log_term
      result = prim__mean loss
      val = prim__item result
  in Var result Nothing val

record Config where
  constructor MkConfig
  lr : Double
  epochs : Nat
  seed : Bits64
  lrFind : Bool

defaultConfig : Config
defaultConfig = MkConfig 0.03 2000 42 False

specs : List (ArgSpec Config)
specs = [ Arg "--lr" (\v, c => { lr := cast v } c)
        , Arg "--epochs" (\v, c => { epochs := castNat v } c)
        , Arg "--seed" (\v, c => { seed := castBits64 v } c)
        -- See Supervised.idr for the boolean-flag rationale.
        , Arg "--lr-find" (\v, c => { lrFind := (v == "1" || v == "true") } c) ]

main : IO ()
main = do
  args <- getArgs
  let cfg = parseArgs defaultConfig specs (drop 1 args)

  srand cfg.seed

  let opt = nativeSgd cfg.lr

  putStrLn "=== RNN Pattern Prediction ==="
  putStrLn $ "Config: lr=" ++ show cfg.lr ++ " epochs=" ++ show cfg.epochs
           ++ " seed=" ++ show cfg.seed

  rnn <- rnnLayer
  let model = autoName $ OutputLayer rnn
  putStrLn $ "Architecture: " ++ show model
  putStrLn ""

  when cfg.lrFind $ do
    let lrCfg : LrFindConfig
        lrCfg = { numIters := 100 } defaultLrFindConfig
    _ <- lrFind lrCfg
      (\m, d => let (m', loss) = epochRecurrentNativeTensor opt d bceLossTensor m
                in pure (m', loss))
      (pure (patternData 8)) opt model
    putStrLn ""
    putStrLn "Done — re-run without --lr-find at the recommended LR."
    exitSuccess

  (trained, epochsDone, _) <- runTraining
    (\m, d => epochRecurrentNativeTensor opt d bceLossTensor m) (pure (patternData 8)) (simpleConfig cfg.epochs) model

  let dblModel = toDoubleNetwork (emap refreshValue trained)
  let predictions : Vect 8 (List (Vector 1 Double))
      predictions = map (map (map (\x => cast (0 < x)))) (evaluateRecurrent dblModel (patternData 8))
  let finalLoss = calculateLossRecurrent binaryCrossEntropyWithLogits dblModel (patternData 8)

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
