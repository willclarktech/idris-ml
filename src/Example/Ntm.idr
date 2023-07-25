module Example.Ntm

import Data.List
import Data.Stream
import Data.Vect
import System.Random

import Backprop
import DataPoint
import Floating
import Layer
import Math
import Network
import Ntm
import Tensor
import Util
import Variable


||| Input/output size
W : Nat
W = 5

||| Memory size (eg max length of input)
N : Nat
N = 6

||| Number of training examples
E : Nat
E = 4

||| 0 is reserved for the <BLANK> token
sequences : Vect E (List (Fin W))
sequences =
  [ [1,2,3,4]
  , [4,3,2,1]
  , [1,2,3,1,2,3]
  , [1]
  ]

-- rawData : Vect n (RecurrentDataPoint N N Double)
-- rawData = map (uncurry MkRecurrentDataPoint . prep) sequences
--   where
--     prep : (sequence : List Double) -> (List (Vector N Double), List (Vector N Double))
--     prep sequence =
--       let
--         pad = Data.List.replicate (length sequence) 0
--         inp = sequence ++ pad
--         outp = pad ++ sequence
--       in (VTensor $ inp, [VTensor $ fromList outp])

-- dataSet : List (Vector N Nat, Vector N Nat)
-- dataSet = map getInputOutputPair sequences
--   where
--     getInputOutputPair : List Nat -> (Vect N Nat, Vect N Nat)
--     getInputOutputPair sequence =
--       let
--         pad = Data.List.replicate (length sequence) 0
--         inp = sequence ++ pad
--         outp = pad ++ sequence
--       in (map oneHotEncode inp, map oneHotEncode outp)

prep : List (Fin W) -> RecurrentDataPoint W W Double
prep sequence =
  let
    pad = Data.List.replicate (length sequence) 0
    inp = sequence ++ pad
    outp = pad ++ sequence
    xs = map (oneHotEncode {n=W}) inp
    ys = map (oneHotEncode {n=W}) outp
    blab = ?blah
  in map (fromInteger . natToInteger) $ MkRecurrentDataPoint xs ys

rawData : Vect E (RecurrentDataPoint W W Double)
rawData = map prep sequences

-- generateData : Nat -> (List Double, List Double)
-- generateData n =
--   let infinitePattern = cycle [0, 1, 0]
--   in (take n infinitePattern, take n (drop 1 infinitePattern))

-- generateDataSet : {n : Nat} -> Vect n (List Double, List Double)
-- generateDataSet = map (generateData. (+3) . finToNat) Data.Vect.Fin.range

-- rawData : (n : Nat) -> Vect n (RecurrentDataPoint 1 1 Double)
-- rawData n = map (\(is, os) => MkRecurrentDataPoint (prep is) (prep os)) $ generateDataSet {n}
--   where
--     prep : (ns : List Double) -> List (Vector 1 Double)
--     prep ns = map (flatten . STensor) ns

-- decodeYs : Vect n (RecurrentDataPoint 1 1 Variable) -> Vect n (List (Vector 1 Double))
-- decodeYs = map (map (map value) . ys)

-- decodeOutput : Vect n (List (Vector o Variable)) -> Vect n (List (Vector o Double))
-- decodeOutput = map (map (map (cast . (0<))))


main : IO ()
main = do
  srand 123456

  -- rnn <- nameParams "rnn" <$> rnnLayer
  -- let myNetwork = OutputLayer rnn
  -- let myMemory = zeros
  -- let myNtm = initNtm {n=3,w=5} myNetwork myMemory
  -- printLn myNtm
  -- printLn myNetwork
  -- printLn "hi"

  -- let inp = VTensor [1, 2, 3, 4, 5]
  -- let (newNtm, out1) = forwardNtm myNtm inp
  -- printLn $ "Output: " ++ show out1
  -- printLn $ "New NTM: " ++ show newNtm
  -- let (newNtm', out2) = forwardNtm newNtm inp
  -- printLn $ "Output: " ++ show out2

  let epochs = 1000
  let lr = 0.03
  let lossFn = binaryCrossEntropyWithLogits

  -- rnn <- nameParams "rnn" <$> rnnLayer
  -- let model = OutputLayer rnn
  -- putStr "Model: "
  -- printLn model
  printLn rawData
  -- let dataPoints = map (map fromDouble) (rawData 8)
  -- putStr "Targets: "
  -- printLn $ decodeYs dataPoints
  -- let predictions = decodeOutput $ evaluateRecurrent model dataPoints
  -- let loss = calculateLossRecurrent lossFn model dataPoints

  -- putStr "Pre loss: "
  -- printLn $ value loss
  -- putStr "Predictions: "
  -- printLn $ predictions

  -- let trained = trainRecurrent lr model dataPoints lossFn epochs
  -- let predictions' = decodeOutput $ evaluateRecurrent trained dataPoints
  -- let loss' = calculateLossRecurrent lossFn trained dataPoints

  -- putStr "Post loss: "
  -- printLn $ value loss'
  -- putStr "Predictions: "
  -- printLn $ predictions'
