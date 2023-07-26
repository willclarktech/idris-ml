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


||| Input/output size = token space + 1 (<BLANK>)
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

decodeYs : Vect E (RecurrentDataPoint W W Variable) -> Vect E (List (Vector W Double))
decodeYs = map (map (map value) . ys)

decodeOutput : Vect E (List (Vector W Variable)) -> Vect E (List (Fin W))
decodeOutput = map (map argmax)


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

  -- let
  --   inp : Vector 5 Nat
  --   inp = VTensor [1, 2, 3, 4, 5]
  -- printLn "ARGMAX:"
  -- printLn $ argmax inp
  -- let (newNtm, out1) = forwardNtm myNtm inp
  -- printLn $ "Output: " ++ show out1
  -- printLn $ "New NTM: " ++ show newNtm
  -- let (newNtm', out2) = forwardNtm newNtm inp
  -- printLn $ "Output: " ++ show out2

  let epochs = 1
  let lr = 0.03
  let lossFn = crossEntropy

  rnn <- nameParams "rnn" <$> rnnLayer
  let model = OutputLayer rnn
  putStr "Model:\t\t"
  printLn model
  let dataPoints = map (map fromDouble) rawData
  putStr "Targets:\t"
  printLn $ decodeOutput $ map ys dataPoints
  let predictions = decodeOutput $ evaluateRecurrent model dataPoints
  let loss = calculateLossRecurrent lossFn model dataPoints

  putStr "Pre loss:\t"
  printLn $ value loss
  putStr "Predictions:\t"
  printLn $ predictions

  let trained = trainRecurrent lr model dataPoints lossFn epochs
  let predictions' = decodeOutput $ evaluateRecurrent trained dataPoints
  let loss' = calculateLossRecurrent lossFn trained dataPoints

  putStr "Post loss:\t"
  printLn $ value loss'
  putStr "Predictions:\t"
  printLn $ predictions'
