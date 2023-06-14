module Backprop

import Data.Vect

import DataPoint
import Endofunctor
import Math
import Network
import Tensor
import Variable


export
zeroGrad : Network i hs o Variable -> Network i hs o Variable
zeroGrad = emap {grad := 0}

export
transferGrads : Network i hs o Variable -> List (String, Double) -> Network i hs o Variable
transferGrads = foldr addGrad
  where
    addGrad : (String, Double) -> (Network i hs o Variable) -> (Network i hs o Variable)
    addGrad (pid, g) =
      emap
        ( \v =>
            case paramId v of
              Nothing => v
              Just p => if p == pid then {grad $= (+ g)} v else v
        )

export
backward : Variable -> Network i hs o Variable -> Network i hs o Variable
backward loss m =
  let propagated = backwardVariable 1 loss
      grads = gradMap propagated
   in transferGrads m grads

export
updateParam : Double -> Variable -> Variable
updateParam lr p = {value $= \v => v - lr * p.grad} p

export
step : Double -> Network i hs o Variable -> Network i hs o Variable
step lr = emap (updateParam lr)

export
epoch :
  {i, o, n : Nat} ->
  {hs : List Nat} ->
  Double ->
  Vect n (DataPoint i o Variable) ->
  LossFunction Variable ->
  Network i hs o Variable ->
  Network i hs o Variable
epoch lr dataPoints lossFn model =
  let zeroed = zeroGrad model
      loss = calculateLoss lossFn zeroed dataPoints
      propagated = backward loss zeroed
   in step lr propagated

export
train :
  {i, o, n : Nat} ->
  {hs : List Nat} ->
  Double ->
  Network i hs o Variable ->
  Vect n (DataPoint i o Variable) ->
  LossFunction Variable ->
  Int ->
  Network i hs o Variable
train lr model dataPoints lossFn epochs = foldr (const $ epoch lr dataPoints lossFn) model [1 .. epochs]

export
epochRecurrent :
  {i, o, n : Nat} ->
  {hs : List Nat} ->
  Double ->
  Vect n (RecurrentDataPoint i o Variable) ->
  LossFunction Variable ->
  Network i hs o Variable ->
  Network i hs o Variable
epochRecurrent lr dataPoints lossFn model =
  let zeroed = zeroGrad model
      loss = calculateLossRecurrent lossFn zeroed dataPoints
      propagated = backward loss zeroed
   in step lr propagated

export
trainRecurrent :
  {i, o, n : Nat} ->
  {hs : List Nat} ->
  Double ->
  Network i hs o Variable ->
  Vect n (RecurrentDataPoint i o Variable) ->
  LossFunction Variable ->
  Int ->
  Network i hs o Variable
trainRecurrent lr model dataPoints lossFn epochs = foldr (const $ epochRecurrent lr dataPoints lossFn) model [1 .. epochs]
