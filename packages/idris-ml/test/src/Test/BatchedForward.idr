module Test.BatchedForward

import Data.Vect
import System.Random

import Harness
import Floating
import Layer
import Tensor
import Device
import Variable


----------------------------------------------------------------------
-- Helpers
----------------------------------------------------------------------

-- Read element [b, j] of a [B, o] tensor.
readBO : AnyPtr -> Int -> Int -> Double
readBO t b j =
  let row = prim__select t 0 b
      elem = prim__select row 0 j
  in prim__item elem

-- Read element [j] of a [o] tensor.
readO : AnyPtr -> Int -> Double
readO t j = prim__item (prim__select t 0 j)


----------------------------------------------------------------------
-- Tests
----------------------------------------------------------------------

export
tests : List (IO Bool)
tests =
  [ -- Batched forward equals per-sample forward for an MLP (Linear -> Tanh -> Linear).
    do srand 42
       l1 <- linearLayer {ty = Variable CPU} {i=4, o=8}
       l2 <- linearLayer {ty = Variable CPU} {i=8, o=2}
       let net = autoName $ l1 ~> tanhLayer ~> OutputLayer l2

       -- Three samples
       let s0 : Vect 4 Double = [1.0, -0.5, 0.3, 0.8]
       let s1 : Vect 4 Double = [0.2, 0.4, -0.6, 1.2]
       let s2 : Vect 4 Double = [-0.7, 0.1, 0.9, -0.4]

       let t0 = bulkToTensor (the (Vector 4 Double) (VTensor (map STensor s0)))
       let t1 = bulkToTensor (the (Vector 4 Double) (VTensor (map STensor s1)))
       let t2 = bulkToTensor (the (Vector 4 Double) (VTensor (map STensor s2)))

       -- Per-sample forward
       let (_, y0) = forwardVarTensor net t0
       let (_, y1) = forwardVarTensor net t1
       let (_, y2) = forwardVarTensor net t2

       -- Batched forward on stacked [3, 4] input
       let stacked : Vect 3 (Vector 4 Double) =
             [VTensor (map STensor s0), VTensor (map STensor s1), VTensor (map STensor s2)]
       let bT = bulkToTensor2d stacked
       let (_, yB) = forwardVarTensorBatch net 3 bT

       -- Compare each [B, o] vs each per-sample [o]
       let tol = 1.0e-6
       let ok00 = abs (readBO yB 0 0 - readO y0 0) < tol
       let ok01 = abs (readBO yB 0 1 - readO y0 1) < tol
       let ok10 = abs (readBO yB 1 0 - readO y1 0) < tol
       let ok11 = abs (readBO yB 1 1 - readO y1 1) < tol
       let ok20 = abs (readBO yB 2 0 - readO y2 0) < tol
       let ok21 = abs (readBO yB 2 1 - readO y2 1) < tol
       check "batched forward matches per-sample (Linear-Tanh-Linear)"
             (ok00 && ok01 && ok10 && ok11 && ok20 && ok21)

  -- Same check with ReLU activation.
  , do srand 7
       l1 <- linearLayer {ty = Variable CPU} {i=3, o=5}
       l2 <- linearLayer {ty = Variable CPU} {i=5, o=2}
       let net = autoName $ l1 ~> reluLayer ~> OutputLayer l2

       let s0 : Vect 3 Double = [0.6, -1.1, 0.4]
       let s1 : Vect 3 Double = [-0.2, 0.8, -0.5]

       let t0 = bulkToTensor (the (Vector 3 Double) (VTensor (map STensor s0)))
       let t1 = bulkToTensor (the (Vector 3 Double) (VTensor (map STensor s1)))

       let (_, y0) = forwardVarTensor net t0
       let (_, y1) = forwardVarTensor net t1

       let stacked : Vect 2 (Vector 3 Double) =
             [VTensor (map STensor s0), VTensor (map STensor s1)]
       let bT = bulkToTensor2d stacked
       let (_, yB) = forwardVarTensorBatch net 2 bT

       let tol = 1.0e-6
       let ok = abs (readBO yB 0 0 - readO y0 0) < tol
             && abs (readBO yB 0 1 - readO y0 1) < tol
             && abs (readBO yB 1 0 - readO y1 0) < tol
             && abs (readBO yB 1 1 - readO y1 1) < tol
       check "batched forward matches per-sample (Linear-ReLU-Linear)" ok
  ]
