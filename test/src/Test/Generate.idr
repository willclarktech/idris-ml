module Test.Generate

import Data.List
import Data.Vect

import DataPoint
import Generate
import Harness
import Tensor


||| Get the value at position j in a vector.
getElem : {n : Nat} -> Nat -> Vector n Double -> Double
getElem j (VTensor elems) = case getAt j (toList elems) of
  Just (STensor v) => v
  _ => 0.0

export
tests : List (IO Bool)
tests =
  [ -- copyTaskBinary: correct encoding/target lengths
    do dp <- copyTaskBinary {w = 8} 5
       let encLen = length (encodingInputs dp)
           tgtLen = length (targets dp)
       -- encoding: 5 data rows + 1 delimiter = 6, targets: 5 rows
       check "copyTaskBinary lengths" (encLen == 6 && tgtLen == 5)

  -- copyTaskBinary: delimiter is last encoding row, has 1 in last channel
  , do dp <- copyTaskBinary {w = 3} 2
       let enc = encodingInputs dp
           -- Last row (index 2) is the delimiter: [0, 0, 0, 1]
           delimRow = fromMaybe zeros (last' enc)
       check "copyTaskBinary delimiter" (getElem 3 delimRow == 1.0)

  -- copyTaskBinary: first data row has 0 in delimiter channel
  , do dp <- copyTaskBinary {w = 4} 3
       let firstRow = fromMaybe zeros (head' (encodingInputs dp))
       check "copyTaskBinary data delim=0" (getElem 4 firstRow == 0.0)

  -- copyTaskBinary: target values are binary (0 or 1)
  , do dp <- copyTaskBinary {w = 8} 4
       let allBinary = all (\row =>
             all (\v => v == 0.0 || v == 1.0) (toList row)) (targets dp)
       check "copyTaskBinary targets binary" allBinary

  -- copyTaskBinaryBatchVect: correct batch size
  , do batch <- copyTaskBinaryBatchVect {w = 8} 5 1 3
       check "copyTaskBinaryBatchVect size" (length batch == 5)

  -- recallTaskBinary: correct target length = seqLen
  , do dp <- recallTaskBinary {w = 6} 3 4
       check "recallTaskBinary target len" (length (targets dp) == 4)

  -- recallTaskBinary: encoding length
  , do dp <- recallTaskBinary {w = 6} 3 4
       let encLen = length (encodingInputs dp)
       -- 3*(1+4) + 1 + 4 + 1 = 21
       check "recallTaskBinary encoding len" (encLen == 21)

  -- recallTaskBinaryBatchVect: correct batch size
  , do batch <- recallTaskBinaryBatchVect {w = 6} 4 2 3 3
       check "recallTaskBinaryBatchVect size" (length batch == 4)
  ]
