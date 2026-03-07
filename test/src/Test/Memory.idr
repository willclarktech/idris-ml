module Test.Memory

import Data.Fin
import Data.Vect

import Floating
import Harness
import Math
import Memory
import Tensor


tol : Double
tol = 1.0e-6

export
tests : List (IO Bool)
tests =
  [ -- cycleForward: shift by 1 rotates left (output[j] = xs[(j+1) mod n])
    let v = the (Vect 4 Double) [1.0, 2.0, 3.0, 4.0]
        result = cycleForward {n=4} 1 v
    in check "cycleForward by 1" (result == [2.0, 3.0, 4.0, 1.0])

  -- cycleForward by 0
  , let v = the (Vect 3 Double) [1.0, 2.0, 3.0]
        result = cycleForward {n=3} 0 v
    in check "cycleForward by 0" (result == [1.0, 2.0, 3.0])

  -- readOp weighted sum
  , let mem = the (Matrix 3 2 Double) (VTensor
          [ VTensor [STensor 1.0, STensor 0.0]
          , VTensor [STensor 0.0, STensor 1.0]
          , VTensor [STensor 1.0, STensor 1.0]
          ])
        rh = MkReadHead (the (Vector 3 Double) (VTensor [STensor 0.5, STensor 0.3, STensor 0.2]))
        result = readOp rh mem
        (VTensor [STensor r0, STensor r1]) = result
    in check "readOp weighted sum"
       (abs (r0 - (0.5 * 1.0 + 0.3 * 0.0 + 0.2 * 1.0)) < tol
        && abs (r1 - (0.5 * 0.0 + 0.3 * 1.0 + 0.2 * 1.0)) < tol)

  -- getContentAddress finds most similar row
  , let mem = the (Matrix 3 2 Double) (VTensor
          [ VTensor [STensor 1.0, STensor 0.0]
          , VTensor [STensor 0.0, STensor 1.0]
          , VTensor [STensor 0.7, STensor 0.1]
          ])
        key = the (Vector 2 Double) (VTensor [STensor 1.0, STensor 0.0])
        addr = getContentAddress softmax 10.0 mem key
        (VTensor [STensor a0, STensor a1, STensor a2]) = addr
    -- With high beta, weight should concentrate on row 0 (most similar)
    in check "getContentAddress" (a0 > a1 && a0 > a2)

  -- eraseMemory
  , let mem = the (Matrix 2 2 Double) (VTensor
          [ VTensor [STensor 1.0, STensor 1.0]
          , VTensor [STensor 1.0, STensor 1.0]
          ])
        addr = the (Vector 2 Double) (VTensor [STensor 1.0, STensor 0.0])
        erase = the (Vector 2 Double) (VTensor [STensor 1.0, STensor 1.0])
        result = eraseMemory mem addr erase
        -- Row 0 should be erased (1*(1-1*1)=0), row 1 unchanged (1*(1-0*1)=1)
        (VTensor [VTensor [STensor r00, STensor r01], VTensor [STensor r10, STensor r11]]) = result
    in check "eraseMemory" (abs r00 < tol && abs r01 < tol && abs (r10 - 1.0) < tol && abs (r11 - 1.0) < tol)

  -- addMemory
  , let mem = the (Matrix 2 2 Double) (VTensor
          [ VTensor [STensor 0.0, STensor 0.0]
          , VTensor [STensor 0.0, STensor 0.0]
          ])
        addr = the (Vector 2 Double) (VTensor [STensor 0.8, STensor 0.2])
        addVec = the (Vector 2 Double) (VTensor [STensor 1.0, STensor 0.5])
        result = addMemory mem addr addVec
        (VTensor [VTensor [STensor r00, STensor r01], VTensor [STensor r10, STensor r11]]) = result
    in check "addMemory"
       (abs (r00 - 0.8) < tol && abs (r01 - 0.4) < tol
        && abs (r10 - 0.2) < tol && abs (r11 - 0.1) < tol)

  -- focus with gamma=1 is normalize
  , let w = the (Vector 3 Double) (VTensor [STensor 2.0, STensor 3.0, STensor 5.0])
        f = focus 1.0 w
        (VTensor [STensor f0, STensor f1, STensor f2]) = f
    in check "focus gamma=1 = normalize"
       (abs (f0 - 0.2) < tol && abs (f1 - 0.3) < tol && abs (f2 - 0.5) < tol)

  -- focus with gamma > 1 sharpens
  , let w = the (Vector 3 Double) (VTensor [STensor 0.2, STensor 0.3, STensor 0.5])
        f1 = focus 1.0 w
        f2 = focus 5.0 w
        (VTensor [_, _, STensor peak1]) = f1
        (VTensor [_, _, STensor peak2]) = f2
    in check "focus gamma>1 sharpens" (peak2 > peak1)

  -- interpolationWrite: w=1 fully replaces, w=0 keeps original (+ fused tanh)
  , let mem = the (Matrix 2 2 Double) (VTensor
          [ VTensor [STensor 1.0, STensor 2.0]
          , VTensor [STensor 3.0, STensor 4.0]
          ])
        weights = the (Vector 2 Double) (VTensor [STensor 1.0, STensor 0.0])
        addVec = the (Vector 2 Double) (VTensor [STensor 5.0, STensor 6.0])
        result = interpolationWrite mem weights addVec
        -- Row 0: w=1 => tanh(add), row 1: w=0 => tanh(original)
        (VTensor [VTensor [STensor r00, STensor r01], VTensor [STensor r10, STensor r11]]) = result
    in check "interpolationWrite"
       (abs (r00 - 0.999909204263) < tol && abs (r01 - 0.999987711651) < tol
        && abs (r10 - 0.995054753687) < tol && abs (r11 - 0.999329299739) < tol)

  -- interpolationWrite with partial weight (+ fused tanh)
  , let mem = the (Matrix 1 2 Double) (VTensor
          [ VTensor [STensor 10.0, STensor 0.0] ])
        weights = the (Vector 1 Double) (VTensor [STensor 0.5])
        addVec = the (Vector 2 Double) (VTensor [STensor 0.0, STensor 20.0])
        result = interpolationWrite mem weights addVec
        -- tanh(0.5*10 + 0.5*0) = tanh(5), tanh(0.5*0 + 0.5*20) = tanh(10)
        (VTensor [VTensor [STensor r0, STensor r1]]) = result
    in check "interpolationWrite partial"
       (abs (r0 - 0.999909204263) < tol && abs (r1 - 0.999999995878) < tol)

  -- softplus gamma: gamma = 1 + softplus(0) = 1 + ln(2) ≈ 1.693
  , let gammaVal = 1.0 + softplus 0.0
    in checkClose "softplus gamma at 0" 1.6931471805599454 gammaVal tol
  ]
