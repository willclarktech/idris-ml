module Test.Variable

import Data.SortedMap
import Data.Vect

import Harness
import Floating
import Math
import Tensor
import Device
import Variable


tol : Double
tol = 1.0e-4

lookupGrad : String -> SortedMap String Double -> Double
lookupGrad key m = case lookup key m of
  Just v => v
  Nothing => 0.0 / 0.0  -- NaN to signal missing

-- CPU-specialized param for test convenience
cpuParam : String -> Double -> Variable CPU
cpuParam = param

-- Inline FFI + wrapper for the new group-scoped optimizer (see the A2C /
-- PPO / SAC workaround: single-file `idris2 -o file.idr` invocation
-- doesn't always pick up newly-exported idris-ml symbols even after
-- install; inline bindings sidestep this).
%foreign "C:optimizer_create_adam_group,libidrisml"
prim__mkAdamGroupLocal : Double -> Double -> Double -> Double -> String -> AnyPtr

mkAdamGroup : String -> NativeOptimizer
mkAdamGroup scope =
  MkNativeOptimizer (prim__mkAdamGroupLocal 0.1 0.9 0.999 1.0e-8 scope) (NormClip 1.0)

export
tests : List (IO Bool)
tests =
  [ -- Test 1 (from TapeTest): y = w*x + b => dy/dw = x, dy/db = 1
    let w = cpuParam "t1w" 3.0
        b = cpuParam "t1b" 1.0
        x = fromDouble 2.0
        y = w * x + b
        g = collectGrads 1.0 y
    in check "w*x+b: dw=x, db=1"
       (abs (lookupGrad "t1w" g - 2.0) < tol && abs (lookupGrad "t1b" g - 1.0) < tol)

  -- Test 2 (from TapeTest): y = exp(w) => dy/dw = exp(w)
  , let w = cpuParam "t2w" 1.0
        y = exp w
        g = collectGrads 1.0 y
    in checkClose "exp(w): dw=exp(w)" (exp 1.0) (lookupGrad "t2w" g) tol

  -- Test 3 (from TapeTest): y = a/b => dy/da = 1/b, dy/db = -a/b^2
  , let a = cpuParam "t3a" 6.0
        b = cpuParam "t3b" 3.0
        y = a / b
        g = collectGrads 1.0 y
    in check "a/b: da=1/b, db=-a/b^2"
       (abs (lookupGrad "t3a" g - 1.0/3.0) < tol && abs (lookupGrad "t3b" g - (-6.0/9.0)) < tol)

  -- Test 4 (from TapeTest): dot product
  , let w0 = cpuParam "t4w0" 0.5
        w1 = cpuParam "t4w1" 0.5
        x0 = fromDouble 1.0
        x1 = fromDouble 2.0
        y = w0 * x0 + w1 * x1
        g = collectGrads 1.0 y
    in check "dot product grads = x"
       (abs (lookupGrad "t4w0" g - 1.0) < tol && abs (lookupGrad "t4w1" g - 2.0) < tol)

  -- Subtraction
  , let a = cpuParam "t5a" 5.0
        b = cpuParam "t5b" 3.0
        y = a - b
        g = collectGrads 1.0 y
    in check "a-b: da=1, db=-1"
       (abs (lookupGrad "t5a" g - 1.0) < tol && abs (lookupGrad "t5b" g - (-1.0)) < tol)

  -- Negate
  , let w = cpuParam "t6w" 4.0
        y = negate w
        g = collectGrads 1.0 y
    in checkClose "negate: dw=-1" (-1.0) (lookupGrad "t6w" g) tol

  -- Sqrt
  , let w = cpuParam "t7w" 4.0
        y = sqrt w
        g = collectGrads 1.0 y
    in checkClose "sqrt(w): dw=1/(2*sqrt(w))" (1.0 / (2.0 * Prelude.sqrt 4.0)) (lookupGrad "t7w" g) tol

  -- Power
  , let w = cpuParam "t8w" 3.0
        y = pow w (fromDouble 2.0)
        g = collectGrads 1.0 y
    in checkClose "w^2: dw=2w" 6.0 (lookupGrad "t8w" g) tol

  -- Abs (positive)
  , let w = cpuParam "t9w" 3.0
        y = abs w
        g = collectGrads 1.0 y
    in checkClose "abs(3): dw=1" 1.0 (lookupGrad "t9w" g) tol

  -- Abs (negative)
  , let w = cpuParam "t10w" (-3.0)
        y = abs w
        g = collectGrads 1.0 y
    in checkClose "abs(-3): dw=-1" (-1.0) (lookupGrad "t10w" g) tol

  -- Chain rule: exp(w*x)
  , let w = cpuParam "t11w" 2.0
        x = fromDouble 3.0
        y = exp (w * x)
        g = collectGrads 1.0 y
    in checkClose "exp(w*x): dw = x*exp(w*x)" (3.0 * exp 6.0) (lookupGrad "t11w" g) (exp 6.0 * 1.0e-6)

  -- Shared subexpression: y = w*w, dy/dw = 2w
  , let w = cpuParam "t12w" 5.0
        y = w * w
        g = collectGrads 1.0 y
    in checkClose "w*w: dw=2w" 10.0 (lookupGrad "t12w" g) tol

  -- C-backed dotProductVar
  , let a0 = cpuParam "t13a0" 1.0
        a1 = cpuParam "t13a1" 2.0
        b0 = cpuParam "t13b0" 3.0
        b1 = cpuParam "t13b1" 4.0
        va = the (Vector 2 (Variable CPU)) (VTensor [STensor a0, STensor a1])
        vb = the (Vector 2 (Variable CPU)) (VTensor [STensor b0, STensor b1])
        y = dotProductVar va vb
        g = collectGrads 1.0 y
    in check "dotProductVar: grads match"
       (abs (lookupGrad "t13a0" g - 3.0) < tol && abs (lookupGrad "t13a1" g - 4.0) < tol
        && abs (lookupGrad "t13b0" g - 1.0) < tol && abs (lookupGrad "t13b1" g - 2.0) < tol)

  -- C-backed softmaxVar sums to 1
  , let x0 = cpuParam "t14x0" 1.0
        x1 = cpuParam "t14x1" 2.0
        x2 = cpuParam "t14x2" 3.0
        v = the (Vector 3 (Variable CPU)) (VTensor [STensor x0, STensor x1, STensor x2])
        sm = softmaxVar v
        s = sum sm
    in checkClose "softmaxVar sums to 1" 1.0 s.value tol

  -- Stale variable re-registration
  , let w = cpuParam "t15w" 7.0
        y1 = w * (fromDouble 2.0)
        _ = collectGrads 1.0 y1  -- resets tape (gen++)
        -- w is now stale; using it again should re-register
        y2 = w * (fromDouble 3.0)
        g2 = collectGrads 1.0 y2
    in checkClose "stale re-registration" 3.0 (lookupGrad "t15w" g2) tol

  -- C-backed batchCosineSimilarityVar: forward check
  , let k0 = cpuParam "t16k0" 1.0
        k1 = cpuParam "t16k1" 0.0
        m00 = cpuParam "t16m00" 1.0
        m01 = cpuParam "t16m01" 0.0
        m10 = cpuParam "t16m10" 0.0
        m11 = cpuParam "t16m11" 1.0
        beta = cpuParam "t16beta" 10.0
        key = the (Vector 2 (Variable CPU)) (VTensor [STensor k0, STensor k1])
        mem = the (Matrix 2 2 (Variable CPU)) (VTensor [VTensor [STensor m00, STensor m01], VTensor [STensor m10, STensor m11]])
        scores = batchCosineSimilarityVar beta mem key
        (VTensor [STensor s0, STensor s1]) = scores
    in check "batchCosineSimilarityVar forward"
       (abs (s0.value - 10.0) < tol && abs (s1.value - 0.0) < tol)

  -- C-backed batchCosineSimilarityVar: gradient to beta
  , let k0 = cpuParam "t17k0" 2.0
        k1 = cpuParam "t17k1" 1.0
        m00 = cpuParam "t17m00" 3.0
        m01 = cpuParam "t17m01" 1.0
        beta = cpuParam "t17beta" 5.0
        key = the (Vector 2 (Variable CPU)) (VTensor [STensor k0, STensor k1])
        mem = the (Matrix 1 2 (Variable CPU)) (VTensor [VTensor [STensor m00, STensor m01]])
        scores = batchCosineSimilarityVar beta mem key
        y = sum scores
        g = collectGrads 1.0 y
        -- d_beta = cos_sim(key, row0) = (6+1)/(sqrt(5)*sqrt(10))
        expected = 7.0 / (Prelude.sqrt 5.0 * Prelude.sqrt 10.0)
    in checkClose "batchCosineSimilarityVar d_beta" expected (lookupGrad "t17beta" g) 1.0e-4

  -- C-backed readOpVar: forward check
  , let w0 = cpuParam "t18w0" 0.6
        w1 = cpuParam "t18w1" 0.4
        m00 = cpuParam "t18m00" 1.0
        m01 = cpuParam "t18m01" 2.0
        m10 = cpuParam "t18m10" 3.0
        m11 = cpuParam "t18m11" 4.0
        weights = the (Vector 2 (Variable CPU)) (VTensor [STensor w0, STensor w1])
        mem = the (Matrix 2 2 (Variable CPU)) (VTensor [VTensor [STensor m00, STensor m01], VTensor [STensor m10, STensor m11]])
        result = readOpVar weights mem
        (VTensor [STensor r0, STensor r1]) = result
    in check "readOpVar forward"
       (abs (r0.value - 1.8) < tol && abs (r1.value - 2.8) < tol)

  -- C-backed readOpVar: gradient check
  , let w0 = cpuParam "t19w0" 0.6
        w1 = cpuParam "t19w1" 0.4
        m00 = cpuParam "t19m00" 1.0
        m01 = cpuParam "t19m01" 2.0
        m10 = cpuParam "t19m10" 3.0
        m11 = cpuParam "t19m11" 4.0
        weights = the (Vector 2 (Variable CPU)) (VTensor [STensor w0, STensor w1])
        mem = the (Matrix 2 2 (Variable CPU)) (VTensor [VTensor [STensor m00, STensor m01], VTensor [STensor m10, STensor m11]])
        result = readOpVar weights mem
        y = sum result
        g = collectGrads 1.0 y
        -- d_w0 = sum_j mem[0][j] = 1+2 = 3
        -- d_w1 = sum_j mem[1][j] = 3+4 = 7
    in check "readOpVar gradients"
       (abs (lookupGrad "t19w0" g - 3.0) < tol && abs (lookupGrad "t19w1" g - 7.0) < tol
        && abs (lookupGrad "t19m00" g - 0.6) < tol && abs (lookupGrad "t19m10" g - 0.4) < tol)

  -- C-backed writeOpVar: forward check
  , let w0 = cpuParam "t20w0" 1.0
        w1 = cpuParam "t20w1" 0.0
        m00 = cpuParam "t20m00" 1.0
        m01 = cpuParam "t20m01" 1.0
        m10 = cpuParam "t20m10" 1.0
        m11 = cpuParam "t20m11" 1.0
        e0 = cpuParam "t20e0" 1.0
        e1 = cpuParam "t20e1" 1.0
        a0 = cpuParam "t20a0" 0.5
        a1 = cpuParam "t20a1" 0.5
        weights = the (Vector 2 (Variable CPU)) (VTensor [STensor w0, STensor w1])
        mem = the (Matrix 2 2 (Variable CPU)) (VTensor [VTensor [STensor m00, STensor m01], VTensor [STensor m10, STensor m11]])
        erase = the (Vector 2 (Variable CPU)) (VTensor [STensor e0, STensor e1])
        add = the (Vector 2 (Variable CPU)) (VTensor [STensor a0, STensor a1])
        result = writeOpVar weights mem erase add
        (VTensor [VTensor [STensor r00, STensor r01], VTensor [STensor r10, STensor r11]]) = result
    in check "writeOpVar forward"
       (abs (r00.value - 0.5) < tol && abs (r01.value - 0.5) < tol
        && abs (r10.value - 1.0) < tol && abs (r11.value - 1.0) < tol)

  -- nativeAdamGroup filter: only updates params whose name starts with scope.
  -- (Inline wrapper because single-file `-o` invocation doesn't always see
  -- newly-exported idris-ml symbols; same workaround used in Example/A2c.idr.)
  , do
      let a_w = cpuParam "groupTest_a_w" 1.0
          b_w = cpuParam "groupTest_b_w" 1.0
          x2  = fromDouble 2.0
          y   = a_w * x2 + b_w * x2
          optA = mkAdamGroup "groupTest_a_"
      _ <- pure (nativeTrainStep optA y)
      let aAfter = (refreshValue a_w).value
          bAfter = (refreshValue b_w).value
      check "nativeAdamGroup filters by prefix"
        (abs (aAfter - 1.0) > 0.01 && abs (bAfter - 1.0) < tol)
  ]
