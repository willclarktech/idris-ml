module Test.Variable

import Data.SortedMap
import Data.Vect

import Harness
import Floating
import Math
import Tensor
import Variable


tol : Double
tol = 1.0e-4

lookupGrad : String -> SortedMap String Double -> Double
lookupGrad key m = case lookup key m of
  Just v => v
  Nothing => 0.0 / 0.0  -- NaN to signal missing

export
tests : List (IO Bool)
tests =
  [ -- Test 1 (from TapeTest): y = w*x + b => dy/dw = x, dy/db = 1
    let w = param "t1w" 3.0
        b = param "t1b" 1.0
        x = fromDouble 2.0
        y = w * x + b
        g = collectGrads 1.0 y
    in check "w*x+b: dw=x, db=1"
       (abs (lookupGrad "t1w" g - 2.0) < tol && abs (lookupGrad "t1b" g - 1.0) < tol)

  -- Test 2 (from TapeTest): y = exp(w) => dy/dw = exp(w)
  , let w = param "t2w" 1.0
        y = exp w
        g = collectGrads 1.0 y
    in checkClose "exp(w): dw=exp(w)" (exp 1.0) (lookupGrad "t2w" g) tol

  -- Test 3 (from TapeTest): y = a/b => dy/da = 1/b, dy/db = -a/b^2
  , let a = param "t3a" 6.0
        b = param "t3b" 3.0
        y = a / b
        g = collectGrads 1.0 y
    in check "a/b: da=1/b, db=-a/b^2"
       (abs (lookupGrad "t3a" g - 1.0/3.0) < tol && abs (lookupGrad "t3b" g - (-6.0/9.0)) < tol)

  -- Test 4 (from TapeTest): dot product
  , let w0 = param "t4w0" 0.5
        w1 = param "t4w1" 0.5
        x0 = fromDouble 1.0
        x1 = fromDouble 2.0
        y = w0 * x0 + w1 * x1
        g = collectGrads 1.0 y
    in check "dot product grads = x"
       (abs (lookupGrad "t4w0" g - 1.0) < tol && abs (lookupGrad "t4w1" g - 2.0) < tol)

  -- Subtraction
  , let a = param "t5a" 5.0
        b = param "t5b" 3.0
        y = a - b
        g = collectGrads 1.0 y
    in check "a-b: da=1, db=-1"
       (abs (lookupGrad "t5a" g - 1.0) < tol && abs (lookupGrad "t5b" g - (-1.0)) < tol)

  -- Negate
  , let w = param "t6w" 4.0
        y = negate w
        g = collectGrads 1.0 y
    in checkClose "negate: dw=-1" (-1.0) (lookupGrad "t6w" g) tol

  -- Sqrt
  , let w = param "t7w" 4.0
        y = sqrt w
        g = collectGrads 1.0 y
    in checkClose "sqrt(w): dw=1/(2*sqrt(w))" (1.0 / (2.0 * Prelude.sqrt 4.0)) (lookupGrad "t7w" g) tol

  -- Power
  , let w = param "t8w" 3.0
        y = pow w (fromDouble 2.0)
        g = collectGrads 1.0 y
    in checkClose "w^2: dw=2w" 6.0 (lookupGrad "t8w" g) tol

  -- Abs (positive)
  , let w = param "t9w" 3.0
        y = abs w
        g = collectGrads 1.0 y
    in checkClose "abs(3): dw=1" 1.0 (lookupGrad "t9w" g) tol

  -- Abs (negative)
  , let w = param "t10w" (-3.0)
        y = abs w
        g = collectGrads 1.0 y
    in checkClose "abs(-3): dw=-1" (-1.0) (lookupGrad "t10w" g) tol

  -- Chain rule: exp(w*x)
  , let w = param "t11w" 2.0
        x = fromDouble 3.0
        y = exp (w * x)
        g = collectGrads 1.0 y
    in checkClose "exp(w*x): dw = x*exp(w*x)" (3.0 * exp 6.0) (lookupGrad "t11w" g) (exp 6.0 * 1.0e-6)

  -- Shared subexpression: y = w*w, dy/dw = 2w
  , let w = param "t12w" 5.0
        y = w * w
        g = collectGrads 1.0 y
    in checkClose "w*w: dw=2w" 10.0 (lookupGrad "t12w" g) tol

  -- C-backed dotProductVar
  , let a0 = param "t13a0" 1.0
        a1 = param "t13a1" 2.0
        b0 = param "t13b0" 3.0
        b1 = param "t13b1" 4.0
        va = the (Vector 2 Variable) (VTensor [STensor a0, STensor a1])
        vb = the (Vector 2 Variable) (VTensor [STensor b0, STensor b1])
        y = dotProductVar va vb
        g = collectGrads 1.0 y
    in check "dotProductVar: grads match"
       (abs (lookupGrad "t13a0" g - 3.0) < tol && abs (lookupGrad "t13a1" g - 4.0) < tol
        && abs (lookupGrad "t13b0" g - 1.0) < tol && abs (lookupGrad "t13b1" g - 2.0) < tol)

  -- C-backed softmaxVar sums to 1
  , let x0 = param "t14x0" 1.0
        x1 = param "t14x1" 2.0
        x2 = param "t14x2" 3.0
        v = the (Vector 3 Variable) (VTensor [STensor x0, STensor x1, STensor x2])
        sm = softmaxVar v
        s = sum sm
    in checkClose "softmaxVar sums to 1" 1.0 s.value tol

  -- Stale variable re-registration
  , let w = param "t15w" 7.0
        y1 = w * (fromDouble 2.0)
        _ = collectGrads 1.0 y1  -- resets tape (gen++)
        -- w is now stale; using it again should re-register
        y2 = w * (fromDouble 3.0)
        g2 = collectGrads 1.0 y2
    in checkClose "stale re-registration" 3.0 (lookupGrad "t15w" g2) tol
  ]
