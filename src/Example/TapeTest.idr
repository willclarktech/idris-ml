module Example.TapeTest

import Data.SortedMap
import Floating
import Variable

-- Quick smoke test for tape-based autograd gradients.
main : IO ()
main = do
  putStrLn "=== Tape gradient tests ==="

  -- Test 1: y = w*x + b => dy/dw = x, dy/db = 1
  let w = param "w" 3.0
  let b = param "b" 1.0
  let x = fromDouble 2.0
  let y = w * x + b
  let g1 = Data.SortedMap.toList $ collectGrads 1.0 y
  putStr "1. w*x+b grads: "
  printLn g1  -- expect [("b",1.0), ("w",2.0)]

  -- Test 2: y = exp(w) => dy/dw = exp(w)
  let w2 = param "w2" 1.0
  let y2 = exp w2
  let g2 = Data.SortedMap.toList $ collectGrads 1.0 y2
  putStr "2. exp(w) grads: "
  printLn g2  -- expect [("w2", e)]

  -- Test 3: y = a / b => dy/da = 1/b, dy/db = -a/b^2
  let a = param "a" 6.0
  let b3 = param "b3" 3.0
  let y3 = a / b3
  let g3 = Data.SortedMap.toList $ collectGrads 1.0 y3
  putStr "3. a/b grads:   "
  printLn g3  -- expect [("a", 0.333), ("b3", -0.667)]

  -- Test 4: dot product w . x => grads = x components
  let w0 = param "w0" 0.5
  let w1 = param "w1" 0.5
  let x0 = fromDouble 1.0
  let x1 = fromDouble 2.0
  let y4 = w0 * x0 + w1 * x1
  let g4 = Data.SortedMap.toList $ collectGrads 1.0 y4
  putStr "4. dot product:  "
  printLn g4  -- expect [("w0",1.0), ("w1",2.0)]
