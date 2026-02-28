module Sampler

import System.Random


----------------------------------------------------------------------
-- Samplers (distribution shapes)
----------------------------------------------------------------------

||| Given a target variance, produce one random sample in IO.
public export
Sampler : Type
Sampler = Double -> IO Double

||| Uniform sampler: U(-sqrt(3v), sqrt(3v)), which has variance v.
export
uniform : Sampler
uniform var = do
  let limit = prim__doubleSqrt (3.0 * var)
  randomRIO (-limit, limit)

||| Standard normal sample N(0,1) via Box-Muller transform.
export
normalSample : IO Double
normalSample = do
  u1 <- randomRIO (the Double 1.0e-10, 1.0)
  u2 <- randomRIO (the Double 0.0, 1.0)
  pure $ prim__doubleSqrt (-2.0 * prim__doubleLog u1)
       * prim__doubleCos (2.0 * 3.141592653589793 * u2)

||| Normal sampler: N(0, sqrt(v)), which has variance v.
export
normal : Sampler
normal var = map (* prim__doubleSqrt var) normalSample
