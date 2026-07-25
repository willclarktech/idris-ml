||| Effectful samplers over the process-global generator.
|||
||| The distribution shapes themselves live in `Random.Dist` and are pure;
||| these draw the raw uniforms ambiently and hand them straight to it via a
||| `Recorded` source. One implementation of Box-Muller and of the categorical
||| walk, used both purely and effectfully.
module Ml.Sampler

import Random.Dist
import Random.Source

import Ml.Compat.Random

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

||| Standard normal sample N(0,1) via Box-Muller.
|||
||| The low end of the first draw is bounded here rather than left to
||| `Dist.normal`'s own clamp, so the two uniforms consumed are exactly the
||| ones this has always drawn.
export
normalSample : IO Double
normalSample = do
  u1 <- randomRIO (the Double 1.0e-10, 1.0)
  u2 <- randomRIO (the Double 0.0, 1.0)
  pure (fst (Dist.normal (Recorded [u1, u2])))

||| Normal sampler: N(0, sqrt(v)), which has variance v.
export
normal : Sampler
normal var = map (* prim__doubleSqrt var) normalSample

----------------------------------------------------------------------
-- Categorical sampling
----------------------------------------------------------------------

||| Sample from a categorical distribution via cumulative sum.
||| @probs  probability for each category (should sum to ~1.0)
||| @r      uniform random value in [0, 1)
||| Returns the index of the sampled category.
export
categoricalSample : List Double -> Double -> Nat
categoricalSample probs r = fst (Dist.categorical (Recorded [r]) probs)
