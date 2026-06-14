module Init

import Data.Vect

import public Sampler
import Compat.Random

----------------------------------------------------------------------
-- InitSpec (typed construction facade)
----------------------------------------------------------------------

||| How to fill a tensor of `n` elements. The index ties `FromVect`'s
||| length to the target shape's element count at compile time —
||| a length mismatch is a type error, not a runtime crash.
public export
data InitSpec : Nat -> Type where
  Zeros    : InitSpec n
  Const    : Double -> InitSpec n
  Normal   : (mu : Double) -> (sd : Double) -> InitSpec n
  Uniform  : (lo : Double) -> (hi : Double) -> InitSpec n
  FromVect : Vect n Double -> InitSpec n

||| Element count of a shape. The singleton clause keeps `Numel [n]`
||| definitionally `n` (no `n * 1` wart), so 1-D `FromVect` literals
||| check without rewriting.
public export
Numel : Vect rank Nat -> Nat
Numel []        = 1
Numel [n]       = n
Numel (n :: ns) = n * Numel ns

||| Stack rows into the `FromVect` spec for a `[b, i]` tensor.
public export
fromRows : Vect b (Vect i Double) -> InitSpec (b * i)
fromRows rows = FromVect (concat rows)

----------------------------------------------------------------------
-- Init Strategies (method + distribution -> one weight sample)
----------------------------------------------------------------------

||| Given (fanIn, fanOut), produce one random weight sample in IO.
public export
InitStrategy : Type
InitStrategy = (fanIn : Nat) -> (fanOut : Nat) -> IO Double

||| Xavier/Glorot: variance = 2 / (fanIn + fanOut)
export
xavier : Sampler -> InitStrategy
xavier sampler fanIn fanOut = sampler (2.0 / cast (fanIn + fanOut))

||| He/Kaiming: variance = 2 / fanIn
export
he : Sampler -> InitStrategy
he sampler fanIn _ = sampler (2.0 / cast fanIn)

||| LeCun: variance = 1 / fanIn
export
lecun : Sampler -> InitStrategy
lecun sampler fanIn _ = sampler (1.0 / cast fanIn)

||| Xavier/Glorot with gain: variance = 2 * gain^2 / (fanIn + fanOut)
export
xavierGain : Double -> Sampler -> InitStrategy
xavierGain gain sampler fanIn fanOut = sampler (2.0 * gain * gain / cast (fanIn + fanOut))

||| Fixed range U(-bound, bound), ignoring dimensions.
export
fixedRange : Double -> InitStrategy
fixedRange bound _ _ = randomRIO (-bound, bound)

||| PyTorch's default `nn.init.kaiming_uniform_(tensor)` bound is
||| 1/sqrt(fan_in) — equivalent to Kaiming with `a=sqrt(5)`,
||| `nonlinearity='leaky_relu'`, `mode='fan_in'`. In variance terms,
||| this is `var = 1/(3*fan_in)` for the uniform sampler (since
||| `uniform var` produces U(-sqrt(3v), sqrt(3v))).
|||
||| This is what `nn.Linear` uses by default and what the NTM/DNC
||| output FC uses explicitly.
export
ptKaimingDefault : Sampler -> InitStrategy
ptKaimingDefault sampler fanIn _ = sampler (1.0 / (3.0 * cast fanIn))
