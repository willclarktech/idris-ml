module Init

import public Sampler
import Compat.Random


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
