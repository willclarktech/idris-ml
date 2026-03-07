module Init

import public Sampler
import System.Random


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
