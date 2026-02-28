module Init


----------------------------------------------------------------------
-- Weight Initialization Strategies
----------------------------------------------------------------------

||| Given (fanIn, fanOut), produce the uniform range limit.
||| Layers sample weights from U(-limit, limit).
public export
InitStrategy : Type
InitStrategy = (fanIn : Nat) -> (fanOut : Nat) -> Double

||| Xavier/Glorot uniform: limit = sqrt(6 / (fanIn + fanOut))
export
xavierInit : InitStrategy
xavierInit fanIn fanOut = Prelude.sqrt (6.0 / cast (fanIn + fanOut))

||| He/Kaiming uniform: limit = sqrt(6 / fanIn)
export
heInit : InitStrategy
heInit fanIn _ = Prelude.sqrt (6.0 / cast fanIn)

||| LeCun uniform: limit = sqrt(3 / fanIn)
export
lecunInit : InitStrategy
lecunInit fanIn _ = Prelude.sqrt (3.0 / cast fanIn)

||| Fixed range uniform (the old default): limit = bound
export
uniformInit : Double -> InitStrategy
uniformInit bound _ _ = bound
