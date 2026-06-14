module Schedule

----------------------------------------------------------------------
-- Schedule Type
----------------------------------------------------------------------

||| A schedule maps epoch number (0-indexed) to learning rate.
public export
Schedule : Type
Schedule = Nat -> Double

----------------------------------------------------------------------
-- Constant
----------------------------------------------------------------------

||| Fixed learning rate across all epochs.
export
constant : Double -> Schedule
constant lr _ = lr

----------------------------------------------------------------------
-- Cosine Annealing
----------------------------------------------------------------------

||| Cosine decay from lrMax to lrMin over totalEpochs.
||| After totalEpochs, returns lrMin.
export
cosineAnnealing : (lrMax : Double) -> (lrMin : Double) -> (totalEpochs : Nat) -> Schedule
cosineAnnealing lrMax lrMin totalEpochs epoch =
  if epoch >= totalEpochs then lrMin
  else
    let t = cast epoch / cast totalEpochs
    in lrMin + 0.5 * (lrMax - lrMin) * (1.0 + cos (t * pi))

----------------------------------------------------------------------
-- One-Cycle (fastai)
----------------------------------------------------------------------

||| Fastai's one-cycle policy: linear warmup then cosine annealing.
|||
||| Phase 1 (warmup): linear from lrMax/div to lrMax over pctStart fraction.
||| Phase 2 (decay): cosine from lrMax to lrMax/divFinal over remaining.
|||
||| Typical defaults: div=25, divFinal=1e5, pctStart=0.25
export
oneCycle : (lrMax : Double) -> (div : Double) -> (divFinal : Double)
        -> (pctStart : Double) -> (totalEpochs : Nat) -> Schedule
oneCycle lrMax div divFinal pctStart totalEpochs epoch =
  let totalD = cast {to=Double} totalEpochs
      warmupEnd = cast {to=Nat} (pctStart * totalD)
      lrMin     = lrMax / div
      lrFinal   = lrMax / divFinal
  in if epoch >= totalEpochs then lrFinal
     else if epoch < warmupEnd
       then -- Phase 1: linear warmup
         let t = cast epoch / cast warmupEnd
         in lrMin + t * (lrMax - lrMin)
       else -- Phase 2: cosine annealing
         let decayEpochs = totalEpochs `minus` warmupEnd
             t = cast (epoch `minus` warmupEnd) / cast decayEpochs
         in lrFinal + 0.5 * (lrMax - lrFinal) * (1.0 + cos (t * pi))

----------------------------------------------------------------------
-- Warmup wrapper
----------------------------------------------------------------------

||| Linear warmup from startLR to the base schedule's value at warmupEnd.
||| After warmupEpochs, delegates to the base schedule (shifted).
export
withWarmup : (warmupEpochs : Nat) -> (startLR : Double) -> Schedule -> Schedule
withWarmup warmupEpochs startLR base epoch =
  if epoch < warmupEpochs
    then let targetLR = base warmupEpochs
             t = cast epoch / cast warmupEpochs
         in startLR + t * (targetLR - startLR)
    else base epoch

----------------------------------------------------------------------
-- Cosine with warmup (standard transformer recipe)
----------------------------------------------------------------------

||| Cosine annealing with linear warmup. The standard modern transformer LR schedule.
export
cosineWithWarmup : (lrMax : Double) -> (lrMin : Double)
                -> (warmupEpochs : Nat) -> (totalEpochs : Nat) -> Schedule
cosineWithWarmup lrMax lrMin warmupEpochs =
  withWarmup warmupEpochs lrMin . cosineAnnealing lrMax lrMin

----------------------------------------------------------------------
-- Step LR
----------------------------------------------------------------------

||| Reduce LR by a factor every stepSize epochs.
||| lr = baseLR * gamma ^ (epoch / stepSize)
export
stepLR : (baseLR : Double) -> (stepSize : Nat) -> (gamma : Double) -> Schedule
stepLR baseLR stepSize gamma epoch =
  let step = cast {to=Double} epoch / cast {to=Double} (max stepSize 1)
  in baseLR * pow gamma (cast {to=Double} (cast {to=Integer} step))

----------------------------------------------------------------------
-- Exponential LR
----------------------------------------------------------------------

||| Exponential decay: lr = baseLR * gamma ^ epoch.
export
exponentialLR : (baseLR : Double) -> (gamma : Double) -> Schedule
exponentialLR baseLR gamma epoch =
  baseLR * pow gamma (cast epoch)
