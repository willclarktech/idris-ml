||| LR-range test (Smith 2017, popularized by fastai's `lr_find`).
|||
||| Trains for a small number of iterations with LR sweeping log-uniformly
||| from a tiny `lrMin` (default 1e-7) to a large `lrMax` (default 10),
||| records the smoothed loss at each LR, and recommends an LR.
|||
||| The recommendation is fastai's heuristic: take the iteration with the
||| steepest negative slope of smoothed loss vs log(LR), and divide that
||| LR by `recommendDiv` (default 10) to give the user some safety margin
||| before the loss starts diverging.
|||
||| Single-seed, single-batch by design — `lr_find` is a quick *screening*
||| tool; multi-seed validation belongs to whatever follow-up training
||| run consumes the recommendation.
|||
||| Note: this mutates the model and the optimizer's LR. Save the
||| optimizer's original LR before calling and restore it afterward, or
||| construct the optimizer fresh after the recommendation.
module Hpo.LrFinder

import Data.List
import System.Clock

import Util
import Variable


----------------------------------------------------------------------
-- Configuration
----------------------------------------------------------------------

||| LR-range-test configuration. `defaultLrFindConfig` matches fastai's
||| defaults; tweak `numIters` if your batches are slow.
public export
record LrFindConfig where
  constructor MkLrFindConfig
  ||| Smallest LR to probe. Default 1e-7.
  lrMin : Double
  ||| Largest LR to probe. Default 10.
  lrMax : Double
  ||| Number of iterations across the LR sweep. Default 100.
  numIters : Nat
  ||| EMA smoothing factor for loss. Default 0.98.
  smoothBeta : Double
  ||| Stop early if smoothed loss exceeds `divergeFactor * minSmoothed`.
  ||| Default 4.0; set to a very large number to disable.
  divergeFactor : Double
  ||| Recommend `lr_at_steepest_descent / recommendDiv`. Default 10.
  recommendDiv : Double

export
defaultLrFindConfig : LrFindConfig
defaultLrFindConfig = MkLrFindConfig 1.0e-7 10.0 100 0.98 4.0 10.0


----------------------------------------------------------------------
-- Result
----------------------------------------------------------------------

||| Result of an LR-range test.
|||
||| `points` is a list of `(lr, smoothedLoss)` pairs in iteration order.
||| Iterations beyond divergence are omitted.
public export
record LrFindResult where
  constructor MkLrFindResult
  points : List (Double, Double)
  recommendedLr : Double


----------------------------------------------------------------------
-- LR schedule for the sweep
----------------------------------------------------------------------

||| LR at iteration `i` of `n` total: `lrMin * (lrMax/lrMin) ^ (i / (n-1))`
||| in log-space. Endpoints are exact: `sweepLr 0 = lrMin`,
||| `sweepLr (n-1) = lrMax`.
export
sweepLr : (lrMin : Double) -> (lrMax : Double) -> (n : Nat) -> (i : Nat) -> Double
sweepLr lrMin lrMax n i =
  if n <= 1 then lrMin
  else
    let frac = cast {to=Double} (cast {to=Integer} i)
              / cast {to=Double} (cast {to=Integer} (n `minus` 1))
        logRatio = Prelude.log (lrMax / lrMin)
    in lrMin * Prelude.exp (frac * logRatio)


----------------------------------------------------------------------
-- Recommended-LR heuristic: steepest negative slope of smoothed loss
-- vs log(lr), divided by `recommendDiv`.
----------------------------------------------------------------------

-- Adjacent slopes: (lr_i, smoothed_i, smoothed_{i+1} - smoothed_i)
-- Drop the last point (no successor).
slopes : List (Double, Double) -> List (Double, Double)
slopes [] = []
slopes [_] = []
slopes ((lr0, l0) :: (lr1, l1) :: rest) =
  (lr0, l1 - l0) :: slopes ((lr1, l1) :: rest)

-- Pick the LR with the most-negative slope. Falls back to the smallest
-- LR if all slopes are non-negative (loss never decreased — unusual).
steepestDescent : List (Double, Double) -> Double
steepestDescent [] = 0.0
steepestDescent ((lr, s) :: rest) = go lr s rest
  where
    go : Double -> Double -> List (Double, Double) -> Double
    go bestLr _ [] = bestLr
    go bestLr bestSlope ((lr', s') :: more) =
      if s' < bestSlope then go lr' s' more else go bestLr bestSlope more

||| Recommend an LR from the swept (lr, smoothedLoss) curve. fastai's
||| heuristic: take the LR at steepest descent and divide by
||| `recommendDiv` (typically 10) for a safety margin.
export
recommendFromCurve : Double -> List (Double, Double) -> Double
recommendFromCurve recommendDiv curve =
  steepestDescent (slopes curve) / recommendDiv


||| Sign-stable divergence check. Returns `True` when the current
||| smoothed loss has worsened by more than `(divergeFactor - 1) × |best|`
||| above the best smoothed loss seen so far. For positive losses this
||| matches the classical `corrected > divergeFactor × best` rule. For
||| negative losses (e.g. RL examples reporting `negate avg_return` as
||| the "loss"), the classical rule trips at iter 1 because
||| `divergeFactor × negative_best` is below any reasonable corrected
||| value; this version stays correct.
export
hasDiverged : (divergeFactor : Double) -> (best : Double) -> (corrected : Double) -> Bool
hasDiverged divergeFactor best corrected =
  let absRef = if abs best < 1.0e-8 then 1.0e-8 else abs best
  in (corrected - best) > (divergeFactor - 1.0) * absRef


||| `True` when the swept (lr, smoothedLoss) curve has no usefully-
||| descending region. In that case, `recommendFromCurve` falls back
||| to `lrMin / recommendDiv` and the recommendation is meaningless.
||| Fallback triggers:
||| - fewer than two points (no slope to measure)
||| - all slopes non-negative (loss never decreased)
|||
||| Callers should treat a fallback recommendation as a "lr_find could
||| not find a useful LR" signal rather than acting on the value.
||| Common in flat-curve regimes: small architectures, Adam already
||| adapting effective LR per parameter, or the LR sweep range missing
||| the actual sweet spot.
export
isFallbackCurve : List (Double, Double) -> Bool
isFallbackCurve curve =
  case slopes curve of
    []  => True
    ss  => all (\(_, s) => s >= 0.0) ss


----------------------------------------------------------------------
-- Main loop
----------------------------------------------------------------------

||| Run the LR range test.
|||
||| Mutates the model and the optimizer's LR. The caller is responsible
||| for using the returned recommendation against a *fresh* optimizer or
||| for restoring the original LR before resuming training.
|||
||| Stdout output: one line per iteration as
|||   `iter\t<i>\tlr\t<lr>\tloss\t<loss>\tsmoothed\t<smoothed>`
||| followed by a final
|||   `RECOMMENDED_LR=<value>`
||| line. Plot the points externally to inspect the curve.
export
lrFind : {0 model : Type} -> {0 dp : Type} ->
         LrFindConfig ->
         (epochFn : model -> dp -> IO (model, Double)) ->
         (dataSrc : IO dp) ->
         NativeOptimizer ->
         model ->
         IO LrFindResult
lrFind {model} cfg epochFn dataSrc opt model0 = do
  putStrLn $ "lr_find: sweeping LR from " ++ show cfg.lrMin
           ++ " to " ++ show cfg.lrMax
           ++ " over " ++ show cfg.numIters ++ " iters"
  tStart <- clockTime Monotonic
  result <- go 0 model0 0.0 (1.0 / 0.0) []
  tEnd <- clockTime Monotonic
  putStrLn $ "lr_find done in " ++ formatElapsed tStart tEnd
  when (isFallbackCurve (points result)) $
    putStrLn "WARNING: fallback recommendation — no negative-slope window in the swept curve."
  putStrLn $ "RECOMMENDED_LR=" ++ show (recommendedLr result)
  pure result
  where
    -- (i, model, prevSmoothed, minSmoothed, accPoints) -> result
    go : Nat -> model -> Double -> Double -> List (Double, Double) ->
         IO LrFindResult
    go i m prevSmoothed minSmoothed accRev =
      if i >= cfg.numIters then
        let pts = reverse accRev
            rec = recommendFromCurve cfg.recommendDiv pts
        in pure (MkLrFindResult pts rec)
      else do
        let lr = sweepLr cfg.lrMin cfg.lrMax cfg.numIters i
        setLearningRate opt lr
        d <- dataSrc
        (m', loss) <- epochFn m d
        let beta = cfg.smoothBeta
            -- bias-corrected EMA: avoids the initial-bias toward zero
            avg = beta * prevSmoothed + (1.0 - beta) * loss
            iD = cast {to=Double} (cast {to=Integer} i)
            corrected = avg / (1.0 - Prelude.pow beta (iD + 1.0))
            minS' = if corrected < minSmoothed then corrected else minSmoothed
            point = (lr, corrected)
            accRev' = point :: accRev
        putStrLn $ "  iter\t" ++ show i ++ "\tlr\t" ++ show lr
                 ++ "\tloss\t" ++ show loss
                 ++ "\tsmoothed\t" ++ show corrected
        let diverged = hasDiverged cfg.divergeFactor minS' corrected && i > 0
        if diverged
          then do
            putStrLn $ "  (diverged at iter " ++ show i
                     ++ ", smoothed=" ++ show corrected
                     ++ " > min=" ++ show minS'
                     ++ " + " ++ show (cfg.divergeFactor - 1.0)
                     ++ "*|min|)"
            let pts = reverse accRev'
                rec = recommendFromCurve cfg.recommendDiv pts
            pure (MkLrFindResult pts rec)
          else go (i + 1) m' avg minS' accRev'
