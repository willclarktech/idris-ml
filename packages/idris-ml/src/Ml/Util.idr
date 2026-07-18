||| Things I wish were provided by the base library
module Ml.Util

import Data.Vect
import System.Clock

export
implementation Cast Bool Integer where
  cast True  = 1
  cast False = 0

export
implementation Cast Bool Double where
  cast = fromInteger . cast

export
signum : Double -> Double
signum x = case compare x 0 of
  GT => 1.0
  EQ => 0.0
  LT => -1.0

export
mean : (Num ty, Fractional ty) => List ty -> ty
mean xs =
  let tot = fromInteger $ natToInteger $ length xs
  in sum xs / tot

export
formatElapsed : Clock Monotonic -> Clock Monotonic -> String
formatElapsed t0 t1 =
  let totalSec = seconds t1 - seconds t0
      hh = totalSec `div` 3600
      mm = (totalSec `mod` 3600) `div` 60
      ss = totalSec `mod` 60
      pad : Integer -> String
      pad n = if n < 10 then "0" ++ show n else show n
  in "[" ++ pad hh ++ ":" ++ pad mm ++ ":" ++ pad ss ++ "]"

||| Format a duration in seconds as "Xm Ys" or "Xh Ym".
export
formatDuration : Integer -> String
formatDuration totalSec =
  let h = totalSec `div` 3600
      m = (totalSec `mod` 3600) `div` 60
      s = totalSec `mod` 60
  in if h > 0 then show h ++ "h " ++ show m ++ "m"
     else if m > 0 then show m ++ "m " ++ show s ++ "s"
     else show s ++ "s"

||| Format a timing summary: "Completed in Xm Ys (N epochs, Xms/epoch)"
export
formatTimingSummary : Clock Monotonic -> Clock Monotonic -> Nat -> String
formatTimingSummary t0 t1 epochs =
  let totalSec = seconds t1 - seconds t0
      dur = formatDuration totalSec
      msPerEpoch : Integer
      msPerEpoch = if epochs == 0 then 0
                   else (totalSec * 1000) `div` (natToInteger epochs)
  in "Completed in " ++ dur ++ " (" ++ show epochs ++ " epochs, "
     ++ show msPerEpoch ++ "ms/epoch)"

||| Float ms/epoch over the timed training loop, parsed by
||| `scripts/perf-baseline.sh` to drive in-script timing without the
||| two-point-subtraction methodology. Uses nanos so sub-second runs
||| don't truncate to zero.
export
formatPerfMsPerEp : Clock Monotonic -> Clock Monotonic -> Nat -> String
formatPerfMsPerEp t0 t1 epochs =
  if epochs == 0
    then "PERF_MS_PER_EP=0.000000"
    else
      let dsec  : Double; dsec  = cast (seconds t1 - seconds t0)
          dnano   : Double; dnano   = cast (nanoseconds t1 - nanoseconds t0)
          totalMs : Double; totalMs = dsec * 1000.0 + dnano / 1.0e6
          eps     : Double; eps     = cast (natToInteger epochs)
          msPerEp                   = totalMs / eps
      in "PERF_MS_PER_EP=" ++ show msPerEp

||| Sigmoid for Double values.
export
sigD : Double -> Double
sigD x = 1.0 / (1.0 + exp (negate x))

||| Round binary prediction: apply sigmoid then threshold at 0.5.
export
roundBinary : Double -> Double
roundBinary x = if sigD x >= 0.5 then 1.0 else 0.0

-- Note: showBinarySeq for Array types is in the examples since Util
-- doesn't import Array.

-- Copied from https://github.com/idris-lang/Idris2/pull/2707/files#diff-ff81a71a1254f20ad8ec34869deb9ada6f744fefee2e584c03a3c32367ddb8f7R395-R405
export
foldlD : (0 accTy : Nat -> Type) ->
  (f : forall k. accTy k -> a -> accTy (S k)) ->
  (acc : accTy Z) ->
  (xs : Vect n a) ->
  accTy n
foldlD _ _ acc []            = acc
foldlD accTy f acc (x :: xs) = foldlD (accTy . S) f (acc `f` x) xs
