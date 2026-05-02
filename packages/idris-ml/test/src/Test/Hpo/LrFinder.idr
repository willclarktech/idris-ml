module Test.Hpo.LrFinder

import Data.List
import Data.Vect

import Harness
import Hpo.LrFinder


tol : Double
tol = 1.0e-6


-- Helper: the LR at the steepest negative slope of the curve, divided by 10.
-- Uses recommendFromCurve directly, so this test pins the heuristic.
syntheticUCurve : List (Double, Double)
syntheticUCurve =
  -- A canonical LR-find shape: log-uniform LRs from 1e-4 to 1.0, with
  -- losses that decrease (steep negative slope between iter 1 and 2),
  -- bottom out, then explode. Steepest descent is at lr=0.001.
  [ (1.0e-4, 1.00)  -- flat
  , (1.0e-3, 0.95)  -- slope -0.45 (after this entry; loss at next is 0.50)
  , (1.0e-2, 0.50)  -- slope -0.40
  , (1.0e-1, 0.10)  -- slope +1.90 (diverges)
  , (1.0,    2.00)
  ]


export
tests : List (IO Bool)
tests =
  [ -- sweepLr endpoints
    checkClose "sweepLr i=0 = lrMin" 1.0e-7 (sweepLr 1.0e-7 10.0 100 0) tol
  , checkClose "sweepLr i=n-1 = lrMax" 10.0 (sweepLr 1.0e-7 10.0 100 99) tol

  -- sweepLr midpoint of log-uniform sweep
  -- frac=0.5 → lr = lrMin * sqrt(lrMax/lrMin) = 1e-7 * sqrt(1e8) = 1e-7 * 1e4 = 1e-3
  , checkClose "sweepLr midpoint" 1.0e-3 (sweepLr 1.0e-7 10.0 3 1) tol

  -- recommendFromCurve picks the LR at steepest descent ÷ recommendDiv.
  -- The synthetic curve has steepest descent at lr=1e-3 (slope -0.45);
  -- with recommendDiv=10, recommended = 1e-4.
  , checkClose "recommendFromCurve picks steepest descent / 10"
      1.0e-4 (recommendFromCurve 10.0 syntheticUCurve) tol

  -- recommendFromCurve with recommendDiv=1 returns the steepest LR itself
  , checkClose "recommendFromCurve with div=1"
      1.0e-3 (recommendFromCurve 1.0 syntheticUCurve) tol

  -- defaultLrFindConfig is the fastai-aligned default
  , check "default lrMin = 1e-7" (defaultLrFindConfig.lrMin == 1.0e-7)
  , check "default lrMax = 10.0" (defaultLrFindConfig.lrMax == 10.0)
  , check "default numIters = 100" (defaultLrFindConfig.numIters == 100)
  , check "default smoothBeta = 0.98" (defaultLrFindConfig.smoothBeta == 0.98)
  , check "default recommendDiv = 10" (defaultLrFindConfig.recommendDiv == 10.0)
  ]
