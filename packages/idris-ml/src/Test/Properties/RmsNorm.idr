-- Test.Properties.RmsNorm — norm-bounded invariant for RMSNorm.
--
-- Tests: for any non-degenerate input x of dim n, RMSNorm with
-- weight=1 produces a tensor whose L2 norm is approximately √n.
--
-- Algebra (weight = 1, default formulation):
--   out[i]  = x[i] / sqrt(mean(x²) + eps)
--   ‖out‖²  = Σᵢ x[i]² / (mean(x²) + eps)
--            = (n · mean(x²)) / (mean(x²) + eps)
--            ≈ n  whenever  eps << mean(x²)
--
-- So ‖out‖ ≈ √n for non-degenerate inputs. This is the structural
-- invariant the property pins; complementary to `Test.RmsNorm`'s
-- single PyTorch-oracle case at fixed input [1,2,3,4].
--
-- Uses `checkPropertyIO` (not Hedgehog's `Property`) because the body
-- calls IO-typed smart constructors (`rmsNormLayer`, `applyRmsNormEps`)
-- which the pure-functional `TestT Gen` runner can't host. Trade-off
-- documented in `packages/idris-test/src/Test/Property.idr`.
module Test.Properties.RmsNorm

import Data.Vect

import Test.Property
import Test.Config
import Test.Harness as Harness

import Executor
import Tensor
import Array
import Nn.Init
import Nn.RmsNorm

%default partial

-- Fixed dim. The invariant holds for any n; pinning n=32 keeps the
-- counterexample `Show` output one screen wide and dodges the
-- dependent-Vect generator dance (Gen (DPair Nat (\n => Vect n Double))).
NN : Nat
NN = 32

readVec : (n : Nat) -> AnyPtr -> IO (List Double)
readVec n p = go (cast {to=Int} n) 0 []
  where
    go : Int -> Int -> List Double -> IO (List Double)
    go end i acc =
      if i >= end
        then pure (reverse acc)
        else let v = primItem1d {ex=TestExecutor} p i
             in go end (i + 1) (v :: acc)

mkInput : {n : Nat} -> Vect n Double -> Tensor [n] TestExecutor TestDType WithGrad
mkInput xs =
  let raw = bulkToTensor {ex=TestExecutor} {dt=TestDType}
                         (VArray (map SArray xs))
  in tinput1d {n} raw

l2Norm : List Double -> Double
l2Norm xs = sqrt (sum (map (\v => v * v) xs))

-- Skip near-zero inputs where eps would dominate and the invariant
-- doesn't apply. mean(x²) >= 0.01 with eps=1e-5 keeps the eps term's
-- contribution to ‖out‖₂² below 1e-3 — outside our 1e-3 rel-diff tol.
isDegenerate : Vect n Double -> Bool
isDegenerate xs =
  let nD     = the Double (cast (length (toList xs)))
      meanSq = the Double (sum (map (\v : Double => v * v) (toList xs)) / nD)
  in meanSq < 0.01

prop_rmsnorm_output_bounded_body : Vect NN Double -> IO Bool
prop_rmsnorm_output_bounded_body xs =
  if isDegenerate xs
    then pure True  -- input fails the eps << mean(x²) precondition
    else do
      rms <- runInit (rmsNorm {ex=TestExecutor} {dt=TestDType} {n=NN})
      out <- rmsNormForward 1.0e-5 rms (mkInput xs)
      vals <- readVec NN out.tensorPtr
      let actualNorm   = l2Norm vals
          expectedNorm = sqrt (cast {to=Double} NN)
          relDiff      = abs (actualNorm - expectedNorm) / expectedNorm
      if relDiff < 1.0e-3
        then pure True
        else do
          putStrLn $ "    ‖out‖₂   = " ++ show actualNorm
          putStrLn $ "    √n       = " ++ show expectedNorm
          putStrLn $ "    relDiff  = " ++ show relDiff ++ "  (tol 1e-3)"
          pure False

prop_rmsnorm_output_bounded : IO Bool
prop_rmsnorm_output_bounded = checkPropertyIOn
  "rmsnorm_output_bounded"
  25  -- each case drives an FFI roundtrip; 25 keeps wall short
  (vect NN (double (linearFracFrom 0.0 (-10.0) 10.0)))
  prop_rmsnorm_output_bounded_body

export
tests : List (IO Bool)
tests = [ prop_rmsnorm_output_bounded ]
