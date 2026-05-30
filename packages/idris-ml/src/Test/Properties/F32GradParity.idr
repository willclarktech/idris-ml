-- Test.Properties.F32GradParity — F32 vs F64 paired-oracle gradient parity.
--
-- Tests: backward-pass-derived gradient on a F32 param matches the
-- backward-pass-derived gradient on a F64 param computing the same
-- mathematical function, within F32-precision tolerance.
--
-- For loss `L(p) = p * p`, dL/dp = 2*p. Both rungs should produce
-- ≈ 2*x; their difference should be bounded by F32 round-off
-- (~1e-7 relative at unit magnitudes).
--
-- Implementation pinned to `TapeDev` because:
--   (a) Phase 3b's F32 storage routing exists on tape via the
--       `tape_load_d` dispatch; the parity test directly exercises
--       that path.
--   (b) tape always admits both Compatible (TapeDev, F32) and
--       Compatible (TapeDev, F64), so the source elaborates on
--       every build config.
--   (c) tape symbols are linked whenever PRIMARY=tape; on other
--       backends the test gates at runtime via TestPrimaryBackend.
--
-- Param-registry note: `tparamScalar` registers each param under a
-- distinct name (see feedback_param_registry_dedup — the C-side
-- registry dedupes by name). Each iteration of `checkPropertyIOn`
-- generates a fresh paramId derived from the registry count, so
-- repeated runs don't collide.
--
-- Uses `checkPropertyIO` since the body calls IO-typed smart
-- constructors and FFI gradient readout. See
-- `packages/idris-test/src/Test/Property.idr` for the trade-off
-- (no shrinking).
module Test.Properties.F32GradParity

import Data.Vect

import Test.Property
import Test.Config
import Test.Harness as Harness

import Device
import Device.Tape
import Tensor

%default partial

-- Build the loss and read out the gradient for one dtype rung.
-- Returns the gradient value read from the param registry.
runRung : (dt : DType) ->
          IsFloating dt =>
          RuntimeDType dt =>
          Compatible TapeDev dt =>
          String -> Double -> IO Double
runRung dt pidPrefix x = do
  countBefore <- getParamCount {d = TapeDev}
  let pid = pidPrefix ++ "_" ++ show countBefore
  p <- tparamScalar {d = TapeDev} {dt} pid x
  loss <- tmul p p
  runBackward loss
  getParamGradAt {d = TapeDev} countBefore 0

prop_f32_grad_matches_f64_body : Double -> IO Bool
prop_f32_grad_matches_f64_body x = do
  gradF32 <- runRung F32 "f32gp_f32" x
  gradF64 <- runRung F64 "f32gp_f64" x
  -- Expected grad ≈ 2*x. Tolerance combines absolute floor (for
  -- near-zero x where relative is meaningless) with a 1e-3 relative
  -- bound that covers F32 ULP error at modest magnitudes.
  let absDiff = abs (gradF32 - gradF64)
      tol     = 1.0e-3 * abs x + 1.0e-6
  if absDiff < tol
    then pure True
    else do
      putStrLn $ "    gradF32  = " ++ show gradF32
      putStrLn $ "    gradF64  = " ++ show gradF64
      putStrLn $ "    expected ≈ " ++ show (2.0 * x)
      putStrLn $ "    abs-diff = " ++ show absDiff ++ "  (tol " ++ show tol ++ ")"
      pure False

prop_f32_grad_matches_f64 : IO Bool
prop_f32_grad_matches_f64 =
  if TestPrimaryBackend == "tape"
    then checkPropertyIOn
           "f32_grad_matches_f64"
           25
           (double (linearFracFrom 0.0 (-1.0) 1.0))
           prop_f32_grad_matches_f64_body
    else Harness.check
           ("f32_grad_matches_f64 (SKIPPED: requires tape primary, "
            ++ "active=" ++ TestPrimaryBackend ++ ")")
           True

export
tests : List (IO Bool)
tests = [ prop_f32_grad_matches_f64 ]
