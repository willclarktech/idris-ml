-- Test.LoadOpts — typed Checkpoint.load surface.
--
-- One test per LoadError constructor reachable from Idris-side file
-- ops (FileNotFound, MalformedFile, DtypeMismatch, ShapeMismatch),
-- plus the allowCast lift and the `only` prefix filter.
-- UnsupportedDtype / ReadFailed need hand-crafted binary containers —
-- covered by the criterion suite (test_safetensors.c
-- load_typed_error_codes); writing NUL-bearing binary from Idris's
-- String-based writeFile isn't reliable.
--
-- The dtype-mismatch test names both `Compatible TestExecutor F32`
-- and `... F64`, so it requires a dual-dtype lane (tape, torch-cpu,
-- torch-cuda, mlx-cpu) — same constraint and precedent as
-- Test.Properties.F32GradParity; the F32-only lanes (torch-mps,
-- mlx-gpu) don't run this suite.
--
-- saveAll dumps the whole registry (other suites' params included —
-- Test.Main runs in one process); that's fine: those entries
-- round-trip onto themselves, and first-error-wins surfaces our
-- deliberately mismatched param regardless of its position.
module Test.LoadOpts

import Data.Vect
import System.File

import Ml.Checkpoint
import Ml.Executor
import Ml.Tensor
import Test.Harness

import Test.Config

packBuf : List Double -> AnyPtr -> Int -> AnyPtr
packBuf [] b _        = b
packBuf (x :: xs) b o = packBuf xs (prim__setDouble b o x) (o + 1)

mkVecParam : (n : Nat) -> String -> List Double ->
             IO (Tensor [n] TestExecutor TestDType WithGrad)
mkVecParam n name xs = do
  buf <- ioRerun (\_ => packBuf xs (prim__allocDoubles (cast n)) 0)
  tparam1d {ex=TestExecutor} {dt=TestDType} {n} name buf

fileNotFound : IO Bool
fileNotFound = do
  r <- load {ex=TestExecutor} "/tmp/idrisml-lo-missing.safetensors" defaultLoadOpts
  check ("missing file -> Left FileNotFound (got " ++ show r ++ ")")
        (r == Left FileNotFound)

malformedFile : IO Bool
malformedFile = do
  let path = "/tmp/idrisml-lo-tiny.safetensors"
  Right () <- writeFile path "AB"
    | Left err => check ("write tiny file: " ++ show err) False
  r <- load {ex=TestExecutor} path defaultLoadOpts
  check ("truncated container -> Left MalformedFile (got " ++ show r ++ ")")
        (r == Left MalformedFile)

shapeMismatch : IO Bool
shapeMismatch = do
  let path = "/tmp/idrisml-lo-shape.safetensors"
  _ <- mkVecParam 2 "lo_shape_w" [1.0, 2.0]
  Right () <- saveAll {ex=TestExecutor} path
    | Left _ => check "saveAll for shape test" False
  _ <- mkVecParam 3 "lo_shape_w" [0.0, 0.0, 0.0]
  r <- load {ex=TestExecutor} path defaultLoadOpts
  check ("numel change -> Left ShapeMismatch (got " ++ show r ++ ")")
        (r == Left ShapeMismatch)

dtypeMismatchAndCast : IO Bool
dtypeMismatchAndCast = do
  let path = "/tmp/idrisml-lo-dtype.safetensors"
  _ <- tparamScalar {ex=TestExecutor} {dt=F64} "lo_dt_w" 1.5
  Right () <- saveAll {ex=TestExecutor} path
    | Left _ => check "saveAll for dtype test" False
  p32 <- tparamScalar {ex=TestExecutor} {dt=F32} "lo_dt_w" 0.0
  rStrict <- load {ex=TestExecutor} path defaultLoadOpts
  rCast <- load {ex=TestExecutor} path ({ allowCast := True } defaultLoadOpts)
  let restored = tensorItem p32
  check ("strict -> Left DtypeMismatch (got " ++ show rStrict
         ++ "); allowCast -> Right (got " ++ show rCast
         ++ ", value " ++ show restored ++ ")")
        (rStrict == Left DtypeMismatch && rCast == Right () && restored == 1.5)

onlyPrefixFilters : IO Bool
onlyPrefixFilters = do
  let path = "/tmp/idrisml-lo-only.safetensors"
  _ <- tparamScalar {ex=TestExecutor} {dt=TestDType} "lo_only_a" 1.0
  _ <- tparamScalar {ex=TestExecutor} {dt=TestDType} "lo_only_b" 2.0
  Right () <- saveAll {ex=TestExecutor} path
    | Left _ => check "saveAll for only test" False
  pa <- tparamScalar {ex=TestExecutor} {dt=TestDType} "lo_only_a" 9.0
  pb <- tparamScalar {ex=TestExecutor} {dt=TestDType} "lo_only_b" 9.0
  r <- load {ex=TestExecutor} path ({ only := Just "lo_only_a" } defaultLoadOpts)
  let (va, vb) = (tensorItem pa, tensorItem pb)
  check ("only \"lo_only_a\": a restored, b untouched (got " ++ show r
         ++ ", a = " ++ show va ++ ", b = " ++ show vb ++ ")")
        (r == Right () && va == 1.0 && vb == 9.0)

export
tests : List (IO Bool)
tests = [ fileNotFound, malformedFile, shapeMismatch
        , dtypeMismatchAndCast, onlyPrefixFilters ]
