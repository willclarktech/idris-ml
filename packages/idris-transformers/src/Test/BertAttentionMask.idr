||| Unit tests for the RT1 attention-mask threading added to
||| `hfBertForward` (and downstream `applyEncoder` / `applyLayer` /
||| `applySelfAttn` / `oneHeadCtx` / `buildHeads`).
|||
||| Resolves `{ex=TestExecutor}` / `{dt=TestDType}` from the
||| Makefile-generated `Test.Config`, so the suite runs on whichever
||| F64-admissible primary the build targets (tape / torch-cpu /
||| mlx-cpu). MaskedFill is the load-bearing primitive being exercised.
module Test.BertAttentionMask

import Data.List
import Data.Vect

import Transformers.Bert
import Test.Harness

import Executor
import Executor.Core
import Test.Config
import Tensor
import Array

-- Build a Tensor [n] from a Vect of doubles (mirrors Test.Bert).
-- `ioRerun` defers the C-side allocation per the pure-typed-FFI
-- reorder gotcha (feedback_pure_typed_ffi_reorders.md).
mkIdsTensor : {n : Nat} -> Vect n Double -> IO (Tensor [n] TestExecutor TestDType WithGrad)
mkIdsTensor xs = do
  raw <- ioRerun (\_ => bulkToTensor {ex=TestExecutor} {dt=TestDType}
                                     (VArray (map SArray xs)))
  pure (tinput1d {n} raw)

-- Build a Tensor [m, n] from a flat Vect of doubles via tparam2d.
-- The mask is treated as a constant (no-grad in practice — the bool
-- cast inside masked_fill drops grad).
mkMask2d : {m, n : Nat} -> Vect (m * n) Double
        -> IO (Tensor [m, n] TestExecutor TestDType WithGrad)
mkMask2d {m} {n} xs = do
  let mn = cast {to=Int} (m * n)
      buf  = prim__allocDoubles mn
      buf' = fill buf 0 xs
  tparam2d {ex=TestExecutor} {dt=TestDType} {o=m} {i=n}
           ("attnmask_test_" ++ show m ++ "x" ++ show n)
           buf'
  where
    fill : AnyPtr -> Int -> Vect k Double -> AnyPtr
    fill b _ []            = b
    fill b off (v :: rest) =
      let b' = prim__setDouble b off v
      in fill b' (off + 1) rest

-- Read [N] tensor values via primItem1d.
readOut : {n : Nat} -> AnyPtr -> IO (List Double)
readOut {n} p = loop (cast {to=Int} n) 0 []
  where
    loop : Int -> Int -> List Double -> IO (List Double)
    loop end i acc =
      if i >= end
        then pure (reverse acc)
        else let v = primItem1d {ex=TestExecutor} p i
             in loop end (i + 1) (v :: acc)

maxAbsDiff : List Double -> List Double -> Double
maxAbsDiff actual expected = go actual expected 0.0
  where
    go : List Double -> List Double -> Double -> Double
    go []        _         m = m
    go _         []        m = m
    go (a :: as) (b :: bs) m =
      let d = abs (a - b)
      in go as bs (if d > m then d else m)

-- Forward through a tiny BERT (caller-built model). Returns the
-- [Hidden]-shape pooled output as a List Double. Both assertions
-- share a single model so the weights are identical across
-- the Nothing/Just paths (the C-side param registry replaces
-- entries by name on every `hfBertModel` call, so re-constructing
-- the model would yield different random weights).
runForwardWith :
     Transformers.Bert.BertModelState 4 8 1 16 4 2 TestExecutor TestDType WithGrad
  -> Maybe (Tensor [3, 3] TestExecutor TestDType WithGrad)
  -> IO (List Double)
runForwardWith model mask = do
  inputIds <- mkIdsTensor (the (Vect 3 Double) [1.0, 2.0, 3.0])
  posIds   <- mkIdsTensor (the (Vect 3 Double) [0.0, 1.0, 2.0])
  typeIds  <- mkIdsTensor (the (Vect 3 Double) [0.0, 0.0, 0.0])
  out <- hfBertForward {ex=TestExecutor} {dt=TestDType}
                       {seqLen       = 3}
                       {vocab        = 4}
                       {hidden       = 8}
                       {numLayers    = 1}
                       {numHeads     = 2}
                       {headDim      = 4}
                       {intermediate = 16}
                       {maxPos       = 4}
                       {typeVocab    = 2}
                       model inputIds posIds typeIds mask
  readOut {n=8} out.tensorPtr

-- Build the shared tiny BERT model. One call per assertion (each
-- builds a fresh paramPrefix, so the two assertions don't collide
-- on the param registry).
buildModel : (pfx : String)
          -> IO (Transformers.Bert.BertModelState 4 8 1 16 4 2 TestExecutor TestDType WithGrad)
buildModel pfx =
  hfBertModel {ex=TestExecutor} {dt=TestDType}
              {vocab        = 4}
              {hidden       = 8}
              {numLayers    = 1}
              {numHeads     = 2}
              {intermediate = 16}
              {maxPos       = 4}
              {typeVocab    = 2}
              pfx

----------------------------------------------------------------------
-- Assertion 1: zero-mask is bit-identical to Nothing
----------------------------------------------------------------------

-- Convention: mask entries `>= 0.5` mean "mask out". A mask of all
-- zeros means "mask nothing" → masked_fill is effectively a no-op
-- (the bool cast in `tensor_masked_fill` reads everything as false,
-- no entries get filled). The forward should produce bit-identical
-- output to the `Nothing` path. Pre-RT1 there was no mask path; this
-- is the regression gate that `Nothing`'s code path stays unchanged.
testZeroMaskMatchesNothing : IO Bool
testZeroMaskMatchesNothing = do
  model <- buildModel "atm_z"
  zeros <- mkMask2d {m=3} {n=3} (the (Vect 9 Double)
             [0.0, 0.0, 0.0,
              0.0, 0.0, 0.0,
              0.0, 0.0, 0.0])
  outNothing <- runForwardWith model Nothing
  outZero    <- runForwardWith model (Just zeros)
  let d = maxAbsDiff outZero outNothing
  if d == 0.0
    then check ("zero-mask forward bit-matches Nothing (max-abs-diff "
                ++ show d ++ ")") True
    else do
      putStrLn ("  FAIL: zero-mask forward differs from Nothing by "
                ++ show d)
      putStrLn ("    nothing: " ++ show (take 3 outNothing) ++ "...")
      putStrLn ("    zero:    " ++ show (take 3 outZero)    ++ "...")
      pure False

----------------------------------------------------------------------
-- Assertion 2: ones-mask measurably changes the forward output
----------------------------------------------------------------------

-- An all-ones mask forces `-1.0e20` into every entry of every layer's
-- attention scores pre-softmax. After softmax, all attention weights
-- become uniform (every entry is `e^(-1e20)` → softmax normalises to
-- 1/seqLen). The forward path then differs from the un-masked path,
-- but on a 1-layer / hidden=8 architecture the effect is heavily
-- diluted by the residual + LN passes around attention (residual adds
-- the original token states back; LN re-normalises). Observed
-- max-abs-diff ~3e-8 on this tiny config is the mask's actual reach
-- into the pooled output — way above the 0.0 we'd see if the mask
-- weren't plumbed at all.
testOnesMaskDiffersFromNothing : IO Bool
testOnesMaskDiffersFromNothing = do
  model <- buildModel "atm_o"
  ones <- mkMask2d {m=3} {n=3} (the (Vect 9 Double)
            [1.0, 1.0, 1.0,
             1.0, 1.0, 1.0,
             1.0, 1.0, 1.0])
  outNothing <- runForwardWith model Nothing
  outOnes    <- runForwardWith model (Just ones)
  let d = maxAbsDiff outOnes outNothing
  if d > 1.0e-10
    then check ("ones-mask forward differs from Nothing (max-abs-diff "
                ++ show d ++ ")") True
    else do
      putStrLn ("  FAIL: ones-mask forward did not differ measurably "
                ++ "from Nothing — max-abs-diff = " ++ show d)
      putStrLn ("    nothing: " ++ show (take 3 outNothing) ++ "...")
      putStrLn ("    ones:    " ++ show (take 3 outOnes)    ++ "...")
      pure False

export
suite : List (String, List (IO Bool))
suite =
  [ ("Transformers.Bert attention-mask threading",
     [ testZeroMaskMatchesNothing
     , testOnesMaskDiffersFromNothing
     ])
  ]
