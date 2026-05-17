-- | Microbenchmark to isolate per-FFI-call overhead.
-- |
-- | Phase 0 attribution surfaced ~165 µs/call for `prim__mv` /
-- | `prim__linear` in NTM-copy, with the C-body itself measured at
-- | ~1 µs. This bench answers: where does the 164 µs go?
-- |
-- | Hypotheses:
-- | (a) Chez Scheme's foreign-call dispatch is genuinely ~150 µs/call.
-- |     Tight-loop `prim__linear` would still cost ~150 µs/call.
-- | (b) The cost is in the surrounding Idris machinery (record-field
-- |     access, Maybe-case extraction, AnyPtr boxing/unboxing). A tight
-- |     loop with no surrounding work would cost ≪ 165 µs/call.
-- |
-- | The diff between (a) and (b) decides whether the next perf lever
-- | requires Idris-codegen work or Idris-source-level refactoring.

module Example.ProfileMicro

import Data.Vect
import System
import System.Clock
import Compat.Random

import Device
import Layer.Core
import Layer.Linear
import Layer.Lstm
import Layer.Ntm
import Tensor
import BuildConfig

%default partial


----------------------------------------------------------------------
-- Timing
----------------------------------------------------------------------

elapsedMs : Clock Monotonic -> Clock Monotonic -> Double
elapsedMs t0 t1 =
  let s = cast {to=Double} (seconds t1 - seconds t0)
      ns = cast {to=Double} (nanoseconds t1 - nanoseconds t0)
  in s * 1000.0 + ns / 1000000.0


----------------------------------------------------------------------
-- Test Fixture
----------------------------------------------------------------------

-- Realistic NTM-FC dimensions: 26 outputs from 100 hidden.
W : Nat
W = 26

H : Nat
H = 100

-- Iteration counts.
WarmupIters : Nat
WarmupIters = 1000

BenchIters : Nat
BenchIters = 50000


----------------------------------------------------------------------
-- Benches
----------------------------------------------------------------------

-- A) Pure prim__linear loop. The `last` arg threads the previous
--    result through to prevent CSE. Each iteration does one FFI call
--    and one recursive call — minimal Idris work in between.
loopLinear : Nat -> (wp, xp, bp : AnyPtr) -> AnyPtr -> AnyPtr
loopLinear Z _ _ _ acc = acc
loopLinear (S k) wp xp bp _ =
  loopLinear k wp xp bp (prim__linear wp xp bp)

-- B) Pure prim__mv loop (no bias). For comparison with linear.
loopMv : Nat -> (wp, xp : AnyPtr) -> AnyPtr -> AnyPtr
loopMv Z _ _ acc = acc
loopMv (S k) wp xp _ =
  loopMv k wp xp (prim__mv wp xp)

-- C) Pure prim__add loop on two same-shape tensors. To establish
--    the per-call FFI floor for a 2-arg ANY simple op.
loopAdd : Nat -> (ap, bp : AnyPtr) -> AnyPtr -> AnyPtr
loopAdd Z _ _ acc = acc
loopAdd (S k) ap bp _ =
  loopAdd k ap bp (prim__add ap bp)

-- D) tlinear loop — uses the typed wrapper which boxes/unboxes
--    the result through MkTensor + .tensorPtr. Same FFI as loopLinear,
--    plus record construction/destruction per iteration.
loopTLinear : Nat -> {h : Nat} -> {o : Nat} ->
                Tensor [o, h] ExampleDevice ExampleDType WithGrad -> Tensor [h] ExampleDevice ExampleDType WithGrad -> Tensor [o] ExampleDevice ExampleDType WithGrad ->
                Tensor [o] ExampleDevice ExampleDType WithGrad -> Tensor [o] ExampleDevice ExampleDType WithGrad
loopTLinear Z _ _ _ acc = acc
loopTLinear (S k) w x b _ =
  loopTLinear k w x b (tlinear w x b)

-- E) Linear's applyVar loop — the most realistic test, since this
--    is the path Layer.Linear actually uses (and what Layer.Lstm /
--    Layer.Ntm / Layer.Dnc all hit). Includes record destructuring
--    of LinearState, .weightT/.biasT extraction, etc.
loopLinearApply : {h : Nat} -> {o : Nat} ->
                    Nat -> LinearState h o CPU -> Tensor [h] ExampleDevice ExampleDType WithGrad ->
                    Tensor [o] ExampleDevice ExampleDType WithGrad -> Tensor [o] ExampleDevice ExampleDType WithGrad
loopLinearApply Z _ _ acc = acc
loopLinearApply (S k) st x _ =
  let (_, y) = applyVar st x
  in loopLinearApply k st x y

-- F) applyLstm loop — full LSTM cell forward (2 prim__linears + 1
--    lstm_gates_pair). Closer to the NTM-realistic pattern where
--    multiple consecutive C calls are interleaved with state
--    record destructuring.
loopLstmApply : {i : Nat} -> {o : Nat} ->
                  Nat -> LstmState i o CPU -> Tensor [i] ExampleDevice ExampleDType WithGrad ->
                  LstmState i o CPU -> LstmState i o CPU
loopLstmApply Z _ _ acc = acc
loopLstmApply (S k) st x _ =
  let (st', _) = applyLstm st x
  in loopLstmApply k st' x st'

-- G) applyNtm loop — the full NTM controller forward. Per-call
--    overhead is what we're trying to attribute. Compares against
--    the per-LINEAR-call cost reported by the C profiler in real
--    NTM-copy runs.
partial
loopNtmApply : {n : Nat} -> {m : Nat} -> {h : Nat} -> {i : Nat} -> {o : Nat} ->
                 Nat -> NtmState n m h i o CPU -> TVec i ExampleDevice ExampleDType WithGrad ->
                 NtmState n m h i o CPU -> NtmState n m h i o CPU
loopNtmApply Z _ _ acc = acc
loopNtmApply (S k) st x _ =
  let (st', _) = applyNtm st x
  in loopNtmApply k st' x st'


----------------------------------------------------------------------
-- Setup helpers
----------------------------------------------------------------------

allocFilled : Nat -> Double -> AnyPtr
allocFilled n v =
  let nI = cast {to=Int} n
      buf = prim__allocDoubles nI
  in fillBuf buf 0 nI v
  where
    fillBuf : AnyPtr -> Int -> Int -> Double -> AnyPtr
    fillBuf b off n v =
      if off >= n then b
      else fillBuf (prim__setDouble b off v) (off + 1) n v


----------------------------------------------------------------------
-- Main
----------------------------------------------------------------------

showMs : Double -> String
showMs d =
  let whole = cast {to=Integer} d
      frac = cast {to=Integer} (abs ((d - cast whole) * 100))
  in show whole ++ "." ++ show frac

showUs : Double -> String
showUs d =
  let whole = cast {to=Integer} d
      frac = cast {to=Integer} (abs ((d - cast whole) * 100))
  in show whole ++ "." ++ show frac

main : IO ()
main = do
  srand 123456
  putStrLn "=== prim__linear / mv / add microbench ==="
  putStrLn $ "Dim: W=" ++ show W ++ " H=" ++ show H
  putStrLn $ "Iters: warmup=" ++ show WarmupIters ++ " bench=" ++ show BenchIters
  putStrLn ""

  -- LSTM bench setup: instantiate a real LstmState (params registered).
  lstm <- lstmLayer {i = H} {o = H} "micro_lstm"
  let xLstm : Tensor [H] ExampleDevice ExampleDType WithGrad
      xLstm = MkTensor (prim__createState1d (cast {to=Int} H) (allocFilled H 0.5)) Nothing

  -- NTM bench setup at NTM-copy default dims (N=128 M=20 H=100 i=9).
  ntm <- ntmLayer {n = 128, m = 20, h = 100, i = 9, o = 8} "micro_ntm"
  let xNtm : TVec 9 ExampleDevice ExampleDType WithGrad
      xNtm = MkTensor (prim__createState1d 9 (allocFilled 9 0.5)) Nothing

  -- Allocate grad-tracked PARAMS (registered in the param registry).
  -- prim__linear with grad-requires-grad inputs fires tape_append +
  -- allocates LinearMeta + memcpys x_vals — same code path as NTM/DNC.
  let wI = cast {to=Int} W
      hI = cast {to=Int} H
      wBuf = allocFilled (W * H) 0.01
      xBuf = allocFilled H 0.5
      bBuf = allocFilled W 0.0
      a2Buf = allocFilled W 0.7
      b2Buf = allocFilled W 0.3
      wPtr  = prim__paramRegister "micro_W" (prim__createParam2d wI hI wBuf)
      xPtr  = prim__paramRegister "micro_x" (prim__createParam1d hI xBuf)
      bPtr  = prim__paramRegister "micro_b" (prim__createParam1d wI bBuf)
      a2Ptr = prim__paramRegister "micro_a2" (prim__createParam1d wI a2Buf)
      b2Ptr = prim__paramRegister "micro_b2" (prim__createParam1d wI b2Buf)
      -- Boxed-tensor versions for tlinear / applyVar benches.
      wT : Tensor [W, H] ExampleDevice ExampleDType WithGrad
      wT = MkTensor wPtr Nothing
      xT : Tensor [H] ExampleDevice ExampleDType WithGrad
      xT = MkTensor xPtr Nothing
      bT : Tensor [W] ExampleDevice ExampleDType WithGrad
      bT = MkTensor bPtr Nothing
      lin : LinearState H W CPU
      lin = MkLinear wT bT

  -- Warmup. Use prim__item to extract a Double from the result so
  -- the compiler can't elide the FFI calls. (prim__linear returns
  -- AnyPtr, which is otherwise discarded as unused.) Using a 1d item
  -- extractor on the resulting [W] vector at index 0.
  let warmL = prim__item1d (loopLinear WarmupIters wPtr xPtr bPtr bPtr) 0
  putStrLn $ "warmup linear last[0] = " ++ show warmL
  let warmM = prim__item1d (loopMv WarmupIters wPtr xPtr xPtr) 0
  putStrLn $ "warmup mv last[0]     = " ++ show warmM
  let warmA = prim__item1d (loopAdd WarmupIters a2Ptr b2Ptr a2Ptr) 0
  putStrLn $ "warmup add last[0]    = " ++ show warmA

  -- Bench: prim__linear
  t0 <- clockTime Monotonic
  let lr = prim__item1d (loopLinear BenchIters wPtr xPtr bPtr bPtr) 0
  t1 <- clockTime Monotonic
  let linearMs = elapsedMs t0 t1
      linearUs = linearMs * 1000.0 / cast (the Nat BenchIters)

  -- Bench: prim__mv
  t2 <- clockTime Monotonic
  let mr = prim__item1d (loopMv BenchIters wPtr xPtr xPtr) 0
  t3 <- clockTime Monotonic
  let mvMs = elapsedMs t2 t3
      mvUs = mvMs * 1000.0 / cast (the Nat BenchIters)

  -- Bench: prim__add (on same-shape vectors of width W)
  t4 <- clockTime Monotonic
  let ar = prim__item1d (loopAdd BenchIters a2Ptr b2Ptr a2Ptr) 0
  t5 <- clockTime Monotonic
  let addMs = elapsedMs t4 t5
      addUs = addMs * 1000.0 / cast (the Nat BenchIters)

  -- Bench: tlinear (typed wrapper)
  t6 <- clockTime Monotonic
  let tr = prim__item1d (loopTLinear BenchIters wT xT bT bT).tensorPtr 0
  t7 <- clockTime Monotonic
  let tlinMs = elapsedMs t6 t7
      tlinUs = tlinMs * 1000.0 / cast (the Nat BenchIters)

  -- Bench: Linear's applyVar (the production path)
  t8 <- clockTime Monotonic
  let apr = prim__item1d (loopLinearApply BenchIters lin xT bT).tensorPtr 0
  t9 <- clockTime Monotonic
  let applyMs = elapsedMs t8 t9
      applyUs = applyMs * 1000.0 / cast (the Nat BenchIters)

  -- Bench: applyLstm (2 prim__linear + 1 lstm_gates_pair per call)
  t10 <- clockTime Monotonic
  let lstmFinal = loopLstmApply (BenchIters `div` 5) lstm xLstm lstm
  -- Force evaluation by extracting from final state's hidden tensor
  let lstmR = case lstmFinal.hiddenT of
                Just h => prim__item1d h.tensorPtr 0
                Nothing => 0.0
  t11 <- clockTime Monotonic
  let lstmIters : Nat
      lstmIters = BenchIters `div` 5
      lstmMs = elapsedMs t10 t11
      lstmUs = lstmMs * 1000.0 / cast lstmIters

  -- Bench: applyNtm (full NTM controller forward, ~30 prims/call)
  t12 <- clockTime Monotonic
  let ntmFinal = loopNtmApply (BenchIters `div` 50) ntm xNtm ntm
  -- Force eval by reading from a state field. Match constructor positionally.
  -- NtmState fields: lstm, readFc, writeFc, outputFc, memInitT, initReadOutT,
  -- memT, raT, waT, roT (10 total).
  let ntmR = case ntmFinal of
               MkNtm _ _ _ _ _ _ _ _ _ (Just rOut) => prim__item1d rOut.tensorPtr 0
               _ => 0.0
  t13 <- clockTime Monotonic
  let ntmIters : Nat
      ntmIters = BenchIters `div` 50
      ntmMs = elapsedMs t12 t13
      ntmUs = ntmMs * 1000.0 / cast ntmIters

  putStrLn $ "linear last[0] = " ++ show lr
  putStrLn $ "mv     last[0] = " ++ show mr
  putStrLn $ "add    last[0] = " ++ show ar
  putStrLn $ "tlin   last[0] = " ++ show tr
  putStrLn $ "apply  last[0] = " ++ show apr
  putStrLn $ "lstm   last[0] = " ++ show lstmR

  putStrLn $ "prim__linear: total " ++ showMs linearMs ++ " ms over "
           ++ show BenchIters ++ " iters -> " ++ showUs linearUs ++ " us/call"
  putStrLn $ "prim__mv:     total " ++ showMs mvMs ++ " ms over "
           ++ show BenchIters ++ " iters -> " ++ showUs mvUs ++ " us/call"
  putStrLn $ "prim__add:    total " ++ showMs addMs ++ " ms over "
           ++ show BenchIters ++ " iters -> " ++ showUs addUs ++ " us/call"
  putStrLn $ "tlinear:      total " ++ showMs tlinMs ++ " ms over "
           ++ show BenchIters ++ " iters -> " ++ showUs tlinUs ++ " us/call"
  putStrLn $ "Linear apply: total " ++ showMs applyMs ++ " ms over "
           ++ show BenchIters ++ " iters -> " ++ showUs applyUs ++ " us/call"
  putStrLn $ "applyLstm:    total " ++ showMs lstmMs ++ " ms over "
           ++ show lstmIters ++ " iters -> " ++ showUs lstmUs
           ++ " us/call (= 3 prims/call internally)"
  putStrLn $ "applyNtm:     total " ++ showMs ntmMs ++ " ms over "
           ++ show ntmIters ++ " iters -> " ++ showUs ntmUs
           ++ " us/call"
  putStrLn ""
  putStrLn "Reference: NTM-copy reports ~165 us/call for LINEAR/MV in"
  putStrLn "tape's per-op forward profiler. Discrepancy = surrounding-"
  putStrLn "code overhead (record/Maybe/etc) that this tight loop avoids."
