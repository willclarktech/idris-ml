-- | RankBroadcastBench: Idris-level rank-3 broadcast `primMul` microbench
-- |
-- | Counterpart to:
-- |   - packages/backends/bench_rank3_broadcast.cpp        (libtorch direct)
-- |   - packages/backends/bench_rank3_broadcast_wrapped.cpp (C wrapper)
-- |   - packages/idris-transformers/scripts/time_rank3_broadcast.py (Python)
-- |
-- | Same shape (`[6, 32, 32] * [6, 1, 32]`), same iteration counts.
-- | Calls `primMul` (the typeclass-dispatched primitive that
-- | `applyRopeAllHeads` in `Layer/RoPE.idr` uses inside its
-- | `ioRerun \_ => let ...` block — i.e., the exact path HfLlama
-- | exercises for the broadcast muls in RoPE).
-- |
-- | The wall delta vs the C-wrapper bench is the Scheme wrap layer
-- | (generated `prim__mulXxx` wrapper: foreign-procedure cache lookup,
-- | tensor-handle-v2 vector unwrap/wrap, guardian register, no-op
-- | retain FFI). On torch/mlx/tape the Scheme wrap template is
-- | identical (only the backend tag string differs), so any wrap-layer
-- | overhead found here applies to all three backends.
-- |
-- | Run:
-- |   make example-rank-broadcast-bench
-- |   BACKEND=torch TORCH_DEVICE=mps make example-rank-broadcast-bench
-- |   BACKEND=mlx MLX_DEVICE=gpu make example-rank-broadcast-bench

module Example.RankBroadcastBench

import Data.String
import System
import System.Clock

import Tensor
import Executor
import BuildConfig

-- Shapes mirror Llama-3.2-1B Q projection's RoPE input.
seqLen : Nat
seqLen = 6

numHeads : Nat
numHeads = 32

halfDim : Nat
halfDim = 32

defaultIters : Nat
defaultIters = 100

defaultWarmup : Nat
defaultWarmup = 10

record Config where
  constructor MkConfig
  iters  : Nat
  warmup : Nat

defaultConfig : Config
defaultConfig = MkConfig defaultIters defaultWarmup

parseArgs : Config -> List String -> Config
parseArgs cfg [] = cfg
parseArgs cfg ("--iters" :: x :: rest) =
  parseArgs ({iters := the Nat (cast (the Int (cast x)))} cfg) rest
parseArgs cfg ("--warmup" :: x :: rest) =
  parseArgs ({warmup := the Nat (cast (the Int (cast x)))} cfg) rest
parseArgs cfg (_ :: rest) = parseArgs cfg rest

-- Build a rank-3 fp32 tensor with one nonzero element. Uses
-- `dtCreate` (rank-generic) since `dtCreateState*` only ships 1d/2d
-- variants. Returns a non-grad-tracked tensor handle.
buildTensor3d : (d0, d1, d2 : Nat) -> AnyPtr
buildTensor3d d0 d1 d2 =
  let d0I = the Int (cast d0)
      d1I = the Int (cast d1)
      d2I = the Int (cast d2)
      numEls = d0I * d1I * d2I
      buf = prim__allocDoubles numEls
      buf' = prim__setDouble buf 0 0.5
      sh = prim__allocInts 3
      sh' = prim__setInt sh 0 d0I
      sh'' = prim__setInt sh' 1 d1I
      sh''' = prim__setInt sh'' 2 d2I
  in dtCreate {ex=ExampleExecutor} {t=ExampleDType} buf' sh''' 3 0
       (deviceStreamTag {ex=ExampleExecutor})

-- Force backend-side eval. mlx is lazy by default; torch-mps queues
-- kernels but doesn't sync without a CPU-side read; tape is eager but
-- the call costs nothing if data's already realised.
forceEval : AnyPtr -> IO ()
forceEval h = do
  let v = primItem {ex=ExampleExecutor} (primSum {ex=ExampleExecutor} h)
  ignore (pure v)

-- Tail-recursive chained-mul loop. Threading the previous result
-- through as the next iteration's left operand prevents Idris-2's
-- code generator from dead-code-eliminating the `primMul` call (the
-- return type is `AnyPtr` and primMul is pure, so an unused let
-- binding would get discarded). Shape stays `[seq, numHeads, halfDim]`
-- across iterations because broadcast preserves outer dims.
loopMul : Nat -> AnyPtr -> AnyPtr -> IO AnyPtr
loopMul Z accum _ = pure accum
loopMul (S k) accum cos = loopMul k (primMul {ex=ExampleExecutor} accum cos) cos

-- Microseconds between two monotonic clock readings.
diffUs : Clock Monotonic -> Clock Monotonic -> Double
diffUs t1 t0 =
  let totalNs = (seconds t1 - seconds t0) * 1000000000 + (nanoseconds t1 - nanoseconds t0)
  in cast totalNs / 1000.0

-- Two-decimal fixed-point formatting (copied from MatmulBench).
fmt2 : Double -> String
fmt2 x =
  let scaled = the Integer (cast (x * 100.0))
      whole  = scaled `div` 100
      frac   = abs (scaled `mod` 100)
      fracS  = if frac < 10 then "0" ++ show frac else show frac
  in show whole ++ "." ++ fracS

main : IO ()
main = do
  requireMachine {m = ChosenMachine}
  args <- getArgs
  let cfg = parseArgs defaultConfig (drop 1 args)

  putStrLn "=== RankBroadcastBench: Idris-level rank-3 primMul ==="
  putStrLn $ "shape: x=[" ++ show seqLen ++ "," ++ show numHeads ++ "," ++ show halfDim
           ++ "] cos=[" ++ show seqLen ++ ",1," ++ show halfDim ++ "]"
  putStrLn $ "warmup=" ++ show cfg.warmup ++ " measure=" ++ show cfg.iters

  let x = buildTensor3d seqLen numHeads halfDim
  let cosT = buildTensor3d seqLen 1 halfDim

  -- Warmup. Sync at the end to flush any one-time backend init
  -- (Metal pipeline cache, MPSGraph compile-cache for this op shape).
  warmed <- loopMul cfg.warmup x cosT
  forceEval warmed

  t0 <- clockTime Monotonic
  result <- loopMul cfg.iters x cosT
  -- Final sync so the timing window covers actually-completed work,
  -- matching the bench_rank3_broadcast{,_wrapped} C harnesses.
  forceEval result
  t1 <- clockTime Monotonic

  let dtUs   = diffUs t1 t0
      perOp  = dtUs / cast cfg.iters
  putStrLn $ "Total wall: " ++ fmt2 dtUs ++ " us"
  putStrLn $ "Per op:     " ++ fmt2 perOp ++ " us  (= " ++ fmt2 (perOp / 1000.0) ++ " ms)"
  putStrLn ""
  putStrLn $ "RESULT\titers=" ++ show cfg.iters
           ++ "\ttotal_us=" ++ fmt2 dtUs
           ++ "\tper_op_us=" ++ fmt2 perOp
