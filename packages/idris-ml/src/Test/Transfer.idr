||| Tests for `toExecutor` — both intra-backend fast paths (matching
||| `backendTag` → in-place primIntraMigrate) and cross-backend
||| round-trips (differing tags → host buffer hop).
|||
||| Requires a multi-backend test build:
|||
|||     make BACKEND=torch,tape,mlx test-multi
|||
||| `make test` (single-backend) does NOT include this module —
||| Test.Transfer references tape / torch / mlx C symbols
||| explicitly via the per-backend `UserExecutorTransfer` instances.
||| Single-backend builds link only one of those sets and would
||| crash at FFI resolution. `MainMulti.idr` is the entry point that
||| wires this module in.
module Test.Transfer

import Data.List
import Data.Vect

import Executor
import Tensor
import Test.Harness

----------------------------------------------------------------------
-- Helpers
----------------------------------------------------------------------

||| Create a 4-element Tensor on the destination backend using its
||| `UserExecutorTransfer.primCreateFromHost`. Mirrors the helper in
||| `Example.Transfer` so the test exercises the same code path.
|||
||| Every side-effecting step (alloc, write, create, free) goes
||| through `primIO` so Idris-2's Chez codegen sequences them with
||| `%World` instead of let-laziness. The naive let-chain version
||| got hoisted to a module-level CSE'd constant whose lambda body
||| references buffers allocated at module load — every subsequent
||| call to the constant re-ran the frees on the same pointers and
||| tripped libsystem_malloc's "pointer being freed was not
||| allocated" abort.
makeVec4 : {0 ex : Type} -> {0 dt : DType} ->
           UserExecutorTransfer ex => RuntimeDType dt => Compatible ex dt =>
           (Double, Double, Double, Double) ->
           IO (Tensor [4] ex dt WithGrad)
makeVec4 (a, b, c, dd) = do
  buf  <- primIO (\w => MkIORes (primAllocHost   {ex} 4)        w)
  buf1 <- primIO (\w => MkIORes (prim__setDouble buf  0 a)     w)
  buf2 <- primIO (\w => MkIORes (prim__setDouble buf1 1 b)     w)
  buf3 <- primIO (\w => MkIORes (prim__setDouble buf2 2 c)     w)
  buf4 <- primIO (\w => MkIORes (prim__setDouble buf3 3 dd)    w)
  sh   <- primIO (\w => MkIORes (primAllocIntHost {ex} 1)       w)
  sh1  <- primIO (\w => MkIORes (primSetIntHost   {ex} sh 0 4)  w)
  ptr  <- primIO (\w =>
            MkIORes (primCreateFromHost {ex} buf4 sh1 1 1 (dtypeTag {t=dt})) w)
  _ <- primIO (\w => MkIORes (primFreeIntHost {ex} sh1)  w)
  _ <- primIO (\w => MkIORes (primFreeHost    {ex} buf4) w)
  pure (MkTensor ptr Nothing)

read4 : {0 ex : Type} -> {0 dt : DType} -> UserExecutorCore ex =>
        Tensor [4] ex dt WithGrad ->
        (Double, Double, Double, Double)
read4 t =
  ( primItem1d {ex} t.tensorPtr 0
  , primItem1d {ex} t.tensorPtr 1
  , primItem1d {ex} t.tensorPtr 2
  , primItem1d {ex} t.tensorPtr 3 )

expected : (Double, Double, Double, Double)
expected = (1.0, 2.0, 3.0, 4.0)

||| Value-preservation gate. Inputs (1.0, 2.0, 3.0, 4.0) are exactly
||| representable in F32 so the F32 hop should also exact-match.
matchesExpected : (Double, Double, Double, Double) -> Bool
matchesExpected (a, b, c, d) =
  let (ea, eb, ec, ed) = Transfer.expected
      delta = abs (a - ea) + abs (b - eb) + abs (c - ec) + abs (d - ed)
  in delta < 0.000001

----------------------------------------------------------------------
-- Intra-backend smoke (matching backendTag → primIntraMigrate)
----------------------------------------------------------------------

||| Trivially exercises the same-backendTag fast path on Tape.
||| Always works regardless of which other backends are linked
||| (uses only `tensor_to_device_tape`, which is always present in
||| any build that includes tape).
intraTapeSmoke : IO Bool
intraTapeSmoke = do
  src <- makeVec4 {ex=TapeExecutor} {dt = F64} expected
  dst <- toExecutor TapeExecutor src
  check "intra-backend TapeExecutor→TapeExecutor preserves value"
        (matchesExpected (read4 dst))

||| Intra-Torch fast path: TorchExecutor TCpu → TorchExecutor TMps. Exercises
||| libtorch's `.to("mps")` in-place migration via primIntraMigrate.
||| Requires the F32 source — libtorch's MPS rejects F64
||| construction (the `Compatible (TorchExecutor TMps) F64` non-instance
||| pre-empts the type, but we still need to land on F32 at runtime).
intraTorchHwSmoke : IO Bool
intraTorchHwSmoke = do
  -- Build F64 then narrow to F32 via tcastUnsafe — keeps the cast
  -- path exercised alongside the direct F32 create the
  -- f32StorageProbe / roundtripF32Smoke tests below use.
  src64 <- makeVec4 {ex=TorchExecutor TCpu} {dt = F64} expected
  src   <- tcastUnsafe F32 src64
  dst   <- toExecutor (TorchExecutor TMps) src
  check "intra-torch TorchExecutor TCpu→TMps preserves value"
        (matchesExpected (read4 dst))

-- (intra-mlx fast path is exercised by `roundtripF32Smoke` below,
-- whose MGpu→MCpu leg goes through the intra-mlx primIntraMigrate;
-- a separate direct smoke would duplicate that coverage.)

----------------------------------------------------------------------
-- Cross-backend smoke (differing backendTag → host round-trip)
----------------------------------------------------------------------

||| TapeExecutor → TorchExecutor TCpu. The simplest cross-backend hop. F64
||| throughout; both ends admit F64.
crossTapeToTorchSmoke : IO Bool
crossTapeToTorchSmoke = do
  src <- makeVec4 {ex=TapeExecutor} {dt = F64} expected
  dst <- toExecutor (TorchExecutor TCpu) src
  check "cross-backend TapeExecutor→TorchExecutor TCpu preserves value"
        (matchesExpected (read4 dst))

||| TorchExecutor TCpu → MlxExecutor MCpu. F64 round-trip through host buffer.
crossTorchToMlxSmoke : IO Bool
crossTorchToMlxSmoke = do
  src <- makeVec4 {ex=TorchExecutor TCpu} {dt = F64} expected
  dst <- toExecutor (MlxExecutor MCpu) src
  check "cross-backend TorchExecutor TCpu→MlxExecutor MCpu preserves value"
        (matchesExpected (read4 dst))

||| MlxExecutor MCpu → TapeExecutor. Closes the F64 round-trip from
||| crossTorchToMlxSmoke's perspective.
crossMlxToTapeSmoke : IO Bool
crossMlxToTapeSmoke = do
  src <- makeVec4 {ex=MlxExecutor MCpu} {dt = F64} expected
  dst <- toExecutor TapeExecutor src
  check "cross-backend MlxExecutor MCpu→TapeExecutor preserves value"
        (matchesExpected (read4 dst))

||| 3-step F64 hop: TapeExecutor → TorchExecutor TCpu → MlxExecutor MCpu → TapeExecutor.
||| End-to-end value preservation across two cross-backend host
||| round-trips.
roundtripF64Smoke : IO Bool
roundtripF64Smoke = do
  v0 <- makeVec4 {ex=TapeExecutor} {dt = F64} expected
  v1 <- toExecutor (TorchExecutor TCpu) v0
  v2 <- toExecutor (MlxExecutor MCpu) v1
  v3 <- toExecutor TapeExecutor v2
  check "F64 roundtrip TapeExecutor→Torch→Mlx→TapeExecutor preserves value"
        (matchesExpected (read4 v3))

----------------------------------------------------------------------
-- F32 storage probes (dtag-threaded primCreateFromHost)
----------------------------------------------------------------------

||| 0.1 is not representable in F32; the nearest F32 value read back
||| as a double is 0.100000001490116…. A create-from-host that honours
||| `dt = F32` must round; one that silently builds F64 storage (the
||| pre-dtag bug this guards against) preserves 0.1 exactly.
f32Rounded : Double
f32Rounded = 0.10000000149011612

||| True iff `a` is 0.1-rounded-to-F32 (storage really is F32).
roundsToF32 : Double -> Bool
roundsToF32 a = abs (a - f32Rounded) < 0.000000000001

f32StorageProbe : {0 ex : Type} ->
                  UserExecutorTransfer ex => Compatible ex F32 =>
                  String -> IO Bool
f32StorageProbe label = do
  t <- makeVec4 {ex} {dt = F32} (0.1, 2.0, 3.0, 4.0)
  let (a, _, _, _) = read4 t
  check ("F32 create-from-host on " ++ label ++ " stores real F32 storage")
        (roundsToF32 a)

||| 4-step F32 hop: TorchExecutor TCpu → TMps → MlxExecutor MGpu →
||| MlxExecutor MCpu → TorchExecutor TCpu. Mirrors Example.Transfer's
||| Part 3. Inputs are exactly representable in F32, so the hop must
||| exact-match end-to-end. Before the dtag threading this crashed the
||| unit-test harness at process-exit GC: the mlx leg's "F32" tensor
||| was really F64 storage and the tensor-guardian's free tripped an
||| MPS dtype-validation assertion.
roundtripF32Smoke : IO Bool
roundtripF32Smoke = do
  v0 <- makeVec4 {ex=TorchExecutor TCpu} {dt = F32} expected
  v1 <- toExecutor (TorchExecutor TMps) v0
  v2 <- toExecutor (MlxExecutor MGpu) v1
  v3 <- toExecutor (MlxExecutor MCpu) v2
  v4 <- toExecutor (TorchExecutor TCpu) v3
  check "F32 roundtrip Torch TCpu→TMps→MlxGpu→MlxCpu→TCpu preserves value"
        (matchesExpected (read4 v4))

----------------------------------------------------------------------
-- Public test list
----------------------------------------------------------------------

export
tests : List (IO Bool)
tests =
  [ intraTapeSmoke
  , intraTorchHwSmoke
  , crossTapeToTorchSmoke
  , crossTorchToMlxSmoke
  , crossMlxToTapeSmoke
  , roundtripF64Smoke
  , f32StorageProbe {ex=TapeExecutor} "TapeExecutor"
  , f32StorageProbe {ex=TorchExecutor TCpu} "TorchExecutor TCpu"
  , f32StorageProbe {ex=MlxExecutor MCpu} "MlxExecutor MCpu"
  , roundtripF32Smoke
  ]
