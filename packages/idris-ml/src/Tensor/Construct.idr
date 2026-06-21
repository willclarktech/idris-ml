||| Typed construction facade: `tensor` / `param` x InitSpec, the
||| `tparam*` family, buffers, cross-dtype `tcast`, and input wrappers.
module Tensor.Construct

import Data.Vect

import Array
import Compat.Random
import DType.Core
import Executor
import GradMode
import Init
import Sampler
import Tensor.Core
import Tensor.Internal

----------------------------------------------------------------------
-- Cross-dtype conversion: lossless via `UpcastableTo`, lossy via
-- explicit `tcastUnsafe`.
----------------------------------------------------------------------

||| Lossless precision upcast within a single dtype family
||| (`F32 → F64`, `Int 16 → Int 32`, `BFloat 16 → BFloat 32`, …).
||| The `UpcastableTo from to` constraint is solved by Idris's
||| auto-search via per-family `LTE m n` instances in `DType.Core`;
||| narrowing casts (`F64 → F32`) and cross-family casts
||| (`UInt 8 → F16`) have no `UpcastableTo` instance and use
||| `tcastUnsafe` (below) instead.
|||
||| Runtime: dispatches through `RuntimeDType to`'s `dtCastFrom`
||| method to the per-dtype `tensor_cast_dtype_<to>` C primitive.
||| Source dtype is read from the handle on the C side; the cast
||| op becomes a node in the autograd graph on backends that trace
||| it (mlx/torch).
export
tcast : {0 ex : Executor} -> Backend ex to =>
        (UpcastableTo from to, IsDType from, IsDType to) =>
        Tensor dims ex from g -> IO (Tensor dims ex to g)
tcast v = ioRerun (\_ => MkTensor (dtCastFrom {ex} {t=to} v.tensorPtr (deviceStreamTag {ex})) Nothing)

||| Explicit precision/dtype cast in ANY direction, including
||| narrowing (`F64 → F32`) and cross-family (`UInt 8 → F16`).
||| The caller takes responsibility for any precision loss or
||| representation change — calling `tcastUnsafe` is the explicit
||| signal that the conversion was intentional (mirrors the
||| `unsafePerformIO` / `believe_me` convention for primitives
||| where the caller takes responsibility).
|||
||| For lossless conversions, prefer `tcast` so the compiler
||| verifies via `UpcastableTo` that no information is lost. Use
||| `tcastUnsafe` only when the conversion is deliberately
||| narrowing or cross-family.
|||
||| Runtime path is the same as `tcast` (both dispatch through
||| `dtCastFrom`); the difference is purely the type-system gate.
export
tcastUnsafe : {0 ex : Executor} -> (0 to : DType) -> Backend ex to =>
              (IsDType from, IsDType to) =>
              Tensor dims ex from g -> IO (Tensor dims ex to g)
tcastUnsafe to v = ioRerun (\_ => MkTensor (dtCastFrom {ex} {t=to} v.tensorPtr (deviceStreamTag {ex})) Nothing)

||| Create a registered learnable [o, i] parameter from a flat (row-major)
||| double buffer. Mirrors Linear.nameLayer's tensor path.
export
tparam2d : {0 ex : Executor} -> Backend ex dt => {o, i : Nat} -> (paramId : String) -> AnyPtr -> IO (Tensor [o, i] ex dt WithGrad)
tparam2d {o} {i} pid buf = ioRerun (\_ =>
  let oI = cast {to=Int} o
      iI  = cast {to=Int} i
      reg = primParamRegister {ex} pid (dtCreateParam2d {ex} {t=dt} oI iI buf (deviceStreamTag {ex}))
  in MkTensor reg (Just pid))

||| Create a registered learnable [n] parameter from a double buffer.
export
tparam1d : {0 ex : Executor} -> Backend ex dt => {n : Nat} -> (paramId : String) -> AnyPtr -> IO (Tensor [n] ex dt WithGrad)
tparam1d {n} pid buf = ioRerun (\_ =>
  let nI = cast {to=Int} n
      reg = primParamRegister {ex} pid (dtCreateParam1d {ex} {t=dt} nI buf (deviceStreamTag {ex}))
  in MkTensor reg (Just pid))

-- ---------------------------------------------------------------
-- Fused param create + in-place init
-- ---------------------------------------------------------------
-- Replaces the `traverse normalSample + packDoubles + tparam*` chain
-- in HF model + core Nn smart constructors. The init runs in the
-- C backend (libtorch's `torch::nn::init::normal_` or `t.fill_`); no
-- per-element host-side loop, no per-element FFI marshalling. See
-- `docs/develop/perf-changes.md` for the head-to-head measurements
-- against PyTorch's `from_pretrained`.
--
-- Each variant registers the resulting tensor in the C-side optimizer
-- registry under `paramId` (same wiring as `tparam<rank>`), so
-- checkpointing + optimizer enumeration just work.

||| Registered learnable [o, i] parameter initialised from a normal
||| distribution `N(mean, std)`. Backend RNG is seeded once via
||| `tsetInitSeed` — runs are otherwise deterministic per (seed, dtype).
export
tparam2dNormal : {0 ex : Executor} -> Backend ex dt
              => {o, i : Nat} -> (paramId : String) -> (mean : Double) -> (std : Double)
              -> IO (Tensor [o, i] ex dt WithGrad)
tparam2dNormal {o} {i} pid mean std = ioRerun (\_ =>
  let oI = cast {to=Int} o
      iI  = cast {to=Int} i
      reg = primParamRegister {ex} pid (dtCreateParam2dNormal {ex} {t=dt} oI iI mean std (deviceStreamTag {ex}))
  in MkTensor reg (Just pid))

||| Registered learnable [n] parameter initialised from `N(mean, std)`.
export
tparam1dNormal : {0 ex : Executor} -> Backend ex dt
              => {n : Nat} -> (paramId : String) -> (mean : Double) -> (std : Double)
              -> IO (Tensor [n] ex dt WithGrad)
tparam1dNormal {n} pid mean std = ioRerun (\_ =>
  let nI = cast {to=Int} n
      reg = primParamRegister {ex} pid (dtCreateParam1dNormal {ex} {t=dt} nI mean std (deviceStreamTag {ex}))
  in MkTensor reg (Just pid))

||| Registered learnable [d0, d1, d2] parameter initialised from `N(mean, std)`.
export
tparam3dNormal : {0 ex : Executor} -> Backend ex dt
              => {a, b, c : Nat} -> (paramId : String) -> (mean : Double) -> (std : Double)
              -> IO (Tensor [a, b, c] ex dt WithGrad)
tparam3dNormal {a} {b} {c} pid mean std = ioRerun (\_ =>
  let aI = cast {to=Int} a
      bI  = cast {to=Int} b
      cI  = cast {to=Int} c
      reg = primParamRegister {ex} pid (dtCreateParam3dNormal {ex} {t=dt} aI bI cI mean std (deviceStreamTag {ex}))
  in MkTensor reg (Just pid))

||| Registered learnable [d0, d1, d2, d3] parameter initialised from `N(mean, std)`.
export
tparam4dNormal : {0 ex : Executor} -> Backend ex dt
              => {a, b, c, e : Nat} -> (paramId : String) -> (mean : Double) -> (std : Double)
              -> IO (Tensor [a, b, c, e] ex dt WithGrad)
tparam4dNormal {a} {b} {c} {e} pid mean std = ioRerun (\_ =>
  let aI = cast {to=Int} a
      bI  = cast {to=Int} b
      cI  = cast {to=Int} c
      eI  = cast {to=Int} e
      reg = primParamRegister {ex} pid (dtCreateParam4dNormal {ex} {t=dt} aI bI cI eI mean std (deviceStreamTag {ex}))
  in MkTensor reg (Just pid))

||| Registered learnable [o, i] parameter filled with `value`. Covers
||| RmsNorm's weight=1.0, BatchNorm beta=0, etc.
export
tparam2dConst : {0 ex : Executor} -> Backend ex dt
             => {o, i : Nat} -> (paramId : String) -> (value : Double)
             -> IO (Tensor [o, i] ex dt WithGrad)
tparam2dConst {o} {i} pid value = ioRerun (\_ =>
  let oI = cast {to=Int} o
      iI  = cast {to=Int} i
      reg = primParamRegister {ex} pid (dtCreateParam2dConst {ex} {t=dt} oI iI value (deviceStreamTag {ex}))
  in MkTensor reg (Just pid))

||| Registered learnable [n] parameter filled with `value`.
export
tparam1dConst : {0 ex : Executor} -> Backend ex dt
             => {n : Nat} -> (paramId : String) -> (value : Double)
             -> IO (Tensor [n] ex dt WithGrad)
tparam1dConst {n} pid value = ioRerun (\_ =>
  let nI = cast {to=Int} n
      reg = primParamRegister {ex} pid (dtCreateParam1dConst {ex} {t=dt} nI value (deviceStreamTag {ex}))
  in MkTensor reg (Just pid))

||| Registered learnable [a, b, c] parameter filled with `value`.
export
tparam3dConst : {0 ex : Executor} -> Backend ex dt
             => {a, b, c : Nat} -> (paramId : String) -> (value : Double)
             -> IO (Tensor [a, b, c] ex dt WithGrad)
tparam3dConst {a} {b} {c} pid value = ioRerun (\_ =>
  let aI = cast {to=Int} a
      bI  = cast {to=Int} b
      cI  = cast {to=Int} c
      reg = primParamRegister {ex} pid (dtCreateParam3dConst {ex} {t=dt} aI bI cI value (deviceStreamTag {ex}))
  in MkTensor reg (Just pid))

||| Registered learnable [a, b, c, e] parameter filled with `value`.
export
tparam4dConst : {0 ex : Executor} -> Backend ex dt
             => {a, b, c, e : Nat} -> (paramId : String) -> (value : Double)
             -> IO (Tensor [a, b, c, e] ex dt WithGrad)
tparam4dConst {a} {b} {c} {e} pid value = ioRerun (\_ =>
  let aI = cast {to=Int} a
      bI  = cast {to=Int} b
      cI  = cast {to=Int} c
      eI  = cast {to=Int} e
      reg = primParamRegister {ex} pid (dtCreateParam4dConst {ex} {t=dt} aI bI cI eI value (deviceStreamTag {ex}))
  in MkTensor reg (Just pid))

||| Seed the backend's init RNG. Subsequent `tparam*Normal` / etc.
||| calls become deterministic per (seed, dtype, shape). No-op on
||| backends without a seedable init-RNG.
export
tsetInitSeed : {0 ex : Executor} -> UserExecutorTraining ex => Bits64 -> IO ()
tsetInitSeed seed = primIO (primSetInitSeedStreamed {ex} seed (deviceStreamTag {ex}))

||| Register an already-constructed tensor (any dtype, grad or not) in the
||| param registry under `paramId`, so checkpointing (`saveModel`) includes
||| it. This is the path for serializing inference-dtype tensors (bf16/f16/
||| int) that aren't learnable params — and the hook the future
||| HuggingFace-checkpoint loader needs (register loaded weights by name).
||| Returns the tensor with its `paramId` set. The `reg` binding is threaded
||| into the result so the registration FFI fires (an unused let is dropped).
export
registerParam : {0 ex : Executor} -> UserExecutorTraining ex => (paramId : String) -> Tensor dims ex dt g -> IO (Tensor dims ex dt g)
registerParam pid t = ioRerun (\_ =>
  let reg = primParamRegister {ex} pid (tensorPtr t)
  in MkTensor reg (Just pid))

||| Wrap an existing 1D tensor handle as a non-parameter input.
||| Pure — no FFI side effect, just record construction.
export
tinput1d : {n : Nat} -> AnyPtr -> Tensor [n] ex dt WithGrad
tinput1d t = MkTensor t Nothing

||| Wrap an existing 2D tensor handle as a non-parameter input.
||| Pure — no FFI side effect, just record construction.
export
tinput2d : {m, n : Nat} -> AnyPtr -> Tensor [m, n] ex dt WithGrad
tinput2d t = MkTensor t Nothing

||| Create a registered learnable scalar parameter (e.g. SAC's
||| state-independent log_std). Mirrors V1's `param`. The optimizer
||| picks it up automatically by paramId scope.
export
tparamScalar : {0 ex : Executor} -> Backend ex dt => (paramId : String) -> (val : Double) -> IO (Tensor [] ex dt WithGrad)
tparamScalar pid val = ioRerun (\_ =>
  let ptr = dtCreateScalar {ex} {t=dt} val 1 (deviceStreamTag {ex})    -- requires_grad=true
      reg = primParamRegister {ex} pid ptr
  in MkTensor reg (Just pid))

----------------------------------------------------------------------
-- Typed construction facade: tensor / param × InitSpec
----------------------------------------------------------------------

-- Fill a fresh host double buffer of n elements per the spec. The
-- random specs sample host-side per element (cold construction path);
-- each prim__setDouble is IO-sequenced via ioRerun so the writes
-- can't reorder across the sampling.
fillSpecBuf : (n : Int) -> InitSpec k -> IO AnyPtr
fillSpecBuf n spec = do
  buf <- ioRerun (\_ => prim__allocDoubles n)
  case spec of
    Zeros         => ioRerun (\_ => fillConst buf 0 0.0)
    Const x       => ioRerun (\_ => fillConst buf 0 x)
    Normal mu sd  => fillIO buf 0 (map (\z => mu + sd * z) normalSample)
    Uniform lo hi => fillIO buf 0 (randomRIO (lo, hi))
    FromVect xs   => ioRerun (\_ => packVect buf 0 xs)
  where
    fillConst : AnyPtr -> Int -> Double -> AnyPtr
    fillConst b i v = if i >= n then b else fillConst (prim__setDouble b i v) (i + 1) v
    fillIO : AnyPtr -> Int -> IO Double -> IO AnyPtr
    fillIO b i sample =
      if i >= n then pure b else do
        v <- sample
        b' <- ioRerun (\_ => prim__setDouble b i v)
        fillIO b' (i + 1) sample
    packVect : AnyPtr -> Int -> Vect m Double -> AnyPtr
    packVect b _ []        = b
    packVect b i (x :: xs) = packVect (prim__setDouble b i x) (i + 1) xs

-- One value for the rank-0 cell of the facade.
scalarSpecValue : InitSpec 1 -> IO Double
scalarSpecValue Zeros           = pure 0.0
scalarSpecValue (Const x)       = pure x
scalarSpecValue (Normal mu sd)  = pure (mu + sd * !normalSample)
scalarSpecValue (Uniform lo hi) = randomRIO (lo, hi)
scalarSpecValue (FromVect [x])  = pure x

||| Create a registered NON-LEARNABLE buffer [n] (PyTorch register_buffer)
||| from a double buffer. Lands in the same registry as a param — so
||| save/load persists it by name with no extra plumbing — but it carries
||| no gradient and `param_is_buffer` flags it so every optimizer / clip /
||| grad-norm walk skips it. Used for running statistics (BatchNorm).
export
tbuffer1d : {0 ex : Executor} -> Backend ex dt => {n : Nat} -> (bufId : String) -> AnyPtr -> IO (Tensor [n] ex dt NoGrad)
tbuffer1d {n} bid buf = ioRerun (\_ =>
  let nI = cast {to=Int} n
      reg = primParamRegisterBuffer {ex} bid (dtCreateState1d {ex} {t=dt} nI buf (deviceStreamTag {ex}))
  in MkTensor reg (Just bid))

||| `tbuffer1d` filled with a constant — the running-stat init path
||| (mean=0, var=1).
export
tbuffer1dConst : {0 ex : Executor} -> Backend ex dt => {n : Nat} -> (bufId : String) -> (value : Double) -> IO (Tensor [n] ex dt NoGrad)
tbuffer1dConst {n} bid value = do
  buf <- fillSpecBuf (cast n) (the (InitSpec n) (Const value))
  tbuffer1d {ex} {dt} {n} bid buf

||| Construct a non-learnable tensor of any rank from an `InitSpec`.
||| `FromVect`'s length is tied to `Numel dims` at compile time, so a
||| data/shape mismatch is a type error. For learnable parameters use
||| `param` (registers with the optimizer registry).
export
tensor : {0 ex : Executor} -> Backend ex dt => {rank : Nat} -> {dims : Vect rank Nat} ->
         InitSpec (Numel dims) -> IO (Tensor dims ex dt NoGrad)
tensor {dims} spec = do
  buf <- fillSpecBuf (cast (Numel dims)) spec
  shp <- ioRerun (\_ => packShape (prim__allocInts (cast rank)) 0 dims)
  ptr <- ioRerun (\_ => dtCreate {ex} {t=dt} buf shp (cast rank) 0 (deviceStreamTag {ex}))
  pure (MkTensor ptr Nothing)
  where
    packShape : AnyPtr -> Int -> Vect m Nat -> AnyPtr
    packShape b _ []        = b
    packShape b i (d :: ds) = packShape (prim__setInt b i (cast d)) (i + 1) ds

||| Construct + register a learnable parameter (rank <= 4 — the C
||| param-create surface's ceiling; rank 5 is a compile error, not a
||| crash). `Normal` / `Const` / `Zeros` route to the fused C init
||| paths (init runs backend-side at memory-bandwidth speed);
||| `Uniform` / `FromVect` fill a host buffer. Always registers under
||| `name` — parameters without a registry entry are invisible to the
||| optimizer.
export
param : {0 ex : Executor} -> Backend ex dt => {rank : Nat} -> {dims : Vect rank Nat} ->
        {auto rankOk : LTE rank 4} ->
        (name : String) -> InitSpec (Numel dims) -> IO (Tensor dims ex dt WithGrad)
param {dims = []} pid spec = do
  v <- scalarSpecValue spec
  tparamScalar {ex} pid v
param {dims = [n]} pid spec = do
  let nI = cast {to=Int} n
  ptr <- case spec of
    Normal mu sd => ioRerun (\_ => dtCreateParam1dNormal {ex} {t=dt} nI mu sd (deviceStreamTag {ex}))
    Const x      => ioRerun (\_ => dtCreateParam1dConst {ex} {t=dt} nI x (deviceStreamTag {ex}))
    Zeros        => ioRerun (\_ => dtCreateParam1dConst {ex} {t=dt} nI 0.0 (deviceStreamTag {ex}))
    _            => do buf <- fillSpecBuf nI spec
                       ioRerun (\_ => dtCreateParam1d {ex} {t=dt} nI buf (deviceStreamTag {ex}))
  reg <- ioRerun (\_ => primParamRegister {ex} pid ptr)
  pure (MkTensor reg (Just pid))
param {dims = [m, n]} pid spec = do
  let mI = cast {to=Int} m
  let nI = cast {to=Int} n
  ptr <- case spec of
    Normal mu sd => ioRerun (\_ => dtCreateParam2dNormal {ex} {t=dt} mI nI mu sd (deviceStreamTag {ex}))
    Const x      => ioRerun (\_ => dtCreateParam2dConst {ex} {t=dt} mI nI x (deviceStreamTag {ex}))
    Zeros        => ioRerun (\_ => dtCreateParam2dConst {ex} {t=dt} mI nI 0.0 (deviceStreamTag {ex}))
    _            => do buf <- fillSpecBuf (mI * nI) spec
                       ioRerun (\_ => dtCreateParam2d {ex} {t=dt} mI nI buf (deviceStreamTag {ex}))
  reg <- ioRerun (\_ => primParamRegister {ex} pid ptr)
  pure (MkTensor reg (Just pid))
param {dims = [a, b, c]} pid spec = do
  let aI = cast {to=Int} a
  let bI = cast {to=Int} b
  let cI = cast {to=Int} c
  ptr <- case spec of
    Normal mu sd => ioRerun (\_ => dtCreateParam3dNormal {ex} {t=dt} aI bI cI mu sd (deviceStreamTag {ex}))
    Const x      => ioRerun (\_ => dtCreateParam3dConst {ex} {t=dt} aI bI cI x (deviceStreamTag {ex}))
    Zeros        => ioRerun (\_ => dtCreateParam3dConst {ex} {t=dt} aI bI cI 0.0 (deviceStreamTag {ex}))
    _            => do buf <- fillSpecBuf (aI * bI * cI) spec
                       ioRerun (\_ => dtCreateParam3d {ex} {t=dt} aI bI cI buf (deviceStreamTag {ex}))
  reg <- ioRerun (\_ => primParamRegister {ex} pid ptr)
  pure (MkTensor reg (Just pid))
param {dims = [a, b, c, e]} pid spec = do
  let aI = cast {to=Int} a
  let bI = cast {to=Int} b
  let cI = cast {to=Int} c
  let eI = cast {to=Int} e
  ptr <- case spec of
    Normal mu sd => ioRerun (\_ => dtCreateParam4dNormal {ex} {t=dt} aI bI cI eI mu sd (deviceStreamTag {ex}))
    Const x      => ioRerun (\_ => dtCreateParam4dConst {ex} {t=dt} aI bI cI eI x (deviceStreamTag {ex}))
    Zeros        => ioRerun (\_ => dtCreateParam4dConst {ex} {t=dt} aI bI cI eI 0.0 (deviceStreamTag {ex}))
    _            => do buf <- fillSpecBuf (aI * bI * cI * eI) spec
                       ioRerun (\_ => dtCreateParam4d {ex} {t=dt} aI bI cI eI buf (deviceStreamTag {ex}))
  reg <- ioRerun (\_ => primParamRegister {ex} pid ptr)
  pure (MkTensor reg (Just pid))
param {dims = _ :: _ :: _ :: _ :: _ :: _} {rankOk = LTESucc (LTESucc (LTESucc (LTESucc p)))} _ _ =
  absurd p
