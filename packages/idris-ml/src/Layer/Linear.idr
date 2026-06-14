module Layer.Linear

import Data.Vect

import Compat.Random
import Executor
import Init
import Layer.Core
import Sampler
import Tensor

----------------------------------------------------------------------
-- Linear State (Path C P3-1 spike)
----------------------------------------------------------------------
--
-- Direct shape-aware tensor weights — no scalar matrix, no view
-- Variables, no `Maybe AnyPtr` dual mode. The Tensor IS the weight.
-- Bias and weight are registered C params at construction time.

public export
record LinearState (i : Nat) (o : Nat) (0 ex : Executor) (0 dt : DType) (0 g : GradMode) where
  constructor MkLinear
  weightT : Tensor [o, i] ex dt g
  biasT   : Tensor [o] ex dt g

----------------------------------------------------------------------
-- LayerLike instance — two-line forward
----------------------------------------------------------------------

%default partial

public export
LayerLike LinearState where
  applyVar st input = do
    out <- tlinear st.weightT input st.biasT
    pure (st, out)

  applyVarBatch st input = do
    out <- tlinear2d st.weightT input st.biasT
    pure (st, out)

  layerPrefix _ = "llv2"

  freezeLayer (MkLinear w b) = do
    w' <- weakenGrad w
    b' <- weakenGrad b
    pure (MkLinear w' b')

  unfreezeLayer (MkLinear w b) = do
    primIO (primSetRequiresGrad {ex} w.tensorPtr 1)
    primIO (primSetRequiresGrad {ex} b.tensorPtr 1)
    pure (MkLinear (retypeGrad w) (retypeGrad b))

----------------------------------------------------------------------
-- Constructor
----------------------------------------------------------------------

||| Pack a Vect of Doubles into a pre-allocated buffer at offset.
||| Exported so other layer modules can reuse the same packing logic.
export
packDoubles : AnyPtr -> Int -> Vect k Double -> AnyPtr
packDoubles buf _ []            = buf
packDoubles buf off (x :: rest) =
  packDoubles (prim__setDouble buf off x) (off + 1) rest

-- Zero a buffer for `n` elements starting at offset.
zeroBuf : AnyPtr -> Int -> Int -> AnyPtr
zeroBuf buf _ 0   = buf
zeroBuf buf off n =
  zeroBuf (prim__setDouble buf off 0.0) (off + 1) (n - 1)

||| Build a `LinearState i o` with normal-distributed weights and a
||| user-specified bias init. `weightStd = 0` is invalid (a zero-std
||| normal would produce a constant-zero weight matrix); pass a
||| positive value. `biasStd > 0` samples bias ~ N(0, biasStd);
||| `biasStd = 0` registers a zero-init bias via tparam1dConst.
|||
||| This replaces the old `mkLinearWith wInit bInit` (which took an
||| `InitStrategy` + `IO Double`). The new shape mirrors PyTorch's
||| `torch.nn.init.normal_(weight, std=...)` + `torch.nn.init.normal_(bias, std=...)`
||| pair; callers wanting other distributions wire the underlying
||| FFI directly.
export
mkLinearWith : Backend ex dt => {i, o : Nat}
            -> (paramPrefix : String)
            -> (weightStd : Double)
            -> (biasStd : Double)
            -> IO (LinearState i o ex dt WithGrad)
mkLinearWith pfx wStd bStd = do
  let wName = pfx ++ "_weights"
      bName = pfx ++ "_biases"
  w <- tparam2dNormal {ex} {dt} {o} {i} wName 0.0 wStd
  b <- if bStd == 0.0
         then tparam1dConst  {ex} {dt} {n=o} bName 0.0
         else tparam1dNormal {ex} {dt} {n=o} bName 0.0 bStd
  pure $ MkLinear w b

||| Build a `LinearState i o` with PyTorch's default `nn.Linear` init
||| (normal approximation): weight ~ N(0, 1/sqrt(fan_in)), zero bias.
||| Was Xavier-uniform pre-2026-05-28; switched to normal here to
||| match the new fused-init primitive surface — see plan P3 lock-in.
||| Registers under `<paramPrefix>_weights` / `<paramPrefix>_biases`.
export
linearLayer : Backend ex dt => {i, o : Nat} -> (paramPrefix : String) -> IO (LinearState i o ex dt WithGrad)
linearLayer paramPrefix =
  mkLinearWith paramPrefix (1.0 / sqrt (cast {to=Double} i)) 0.0

||| Wrap a Linear in `AnyLayer` for use in a `Network`.
export
linearLayerAny : Backend ex dt => {i, o : Nat} -> (paramPrefix : String) -> IO (AnyLayer i o ex dt WithGrad)
linearLayerAny pid = map (MkAnyLayer LinearState) (linearLayer pid)
