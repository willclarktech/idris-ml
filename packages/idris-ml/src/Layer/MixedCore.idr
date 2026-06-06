module Layer.MixedCore

import Data.Vect
import System
import System.File

import Executor
import Tensor
import Layer.Core


----------------------------------------------------------------------
-- LayerLikeMixed: layer interface with separate param/compute dtypes
----------------------------------------------------------------------
--
-- Parallel to `LayerLike`, but the layer's type carries TWO dtype
-- slots: `paramDt` (where weights are stored) and `computeDt` (where
-- activations flow). For mixed-precision training the typical case
-- is `paramDt = F32` master + `computeDt = BF16` compute — the cast
-- happens inside the layer's forward, autograd-aware, so the master
-- weight gets a real F32 gradient back through the cast.
--
-- For plain (non-mixed) layers, `paramDt == computeDt`. A bridge
-- (`AsMixed`) wraps any `AnyLayer` so it satisfies `LayerLikeMixed`
-- with the two slots identified — meaning all 15 existing layers
-- slot in unchanged via `liftAnyLayer` / `liftNetwork`.
--
-- This interface is the structural prerequisite for `LinearMixed`,
-- `BitLinear`, and other quantized / mixed-precision layers, without
-- forcing the full single-dtype `LayerLike` system to migrate.

public export
interface LayerLikeMixed (l : Nat -> Nat -> (0 _ : Executor) -> (0 _ : DType) -> (0 _ : DType) -> (0 _ : GradMode) -> Type) where
  ||| Forward: input is in computeDt, output is in computeDt; weights
  ||| (if any) are stored in paramDt and cast internally.
  ||| Constraints are named so instance bodies can disambiguate when
  ||| `pDt` and `cDt` happen to unify to the same type (e.g. in
  ||| the `AsMixed` bridge). `IsDType pDt` is required so layers can
  ||| call `tcastUnsafe` to materialise the paramDt → computeDt cast
  ||| inside their forward.
  |||
  ||| `UserExecutorQuant ex` is in the constraint list so quantization-
  ||| aware layers (BitLinear under #411) can call `tBitlinearFwd`
  ||| from their `applyVarMixed`. All three built-in backends
  ||| implement `UserExecutorQuant`; BYO backends that want to slot
  ||| layers into a `NetworkMixed` must implement it too (stub the
  ||| methods with `idris_crash` if they don't ship BitNet kernels).
  applyVarMixed : {0 ex : Executor} -> UserExecutorTraining ex => UserExecutorCore ex =>
                  UserExecutorQuant ex =>
                  IsDType pDt => IsDType cDt =>
                  {auto rdtP : RuntimeDType pDt} ->
                  {auto rdtC : RuntimeDType cDt} ->
                  Linked ex =>
                  {auto cmpP : Compatible ex pDt} ->
                  {auto cmpC : Compatible ex cDt} ->
                  {0 g : GradMode} -> {i, o : Nat} ->
                  l i o ex pDt cDt g -> Tensor [i] ex cDt g ->
                  IO (l i o ex pDt cDt g, Tensor [o] ex cDt g)

  ||| Auto-naming prefix (mirrors `LayerLike.layerPrefix`).
  layerPrefixMixed : {0 ex : Executor} -> {0 g : GradMode} -> {i, o : Nat} ->
                     l i o ex pDt cDt g -> String
  layerPrefixMixed _ = ""

  ||| Reset per-sequence state. Default = id; recurrent layers override.
  resetStateMixed : {0 ex : Executor} -> {0 g : GradMode} -> {i, o : Nat} ->
                    l i o ex pDt cDt g -> l i o ex pDt cDt g
  resetStateMixed = id

  ||| Batched forward (default crashes; layers participating in
  ||| batched training override).
  applyVarBatchMixed : {0 ex : Executor} -> UserExecutorTraining ex => UserExecutorCore ex =>
                       UserExecutorQuant ex =>
                       IsDType pDt => IsDType cDt =>
                       {auto rdtP : RuntimeDType pDt} ->
                       {auto rdtC : RuntimeDType cDt} ->
                       Linked ex =>
                       {auto cmpP : Compatible ex pDt} ->
                       {auto cmpC : Compatible ex cDt} ->
                       {0 g : GradMode} -> {i, o : Nat} -> {b : Nat} ->
                       l i o ex pDt cDt g -> Tensor [b, i] ex cDt g ->
                       IO (l i o ex pDt cDt g, Tensor [b, o] ex cDt g)
  applyVarBatchMixed _ _ =
    idris_crash "applyVarBatchMixed: layer does not support batched forward"

  ||| Freeze the layer's parameters. Linear in input (mirrors
  ||| `LayerLike.freezeLayer`).
  freezeLayerMixed : {0 ex : Executor} -> UserExecutorTraining ex =>
                     {0 g : GradMode} -> {i, o : Nat} ->
                     (1 _ : l i o ex pDt cDt g) -> IO (l i o ex pDt cDt NoGrad)

  ||| Inverse of `freezeLayerMixed`. Linear in input.
  unfreezeLayerMixed : {0 ex : Executor} -> UserExecutorTraining ex =>
                       {i, o : Nat} ->
                       (1 _ : l i o ex pDt cDt NoGrad) -> IO (l i o ex pDt cDt WithGrad)


----------------------------------------------------------------------
-- AsMixed: bridge AnyLayer → LayerLikeMixed (paramDt = computeDt)
----------------------------------------------------------------------
--
-- Wraps any `AnyLayer` (existential over LayerLike layers) so it
-- satisfies `LayerLikeMixed` with both dtype slots identified. The
-- constructor only inhabits the diagonal `pDt = cDt`, so the wrapper
-- enforces the auto-conformance precondition at the type level.
--
-- This avoids parameterising `AsMixed` over a higher-order layer-kind
-- `l` (Idris 2 multiplicity inference doesn't propagate erasure
-- annotations through higher-order data parameters cleanly). Going
-- through `AnyLayer` is zero extra cost: the AnyLayer existential is
-- already the standard chaining surface.

public export
data AsMixed : Nat -> Nat -> (0 _ : Executor) -> (0 _ : DType) -> (0 _ : DType) -> (0 _ : GradMode) -> Type where
  MkAsMixed : AnyLayer i o ex dt g -> AsMixed i o ex dt dt g

public export
LayerLikeMixed AsMixed where
  -- After matching `MkAsMixed`, `pDt = cDt = dt`. Bind the cDt-side
  -- auto-implicits by name (`rdtC`, `cmpC`) and pass them explicitly
  -- to `applyVar` to disambiguate from the otherwise-equally-applicable
  -- `rdtP` / `cmpP` (which would also unify after `pDt = cDt`).
  applyVarMixed {rdtC} {cmpC} (MkAsMixed (MkAnyLayer l @{dict} layer)) input = do
    -- Position-skip slots that auto-resolve (UDT, UDC, Linked); pin
    -- the RuntimeDType + Compatible slots to the cDt-side dicts so
    -- they don't conflict with rdtP / cmpP after pDt = cDt.
    (layer', out) <- applyVar @{dict} @{%search} @{%search} @{rdtC} @{%search} @{cmpC} layer input
    pure (MkAsMixed (MkAnyLayer l @{dict} layer'), out)
  layerPrefixMixed (MkAsMixed (MkAnyLayer _ @{dict} layer)) =
    layerPrefix @{dict} layer
  resetStateMixed (MkAsMixed (MkAnyLayer l @{dict} layer)) =
    MkAsMixed (MkAnyLayer l @{dict} (resetState @{dict} layer))
  applyVarBatchMixed {rdtC} {cmpC} (MkAsMixed (MkAnyLayer l @{dict} layer)) input = do
    (layer', out) <- applyVarBatch @{dict} @{%search} @{%search} @{rdtC} @{%search} @{cmpC} layer input
    pure (MkAsMixed (MkAnyLayer l @{dict} layer'), out)
  freezeLayerMixed (MkAsMixed (MkAnyLayer l @{dict} layer)) = do
    layer' <- freezeLayer @{dict} layer
    pure (MkAsMixed (MkAnyLayer l @{dict} layer'))
  unfreezeLayerMixed (MkAsMixed (MkAnyLayer l @{dict} layer)) = do
    layer' <- unfreezeLayer @{dict} layer
    pure (MkAsMixed (MkAnyLayer l @{dict} layer'))


----------------------------------------------------------------------
-- AnyLayerMixed (existential wrapper)
----------------------------------------------------------------------

public export
data AnyLayerMixed : Nat -> Nat -> (0 _ : Executor) -> (0 _ : DType) -> (0 _ : DType) -> (0 _ : GradMode) -> Type where
  MkAnyLayerMixed : (l : Nat -> Nat -> (0 _ : Executor) -> (0 _ : DType) -> (0 _ : DType) -> (0 _ : GradMode) -> Type) ->
                    LayerLikeMixed l =>
                    l i o ex pDt cDt g -> AnyLayerMixed i o ex pDt cDt g

export
applyVarAnyMixed : {0 ex : Executor} -> UserExecutorTraining ex => UserExecutorCore ex =>
                   UserExecutorQuant ex =>
                   IsDType pDt => IsDType cDt =>
                   RuntimeDType pDt => RuntimeDType cDt =>
                   Linked ex => Compatible ex pDt => Compatible ex cDt =>
                   {0 g : GradMode} -> {i, o : Nat} ->
                   AnyLayerMixed i o ex pDt cDt g -> Tensor [i] ex cDt g ->
                   IO (AnyLayerMixed i o ex pDt cDt g, Tensor [o] ex cDt g)
applyVarAnyMixed (MkAnyLayerMixed l @{dict} layer) input = do
  (layer', out) <- applyVarMixed @{dict} layer input
  pure (MkAnyLayerMixed l @{dict} layer', out)

export
applyVarBatchAnyMixed : {0 ex : Executor} -> UserExecutorTraining ex => UserExecutorCore ex =>
                        UserExecutorQuant ex =>
                        IsDType pDt => IsDType cDt =>
                        RuntimeDType pDt => RuntimeDType cDt =>
                        Linked ex => Compatible ex pDt => Compatible ex cDt =>
                        {0 g : GradMode} -> {i, o : Nat} -> {b : Nat} ->
                        AnyLayerMixed i o ex pDt cDt g -> Tensor [b, i] ex cDt g ->
                        IO (AnyLayerMixed i o ex pDt cDt g, Tensor [b, o] ex cDt g)
applyVarBatchAnyMixed (MkAnyLayerMixed l @{dict} layer) input = do
  (layer', out) <- applyVarBatchMixed @{dict} layer input
  pure (MkAnyLayerMixed l @{dict} layer', out)

export
freezeAnyLayerMixed : {0 ex : Executor} -> UserExecutorTraining ex =>
                      {0 g : GradMode} -> {i, o : Nat} ->
                      (1 _ : AnyLayerMixed i o ex pDt cDt g) ->
                      IO (AnyLayerMixed i o ex pDt cDt NoGrad)
freezeAnyLayerMixed (MkAnyLayerMixed l @{dict} layer) = do
  layer' <- freezeLayerMixed @{dict} layer
  pure (MkAnyLayerMixed l @{dict} layer')

export
unfreezeAnyLayerMixed : {0 ex : Executor} -> UserExecutorTraining ex =>
                        {i, o : Nat} ->
                        (1 _ : AnyLayerMixed i o ex pDt cDt NoGrad) ->
                        IO (AnyLayerMixed i o ex pDt cDt WithGrad)
unfreezeAnyLayerMixed (MkAnyLayerMixed l @{dict} layer) = do
  layer' <- unfreezeLayerMixed @{dict} layer
  pure (MkAnyLayerMixed l @{dict} layer')


----------------------------------------------------------------------
-- NetworkMixed (chain)
----------------------------------------------------------------------

public export
data NetworkMixed : (i : Nat) -> (hs : List Nat) -> (o : Nat) -> (0 _ : Executor) -> (0 _ : DType) -> (0 _ : DType) -> (0 _ : GradMode) -> Type where
  OutputLayerMixed : AnyLayerMixed i o ex pDt cDt g -> NetworkMixed i [] o ex pDt cDt g
  (~~~>) : AnyLayerMixed i h ex pDt cDt g -> NetworkMixed h hs o ex pDt cDt g -> NetworkMixed i (h :: hs) o ex pDt cDt g

export infixr 5 ~~~>

||| Array-level forward through a NetworkMixed.
export
forwardVarMixed : {0 ex : Executor} -> UserExecutorTraining ex => UserExecutorCore ex =>
                  UserExecutorQuant ex =>
                  IsDType pDt => IsDType cDt =>
                  RuntimeDType pDt => RuntimeDType cDt =>
                  Linked ex => Compatible ex pDt => Compatible ex cDt =>
                  {0 g : GradMode} -> {i, o : Nat} -> {hs : List Nat} ->
                  NetworkMixed i hs o ex pDt cDt g -> Tensor [i] ex cDt g ->
                  IO (NetworkMixed i hs o ex pDt cDt g, Tensor [o] ex cDt g)
forwardVarMixed (OutputLayerMixed l) input = do
  (l', out) <- applyVarAnyMixed l input
  pure (OutputLayerMixed l', out)
forwardVarMixed {hs = h :: _} (l ~~~> rest) input = do
  (l', mid) <- applyVarAnyMixed l input
  (rest', out) <- forwardVarMixed rest mid
  pure (l' ~~~> rest', out)

||| Freeze a NetworkMixed. Linear in input.
export
freezeNetworkMixed : {0 ex : Executor} -> UserExecutorTraining ex =>
                     {0 g : GradMode} -> {i, o : Nat} -> {hs : List Nat} ->
                     (1 _ : NetworkMixed i hs o ex pDt cDt g) ->
                     IO (NetworkMixed i hs o ex pDt cDt NoGrad)
freezeNetworkMixed (OutputLayerMixed l) = do
  l' <- freezeAnyLayerMixed l
  pure (OutputLayerMixed l')
freezeNetworkMixed {hs = h :: _} (l ~~~> rest) = do
  l' <- freezeAnyLayerMixed l
  rest' <- freezeNetworkMixed rest
  pure (l' ~~~> rest')

||| Inverse of `freezeNetworkMixed`. Linear in input.
export
unfreezeNetworkMixed : {0 ex : Executor} -> UserExecutorTraining ex =>
                       {i, o : Nat} -> {hs : List Nat} ->
                       (1 _ : NetworkMixed i hs o ex pDt cDt NoGrad) ->
                       IO (NetworkMixed i hs o ex pDt cDt WithGrad)
unfreezeNetworkMixed (OutputLayerMixed l) = do
  l' <- unfreezeAnyLayerMixed l
  pure (OutputLayerMixed l')
unfreezeNetworkMixed {hs = h :: _} (l ~~~> rest) = do
  l' <- unfreezeAnyLayerMixed l
  rest' <- unfreezeNetworkMixed rest
  pure (l' ~~~> rest')

||| Reset per-sequence state on every layer.
export
resetNetworkMixed : {0 ex : Executor} -> {0 g : GradMode} -> {i, o : Nat} -> {hs : List Nat} ->
                    NetworkMixed i hs o ex pDt cDt g -> NetworkMixed i hs o ex pDt cDt g
resetNetworkMixed (OutputLayerMixed (MkAnyLayerMixed l @{dict} layer)) =
  OutputLayerMixed (MkAnyLayerMixed l @{dict} (resetStateMixed @{dict} layer))
resetNetworkMixed ((MkAnyLayerMixed l @{dict} layer) ~~~> rest) =
  MkAnyLayerMixed l @{dict} (resetStateMixed @{dict} layer) ~~~> resetNetworkMixed rest

||| Batched tensor-level forward through a NetworkMixed.
export
forwardVarBatchMixed : {0 ex : Executor} -> UserExecutorTraining ex => UserExecutorCore ex =>
                       UserExecutorQuant ex =>
                       IsDType pDt => IsDType cDt =>
                       RuntimeDType pDt => RuntimeDType cDt =>
                       Linked ex => Compatible ex pDt => Compatible ex cDt =>
                       {0 g : GradMode} -> {i, o : Nat} -> {b : Nat} ->
                       {hs : List Nat} ->
                       NetworkMixed i hs o ex pDt cDt g -> Tensor [b, i] ex cDt g ->
                       IO (NetworkMixed i hs o ex pDt cDt g, Tensor [b, o] ex cDt g)
forwardVarBatchMixed (OutputLayerMixed l) input = do
  (l', out) <- applyVarBatchAnyMixed l input
  pure (OutputLayerMixed l', out)
forwardVarBatchMixed {hs = h :: _} (l ~~~> rest) input = do
  (l', mid) <- applyVarBatchAnyMixed l input
  (rest', out) <- forwardVarBatchMixed rest mid
  pure (l' ~~~> rest', out)


----------------------------------------------------------------------
-- Lifts: AnyLayer / Network → AnyLayerMixed / NetworkMixed
----------------------------------------------------------------------

||| Lift an existing single-dtype `AnyLayer` into the mixed-precision
||| world. Useful for chaining ordinary layers alongside mixed-precision
||| ones in a single `NetworkMixed`. Zero runtime cost — the wrapper
||| stores the underlying layer and the `LayerLikeMixed AsMixed`
||| instance delegates each method back to the wrapped `AnyLayer`.
public export
liftAnyLayer : AnyLayer i o ex dt g -> AnyLayerMixed i o ex dt dt g
liftAnyLayer al = MkAnyLayerMixed AsMixed (MkAsMixed al)

||| Lift an entire `Network` into a `NetworkMixed` with both dtype
||| slots identified.
public export
liftNetwork : Network i hs o ex dt g -> NetworkMixed i hs o ex dt dt g
liftNetwork (OutputLayer l) = OutputLayerMixed (liftAnyLayer l)
liftNetwork (l ~~> rest) = liftAnyLayer l ~~~> liftNetwork rest
