module Layer.Core

import Data.Maybe
import Data.String
import Data.Vect
import System
import System.Directory
import System.File

import Executor
import Tensor
import Util.Log


----------------------------------------------------------------------
-- Path C P3-1 spike: rank-aware layer interface
----------------------------------------------------------------------
--
-- Parallel to `Layer.Core.LayerLike`, but operates on rank-aware
-- `Tensor` directly. No Vect-of-Vect packing; no scalar boundaries
-- between layers. Spike-only — only the methods needed for a
-- single Linear + chained forward pass are present. Full
-- migration widens this surface (toDouble, debug, batched, etc.).

public export
interface LayerLike (l : Nat -> Nat -> (0 _ : Executor) -> (0 _ : DType) -> (0 _ : GradMode) -> Type) where
  ||| Array-level forward: `Tensor [i] ex dt g-> Tensor [o] ex dt g`, IO-typed
  ||| because the forward pass triggers FFI side effects (tape append,
  ||| tensor allocation). IO sequencing controls when those fire —
  ||| critical for `withNoGrad` to correctly bracket eval-phase work.
  ||| Polymorphic in `g` so forwarding a `NoGrad` input through a
  ||| frozen layer yields a `NoGrad` output naturally.
  applyVar : {0 ex : Executor} -> UserExecutorTraining ex => UserExecutorCore ex => RuntimeDType dt => Linked ex => Compatible ex dt => {0 g : GradMode} -> {i, o : Nat} ->
              l i o ex dt g -> Tensor [i] ex dt g -> IO (l i o ex dt g, Tensor [o] ex dt g)

  ||| Auto-naming prefix (e.g. "llv2" for Linear).
  layerPrefix : {0 ex : Executor} -> {0 g : GradMode} -> {i, o : Nat} -> l i o ex dt g -> String
  layerPrefix _ = ""

  ||| Reset per-sequence state (recurrent layers override; default = id).
  ||| Used by `resetNetwork` between sequences in recurrent training.
  resetState : {0 ex : Executor} -> {0 g : GradMode} -> {i, o : Nat} -> l i o ex dt g -> l i o ex dt g
  resetState = id

  ||| Batched tensor-level forward: `Tensor [b, i] ex dt g-> Tensor [b, o] ex dt g`.
  ||| Default crashes — layers that participate in batched training
  ||| (Linear, Activation, Dropout) MUST override. Stateful layers
  ||| (LSTM/RNN/GRU/NTM/DNC) keep the default; batched-cell semantics
  ||| are not supported in this surface (use sequence-level batching
  ||| at the example level instead).
  applyVarBatch : {0 ex : Executor} -> UserExecutorTraining ex => UserExecutorCore ex => RuntimeDType dt => Linked ex => Compatible ex dt => {0 g : GradMode} -> {i, o : Nat} -> {b : Nat} ->
                   l i o ex dt g -> Tensor [b, i] ex dt g -> IO (l i o ex dt g, Tensor [b, o] ex dt g)
  applyVarBatch _ _ =
    idris_crash "applyVarBatch: layer does not support batched forward"

  ||| Freeze: flip C-side `requires_grad=false` on every parameter
  ||| tensor in the layer state. Linear in input — consumes the old
  ||| reference so the caller can't keep using the WithGrad-typed
  ||| value after the C-side flags have been mutated. Returns the
  ||| layer retyped as `NoGrad`. Optimizer steps won't update frozen
  ||| params (their gradients don't accumulate on rg=false leaves).
  freezeLayer : {0 ex : Executor} -> UserExecutorTraining ex => {0 g : GradMode} -> {i, o : Nat} ->
                (1 _ : l i o ex dt g) -> IO (l i o ex dt NoGrad)

  ||| Inverse of `freezeLayer`. Sets `requires_grad=true` on every
  ||| parameter and retypes the layer as `WithGrad`. The result is
  ||| trainable again. Linear in input.
  unfreezeLayer : {0 ex : Executor} -> UserExecutorTraining ex => {i, o : Nat} ->
                  (1 _ : l i o ex dt NoGrad) -> IO (l i o ex dt WithGrad)


----------------------------------------------------------------------
-- AnyLayer (existential wrapper)
----------------------------------------------------------------------

public export
data AnyLayer : Nat -> Nat -> (0 _ : Executor) -> (0 _ : DType) -> (0 _ : GradMode) -> Type where
  MkAnyLayer : (l : Nat -> Nat -> (0 _ : Executor) -> (0 _ : DType) -> (0 _ : GradMode) -> Type) -> LayerLike l =>
                 l i o ex dt g -> AnyLayer i o ex dt g

export
applyVarAny : {0 ex : Executor} -> UserExecutorTraining ex => UserExecutorCore ex => RuntimeDType dt => Linked ex => Compatible ex dt => {0 g : GradMode} -> {i, o : Nat} ->
               AnyLayer i o ex dt g -> Tensor [i] ex dt g -> IO (AnyLayer i o ex dt g, Tensor [o] ex dt g)
applyVarAny (MkAnyLayer l @{dict} layer) input = do
  (layer', out) <- applyVar @{dict} layer input
  pure (MkAnyLayer l @{dict} layer', out)

export
applyVarBatchAny : {0 ex : Executor} -> UserExecutorTraining ex => UserExecutorCore ex => RuntimeDType dt => Linked ex => Compatible ex dt => {0 g : GradMode} -> {i, o : Nat} -> {b : Nat} ->
                    AnyLayer i o ex dt g -> Tensor [b, i] ex dt g ->
                    IO (AnyLayer i o ex dt g, Tensor [b, o] ex dt g)
applyVarBatchAny (MkAnyLayer l @{dict} layer) input = do
  (layer', out) <- applyVarBatch @{dict} layer input
  pure (MkAnyLayer l @{dict} layer', out)

export
freezeAnyLayer : {0 ex : Executor} -> UserExecutorTraining ex => {0 g : GradMode} -> {i, o : Nat} ->
                  (1 _ : AnyLayer i o ex dt g) -> IO (AnyLayer i o ex dt NoGrad)
freezeAnyLayer (MkAnyLayer l @{dict} layer) = do
  layer' <- freezeLayer @{dict} layer
  pure (MkAnyLayer l @{dict} layer')

export
unfreezeAnyLayer : {0 ex : Executor} -> UserExecutorTraining ex => {i, o : Nat} ->
                    (1 _ : AnyLayer i o ex dt NoGrad) -> IO (AnyLayer i o ex dt WithGrad)
unfreezeAnyLayer (MkAnyLayer l @{dict} layer) = do
  layer' <- unfreezeLayer @{dict} layer
  pure (MkAnyLayer l @{dict} layer')


----------------------------------------------------------------------
-- Network
----------------------------------------------------------------------

public export
data Network : (i : Nat) -> (hs : List Nat) -> (o : Nat) -> (0 _ : Executor) -> (0 _ : DType) -> (0 _ : GradMode) -> Type where
  OutputLayer : AnyLayer i o ex dt g -> Network i [] o ex dt g
  (~~>) : AnyLayer i h ex dt g -> Network h hs o ex dt g -> Network i (h :: hs) o ex dt g

export infixr 5 ~~>

||| Array-level forward through a Network. Polymorphic in `g`:
||| forwarding a `NoGrad` input through a frozen network yields a
||| `NoGrad` output naturally.
export
forwardVar : {0 ex : Executor} -> UserExecutorTraining ex => UserExecutorCore ex => RuntimeDType dt => Linked ex => Compatible ex dt => {0 g : GradMode} -> {i, o : Nat} -> {hs : List Nat} ->
              Network i hs o ex dt g -> Tensor [i] ex dt g -> IO (Network i hs o ex dt g, Tensor [o] ex dt g)
forwardVar (OutputLayer l) input = do
  (l', out) <- applyVarAny l input
  pure (OutputLayer l', out)
forwardVar {hs = h :: _} (l ~~> rest) input = do
  (l', mid) <- applyVarAny l input
  (rest', out) <- forwardVar rest mid
  pure (l' ~~> rest', out)

||| Freeze a Network: walks each layer and calls `freezeLayer` on it,
||| which flips C-side `requires_grad=false` on every parameter tensor.
||| Linear in input — the original WithGrad-typed reference is consumed
||| so the user can't accidentally train through it (the C-side flags
||| have been mutated under the original Idris variable).
||| Returns the network retyped as `NoGrad`.
|||
||| Frozen networks remain usable with `forwardVar` (now polymorphic in
||| `g`) — output adopts `NoGrad` and the type system prevents feeding
||| it back to `runBackward` / `nativeTrainStep`.
export
freezeNetwork : {0 ex : Executor} -> UserExecutorTraining ex => {0 g : GradMode} -> {i, o : Nat} -> {hs : List Nat} ->
                 (1 _ : Network i hs o ex dt g) -> IO (Network i hs o ex dt NoGrad)
freezeNetwork (OutputLayer l) = do
  l' <- freezeAnyLayer l
  pure (OutputLayer l')
freezeNetwork {hs = h :: _} (l ~~> rest) = do
  l' <- freezeAnyLayer l
  rest' <- freezeNetwork rest
  pure (l' ~~> rest')

||| Inverse of `freezeNetwork`: sets `requires_grad=true` on every
||| parameter and retypes the network as `WithGrad`. Linear in input.
||| Use for progressive fine-tuning workflows (train head with backbone
||| frozen, then unfreeze backbone for joint fine-tuning).
export
unfreezeNetwork : {0 ex : Executor} -> UserExecutorTraining ex => {i, o : Nat} -> {hs : List Nat} ->
                   (1 _ : Network i hs o ex dt NoGrad) -> IO (Network i hs o ex dt WithGrad)
unfreezeNetwork (OutputLayer l) = do
  l' <- unfreezeAnyLayer l
  pure (OutputLayer l')
unfreezeNetwork {hs = h :: _} (l ~~> rest) = do
  l' <- unfreezeAnyLayer l
  rest' <- unfreezeNetwork rest
  pure (l' ~~> rest')

||| Reset per-sequence state on every layer in the network. Use
||| between training sequences for recurrent layers (Lstm, Rnn,
||| Gru). Stateless layers' default `resetState` is identity.
export
resetNetwork : {0 ex : Executor} -> {0 g : GradMode} -> {i, o : Nat} -> {hs : List Nat} ->
                 Network i hs o ex dt g -> Network i hs o ex dt g
resetNetwork (OutputLayer (MkAnyLayer l @{dict} layer)) =
  OutputLayer (MkAnyLayer l @{dict} (resetState @{dict} layer))
resetNetwork ((MkAnyLayer l @{dict} layer) ~~> rest) =
  MkAnyLayer l @{dict} (resetState @{dict} layer) ~~> resetNetwork rest

||| Batched tensor-level forward through a Network: each layer's
||| `applyVarBatch` runs on the threaded `[b, _]` tensor. Linear /
||| Activation / Dropout override; other layers crash via the
||| interface default.
export
forwardVarBatch : {0 ex : Executor} -> UserExecutorTraining ex => UserExecutorCore ex => RuntimeDType dt => Linked ex => Compatible ex dt => {0 g : GradMode} -> {i, o : Nat} -> {b : Nat} ->
                   {hs : List Nat} ->
                   Network i hs o ex dt g -> Tensor [b, i] ex dt g ->
                   IO (Network i hs o ex dt g, Tensor [b, o] ex dt g)
forwardVarBatch (OutputLayer l) input = do
  (l', out) <- applyVarBatchAny l input
  pure (OutputLayer l', out)
forwardVarBatch {hs = h :: _} (l ~~> rest) input = do
  (l', mid) <- applyVarBatchAny l input
  (rest', out) <- forwardVarBatch rest mid
  pure (l' ~~> rest', out)


----------------------------------------------------------------------
-- Lightweight forward tracer + activation dump
----------------------------------------------------------------------
--
-- `forwardVarTraced` is gated on `IDRISML_LOG_LEVEL`:
--
--   * < DEBUG  → fast pass-through to `forwardVar` (no summarise, no
--                allocation, no IO overhead). Default user-facing build.
--   * DEBUG    → per-layer min/max/mean/NaN summary via `logDebug`. The
--                "where did the NaN come from?" debug workflow.
--   * TRACE    → DEBUG output PLUS each per-layer activation tensor is
--                registered under a synthetic paramId `__act/<label>/<i>`,
--                flushed to a SafeTensors file at
--                `${IDRISML_ACTIVATION_DIR:-./activations}/<label>-<seq>.safetensors`,
--                then the synthetic entries are erased via
--                `primParamEraseByPrefix` BEFORE `backward()` so the
--                optimizer's full-registry walk never sees them.
--                Frequency throttled by `IDRISML_ACTIVATION_EVERY_N`
--                (default `1` — every forward).
--
-- The TRACE level requires the dylib to be built with
-- `IDRISML_LOG=trace make backend install`; the build ceiling clamps
-- the runtime cap (`log.c:30`).

-- Chez `top-level-value` counter; the world-arg keeps Idris-Chez from
-- CSE-collapsing the body into a load-time constant (see the
-- "Zero-arg %noinline defs are constants" gotcha).
%foreign "scheme:(lambda (w) (when (not (top-level-bound? 'idrisml-act-seq)) (set-top-level-value! 'idrisml-act-seq 0)) (set-top-level-value! 'idrisml-act-seq (+ (top-level-value 'idrisml-act-seq) 1)) (top-level-value 'idrisml-act-seq))"
prim__nextActivationSeq : PrimIO Int

nextActivationSeq : IO Int
nextActivationSeq = primIO prim__nextActivationSeq

activationDir : IO String
activationDir = do
  mv <- getEnv "IDRISML_ACTIVATION_DIR"
  pure (fromMaybe "./activations" mv)

activationEveryN : IO Int
activationEveryN = do
  mv <- getEnv "IDRISML_ACTIVATION_EVERY_N"
  case mv of
    Nothing => pure 1
    Just s => case parseInteger {a=Int} s of
                Just n => if n <= 0 then pure 1 else pure n
                Nothing => pure 1

-- Replace label chars that aren't safe for a path segment. Keep
-- alnum + dash + underscore; collapse everything else to '_'. The
-- label is user-supplied (e.g. "epoch5") so this is defensive
-- against accidental slashes / spaces / dots.
sanitizeLabel : String -> String
sanitizeLabel s = pack (map sanitize (unpack s))
  where
    sanitize : Char -> Char
    sanitize c =
      if isAlphaNum c || c == '-' || c == '_' then c else '_'

||| Walks the Network like `forwardVar`. Behavior gated on
||| `IDRISML_LOG_LEVEL` — see the module-level comment block.
|||
||| The autograd graph is preserved across all branches — this just
||| adds side-effecting reads (DEBUG) and registry round-trips (TRACE)
||| between layer applications. The returned Tensor is the same one a
||| plain `forwardVar` would produce.
|||
||| Usage: swap `forwardVar` for `forwardVarTraced "epoch5"` at any
||| call site. The `label` becomes the SafeTensors filename stem and
||| the per-layer paramId prefix.
|||
||| DEBUG-level stderr lines look like:
|||
|||     epoch5:0 min=-0.123 max=0.456 mean=0.012
|||     epoch5:1 min=-0.234 max=0.567 mean=0.099
|||     epoch5:out min=-0.300 max=0.700 mean=0.150  [NaN]
|||
||| TRACE-level SafeTensors files contain keys
||| `__act/epoch5/0`, `__act/epoch5/1`, ... readable in Python via
||| `safetensors.numpy.load_file(path)`.
export
forwardVarTraced : {0 ex : Executor} -> UserExecutorTraining ex => UserExecutorCore ex => RuntimeDType dt => Linked ex => Compatible ex dt => {0 g : GradMode} -> {i, o : Nat} -> {hs : List Nat} ->
                   (label : String) ->
                   Network i hs o ex dt g -> Tensor [i] ex dt g ->
                   IO (Network i hs o ex dt g, Tensor [o] ex dt g)
forwardVarTraced label net input = do
  lvl <- getLogLevel
  if lvl < levelDebug
    then forwardVar net input
    else do
      -- TRACE branch decides up front whether THIS call dumps. The DEBUG
      -- branch (and TRACE-but-throttled-out) still emits the summary.
      everyN <- if lvl >= levelTrace then activationEveryN else pure 1
      seq    <- if lvl >= levelTrace then nextActivationSeq else pure 0
      let dump = lvl >= levelTrace && mod seq everyN == 0
      (net', out, names) <- go dump 0 [] net input
      when dump $ flushActivations seq names
      pure (net', out)
  where
    summarize : (idxLabel : String) -> AnyPtr -> IO ()
    summarize idxLabel ptr = do
      let mn = primItem {ex} (primTensorMin {ex} ptr)
          mx = primItem {ex} (primTensorMax {ex} ptr)
          me = primItem {ex} (primMean {ex} ptr)
          isNaN : Double -> Bool
          isNaN x = x /= x
          tag = if isNaN mn || isNaN mx || isNaN me then "  [NaN]" else ""
      logDebug $
        label ++ ":" ++ idxLabel
          ++ " min=" ++ show mn
          ++ " max=" ++ show mx
          ++ " mean=" ++ show me ++ tag

    -- Synthetic paramId for the i-th layer's activation in this call.
    actName : Nat -> String
    actName idx = "__act/" ++ sanitizeLabel label ++ "/" ++ show idx

    -- Register the activation under its synthetic name. `ioRerun`
    -- forces evaluation of the pure-typed `primParamRegister` FFI
    -- (see `feedback_pure_typed_ffi_reorders`).
    registerAct : Nat -> AnyPtr -> IO ()
    registerAct idx ptr =
      ignore $ ioRerun (\_ => primParamRegister {ex} (actName idx) ptr)

    flushActivations : Int -> List String -> IO ()
    flushActivations seq names = do
      dir <- activationDir
      _   <- createDir dir   -- ignores "already exists"
      let path = dir ++ "/" ++ sanitizeLabel label ++ "-" ++ show seq ++ ".safetensors"
          namesNl = unlines names
      _ <- primIO (primParamSaveByName {ex} path namesNl (cast (length names)))
      primIO (primParamEraseByPrefix {ex} ("__act/" ++ sanitizeLabel label ++ "/"))

    -- `dump` controls whether to register per-layer activations. The
    -- accumulator `names` collects the synthetic paramIds for the
    -- flush step. Names are pushed in walk order (idx 0 first) so the
    -- on-disk safetensors header lists layers in network order.
    go : {0 ex : Executor} -> UserExecutorTraining ex => Linked ex => Compatible ex dt => {0 g : GradMode} -> {i, o : Nat} -> {hs : List Nat} ->
         (dump : Bool) -> Nat -> List String ->
         Network i hs o ex dt g -> Tensor [i] ex dt g ->
         IO (Network i hs o ex dt g, Tensor [o] ex dt g, List String)
    go dump idx names (OutputLayer l) inp = do
      (l', out) <- applyVarAny l inp
      summarize (show idx ++ "(out)") out.tensorPtr
      when dump $ registerAct idx out.tensorPtr
      pure (OutputLayer l', out, names ++ [actName idx])
    go {hs = h :: _} dump idx names (l ~~> rest) inp = do
      (l', mid) <- applyVarAny l inp
      summarize (show idx) mid.tensorPtr
      when dump $ registerAct idx mid.tensorPtr
      (rest', out, names') <- go dump (idx + 1) (names ++ [actName idx]) rest mid
      pure (l' ~~> rest', out, names')
