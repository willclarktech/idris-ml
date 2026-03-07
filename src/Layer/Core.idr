module Layer.Core

import Data.List
import Data.SortedMap
import Data.Vect
import Data.Zippable

import DataPoint
import Endofunctor
import Floating
import Math
import Tensor
import Util
import Variable


----------------------------------------------------------------------
-- Debug Types (here so the interface can reference them)
----------------------------------------------------------------------

||| A debug snapshot for one layer at one timestep
public export
record DebugEntry where
  constructor MkDebugEntry
  layerName : String
  fields : List (String, String)

||| Per-timestep snapshot of the entire network
public export
DebugSnapshot : Type
DebugSnapshot = List DebugEntry


----------------------------------------------------------------------
-- LayerLike Interface
----------------------------------------------------------------------

public export
interface LayerLike (l : Nat -> Nat -> Type -> Type) where
  -- Forward pass (generic, for Double-based evaluation/debug)
  applyGeneric : (FromDouble ty, Floating ty, Fractional ty, Neg ty, Num ty, Ord ty) =>
                 {i, o : Nat} -> l i o ty -> Vector i ty -> (l i o ty, Vector o ty)

  -- Forward pass (Variable-specialized, for training with C-backed ops)
  applyVar : {i, o : Nat} -> l i o Variable -> Vector i Variable -> (l i o Variable, Vector o Variable)

  -- Type-preserving map (for applyDeltas in non-dense path)
  emapLayer : {i, o : Nat} -> (ty -> ty) -> l i o ty -> l i o ty

  -- Display
  showLayer : {i, o : Nat} -> Show ty => l i o ty -> String

  -- Parameter naming (assigns paramIds for gradient collection)
  nameLayer : {i, o : Nat} -> String -> l i o Variable -> l i o Variable

  -- Auto-naming prefix (e.g., "ll" for linear, "lstm" for LSTM)
  layerPrefix : {i, o : Nat} -> l i o ty -> String
  layerPrefix _ = ""

  -- Convert Variable-typed layer to Double (for evaluation/debug)
  toDoubleLayer : {i, o : Nat} -> l i o Variable -> l i o Double

  -- Debug forward: run applyGeneric + capture internal state snapshot
  debugApply : {i, o : Nat} -> l i o Double -> Vector i Double -> (l i o Double, Vector o Double, DebugEntry)

  -- Buffer sync after optimizer deltas (default: identity)
  syncBuffers : {i, o : Nat} -> l i o Variable -> l i o Variable
  syncBuffers x = x

  -- Direct delta application to C buffers (dense optimizer path)
  applyDeltasAndSync : {i, o : Nat} -> AnyPtr -> l i o Variable -> l i o Variable
  applyDeltasAndSync _ x = x

  -- Read C buffer values back to Variable records
  readFromBuffers : {i, o : Nat} -> l i o Variable -> l i o Variable
  readFromBuffers x = x

  -- Reset per-sequence state (NTM memory reset, default: identity)
  resetState : {i, o : Nat} -> l i o Variable -> l i o Variable
  resetState x = x

  -- Get all parameter IDs (for testing)
  getParamIds : {i, o : Nat} -> l i o Variable -> List String
  getParamIds _ = []


----------------------------------------------------------------------
-- AnyLayer (Existential Wrapper)
----------------------------------------------------------------------

||| Existential: hides the concrete layer type behind the interface.
||| The type constructor `l` is stored as a non-erased parameter so it
||| remains accessible for interface dispatch after pattern matching.
public export
data AnyLayer : Nat -> Nat -> Type -> Type where
  MkAnyLayer : (l : Nat -> Nat -> Type -> Type) -> LayerLike l => l i o ty -> AnyLayer i o ty


----------------------------------------------------------------------
-- AnyLayer Dispatch Helpers
----------------------------------------------------------------------

export
applyGenericAny : (FromDouble ty, Floating ty, Fractional ty, Neg ty, Num ty, Ord ty) =>
                  {i, o : Nat} -> AnyLayer i o ty -> Vector i ty -> (AnyLayer i o ty, Vector o ty)
applyGenericAny (MkAnyLayer l @{dict} layer) xs =
  case applyGeneric @{dict} layer xs of
    (layer', out) => (MkAnyLayer l @{dict} layer', out)

export
applyVarAny : {i, o : Nat} -> AnyLayer i o Variable -> Vector i Variable -> (AnyLayer i o Variable, Vector o Variable)
applyVarAny (MkAnyLayer l @{dict} layer) xs =
  case applyVar @{dict} layer xs of
    (layer', out) => (MkAnyLayer l @{dict} layer', out)

public export
{i, o : Nat} -> Endofunctor (AnyLayer i o) where
  emap f (MkAnyLayer l @{dict} layer) = MkAnyLayer l @{dict} (emapLayer @{dict} f layer)

public export
{i, o : Nat} -> Show ty => Show (AnyLayer i o ty) where
  show (MkAnyLayer _ @{dict} layer) = showLayer @{dict} layer


----------------------------------------------------------------------
-- Network Type
----------------------------------------------------------------------

public export
data Network : (inputDims : Nat) -> (hiddenDims : List Nat) -> (outputDims : Nat) -> Type -> Type where
  OutputLayer : AnyLayer i o ty -> Network i [] o ty
  (~>) : AnyLayer i h ty -> Network h hs o ty -> Network i (h :: hs) o ty

export infixr 5 ~>

public export
implementation {i, o : Nat} -> Show ty => Show (Network i [] o ty) where
  show (OutputLayer layer) = show layer

public export
implementation {i, h : Nat} -> (Show ty, Show (Network h hs o ty)) => Show (Network i (h :: hs) o ty) where
  show (layer ~> layers) = show layer ++ " ~> " ++ show layers

emapNetwork : {i, o : Nat} -> {hs : List Nat} -> (ty -> ty) -> Network i hs o ty -> Network i hs o ty
emapNetwork f (OutputLayer l) = OutputLayer (emap f l)
emapNetwork {hs = h :: _} f (l ~> rest) = emap f l ~> emapNetwork f rest

public export
{i, o : Nat} -> {hs : List Nat} -> Endofunctor (Network i hs o) where
  emap = emapNetwork


----------------------------------------------------------------------
-- Generic Forward Pass
----------------------------------------------------------------------

export
forward : (FromDouble ty, Floating ty, Fractional ty, Neg ty, Num ty, Ord ty) =>
          {i, o : Nat} -> {hs : List Nat} -> Network i hs o ty -> Vector i ty ->
          (Network i hs o ty, Vector o ty)
forward (OutputLayer l) x =
  case applyGenericAny l x of
    (l', output) => (OutputLayer l', output)
forward {hs = h :: _} (l ~> layers) x =
  case applyGenericAny l x of
    (l', layerOutput) =>
      case forward layers layerOutput of
        (rest', networkOutput) => (l' ~> rest', networkOutput)


----------------------------------------------------------------------
-- Variable-Specialized Forward Pass
----------------------------------------------------------------------

export
forwardVar : {i, o : Nat} -> {hs : List Nat} ->
             Network i hs o Variable -> Vector i Variable ->
             (Network i hs o Variable, Vector o Variable)
forwardVar (OutputLayer l) x =
  case applyVarAny l x of
    (l', output) => (OutputLayer l', output)
forwardVar {hs = h :: _} (l ~> layers) x =
  case applyVarAny l x of
    (l', layerOutput) =>
      case forwardVar layers layerOutput of
        (rest', networkOutput) => (l' ~> rest', networkOutput)


----------------------------------------------------------------------
-- Supervised Loss / Evaluation
----------------------------------------------------------------------

forwardNext : (FromDouble ty, Floating ty, Fractional ty, Neg ty, Num ty, Ord ty) =>
              {i, o : Nat} -> {hs : List Nat} ->
              (Network i hs o ty, Vect n (Vector o ty)) -> Vector i ty ->
              (Network i hs o ty, Vect (S n) (Vector o ty))
forwardNext (nn, outputs) inp =
  let (updatedModel, newOutput) = forward nn inp
  in (updatedModel, snoc outputs newOutput)

forwardMany : (FromDouble ty, Floating ty, Fractional ty, Neg ty, Num ty, Ord ty) =>
              {i, o : Nat} -> {hs : List Nat} ->
              Network i hs o ty -> Vect n (Vector i ty) ->
              (Network i hs o ty, Vect n (Vector o ty))
forwardMany network xs =
  foldlD (\k => (Network i hs o ty, Vect k (Vector o ty))) forwardNext (network, []) xs

export
calculateLoss : (FromDouble ty, Floating ty, Fractional ty, Neg ty, Num ty, Ord ty) =>
                {i, o, n : Nat} -> {hs : List Nat} ->
                LossFunction ty -> Network i hs o ty -> Vect n (DataPoint i o ty) -> ty
calculateLoss lossFn model dataPoints =
  let xs = map x dataPoints
      ys = map y dataPoints
      (_, predictions) = forwardMany model xs
      losses = zipWith lossFn predictions ys
  in mean $ VTensor $ map STensor losses

evaluateSingleDataPoint : (FromDouble ty, Floating ty, Fractional ty, Neg ty, Num ty, Ord ty) =>
                          {i, o : Nat} -> {hs : List Nat} ->
                          Network i hs o ty -> DataPoint i o ty -> Vector o ty
evaluateSingleDataPoint model = snd . (forward model) . x

export
evaluate : (FromDouble ty, Floating ty, Fractional ty, Neg ty, Num ty, Ord ty) =>
           {i, o : Nat} -> {hs : List Nat} ->
           Network i hs o ty -> Vect n (DataPoint i o ty) -> Vect n (Vector o ty)
evaluate model = map (evaluateSingleDataPoint model)


----------------------------------------------------------------------
-- Variable Supervised Loss
----------------------------------------------------------------------

forwardNextVar : {i, o : Nat} -> {hs : List Nat} ->
                 (Network i hs o Variable, Vect n (Vector o Variable)) -> Vector i Variable ->
                 (Network i hs o Variable, Vect (S n) (Vector o Variable))
forwardNextVar (nn, outputs) inp =
  let (updatedModel, newOutput) = forwardVar nn inp
  in (updatedModel, snoc outputs newOutput)

forwardManyVar : {i, o : Nat} -> {hs : List Nat} ->
                 Network i hs o Variable -> Vect n (Vector i Variable) ->
                 (Network i hs o Variable, Vect n (Vector o Variable))
forwardManyVar network xs =
  foldlD (\k => (Network i hs o Variable, Vect k (Vector o Variable))) forwardNextVar (network, []) xs

export
calculateLossVar : {i, o, n : Nat} -> {hs : List Nat} ->
                   LossFunction Variable -> Network i hs o Variable ->
                   Vect n (DataPoint i o Variable) -> Variable
calculateLossVar lossFn model dataPoints =
  let xs = map x dataPoints
      ys = map y dataPoints
      (_, predictions) = forwardManyVar model xs
      losses = zipWith lossFn predictions ys
  in mean $ VTensor $ map STensor losses


----------------------------------------------------------------------
-- Recurrent Forward / Loss / Evaluation
----------------------------------------------------------------------

recur : (FromDouble ty, Floating ty, Fractional ty, Neg ty, Num ty, Ord ty) =>
        {i, o : Nat} -> {hs : List Nat} ->
        (Network i hs o ty, List (Vector o ty)) -> Vector i ty ->
        (Network i hs o ty, List (Vector o ty))
recur (m, os) i =
  let (updatedModel, output) = forward m i
  in (updatedModel, snoc os output)

export
forwardRecurrent : (FromDouble ty, Floating ty, Fractional ty, Neg ty, Num ty, Ord ty) =>
                   {i, o : Nat} -> {hs : List Nat} ->
                   Network i hs o ty -> List (Vector i ty) ->
                   (Network i hs o ty, List (Vector o ty))
forwardRecurrent model = foldl recur (model, [])

export
calculateLossRecurrent : (FromDouble ty, Floating ty, Fractional ty, Neg ty, Num ty, Ord ty) =>
                         {i, o, n : Nat} -> {hs : List Nat} ->
                         LossFunction ty -> Network i hs o ty ->
                         Vect n (RecurrentDataPoint i o ty) -> ty
calculateLossRecurrent lossFn model dataPoints =
  let perSequence : RecurrentDataPoint i o ty -> List ty
      perSequence dp = let (_, preds) = forwardRecurrent model (xs dp)
                       in zipWith lossFn preds (ys dp)
      losses = map perSequence dataPoints
  in mean . VTensor $ map (STensor . mean) losses

evaluateSingleRecurrentDataPoint : (FromDouble ty, Floating ty, Fractional ty, Neg ty, Num ty, Ord ty) =>
                                   {i, o : Nat} -> {hs : List Nat} ->
                                   Network i hs o ty -> RecurrentDataPoint i o ty -> List (Vector o ty)
evaluateSingleRecurrentDataPoint model dataPoints = snd $ (forwardRecurrent model) dataPoints.xs

export
evaluateRecurrent : (FromDouble ty, Floating ty, Fractional ty, Neg ty, Num ty, Ord ty) =>
                    {i, o : Nat} -> {hs : List Nat} ->
                    Network i hs o ty -> Vect n (RecurrentDataPoint i o ty) ->
                    Vect n (List (Vector o ty))
evaluateRecurrent model dataPoints = map (evaluateSingleRecurrentDataPoint model) dataPoints


----------------------------------------------------------------------
-- Variable Recurrent Forward / Loss
----------------------------------------------------------------------

recurVar : {i, o : Nat} -> {hs : List Nat} ->
           (Network i hs o Variable, List (Vector o Variable)) -> Vector i Variable ->
           (Network i hs o Variable, List (Vector o Variable))
recurVar (m, os) inp =
  let (updatedModel, output) = forwardVar m inp
  in (updatedModel, snoc os output)

export
forwardRecurrentVar : {i, o : Nat} -> {hs : List Nat} ->
                      Network i hs o Variable -> List (Vector i Variable) ->
                      (Network i hs o Variable, List (Vector o Variable))
forwardRecurrentVar model = foldl recurVar (model, [])

export
calculateLossRecurrentVar : {i, o, n : Nat} -> {hs : List Nat} ->
                            LossFunction Variable -> Network i hs o Variable ->
                            Vect n (RecurrentDataPoint i o Variable) -> Variable
calculateLossRecurrentVar lossFn model dataPoints =
  let perSequence : RecurrentDataPoint i o Variable -> List Variable
      perSequence dp = let (_, preds) = forwardRecurrentVar model (xs dp)
                       in zipWith lossFn preds (ys dp)
      losses = map perSequence dataPoints
  in mean . VTensor $ map (STensor . mean) losses


----------------------------------------------------------------------
-- Two-Phase Forward / Loss
----------------------------------------------------------------------

||| Reset per-sequence state for all layers in the network.
export
resetNetworkState : {i, o : Nat} -> {hs : List Nat} -> Network i hs o Variable -> Network i hs o Variable
resetNetworkState (OutputLayer (MkAnyLayer l @{dict} layer)) = OutputLayer (MkAnyLayer l @{dict} (resetState @{dict} layer))
resetNetworkState ((MkAnyLayer l @{dict} layer) ~> rest) = MkAnyLayer l @{dict} (resetState @{dict} layer) ~> resetNetworkState rest

||| Two-phase forward: encoding phase then output phase with zeros.
export
forwardTwoPhase : (FromDouble ty, Floating ty, Fractional ty, Neg ty, Num ty, Ord ty) =>
                  {i, o : Nat} -> {hs : List Nat} ->
                  Network i hs o ty -> TwoPhaseDataPoint i o ty ->
                  (Network i hs o ty, List (Vector o ty))
forwardTwoPhase model dp =
  let encResult = forwardRecurrent model (encodingInputs dp)
      zeroInput : Vector i ty
      zeroInput = zeros
      outputInputs = Data.List.replicate (length (targets dp)) zeroInput
  in forwardRecurrent (fst encResult) outputInputs

||| Two-phase loss: encoding phase (discard outputs), then output phase
||| (feed zeros, compute loss on collected outputs vs targets).
export
calculateLossTwoPhaseVar : {i, o, n : Nat} -> {hs : List Nat} ->
                           LossFunction Variable -> Network i hs o Variable ->
                           Vect n (TwoPhaseDataPoint i o Variable) -> Variable
calculateLossTwoPhaseVar lossFn model dataPoints =
  let perSequence : TwoPhaseDataPoint i o Variable -> List Variable
      perSequence dp =
        let zeroInput : Vector i Variable
            zeroInput = map (const (fromDouble 0.0)) zeros
            outputInputs = Data.List.replicate (length (targets dp)) zeroInput
            model' = resetNetworkState model
            encResult = forwardRecurrentVar model' (encodingInputs dp)
            outResult = forwardRecurrentVar (fst encResult) outputInputs
        in zipWith lossFn (snd outResult) (targets dp)
      losses = map perSequence dataPoints
  in mean . VTensor $ map (STensor . mean) losses

||| Two-phase loss with C-backed BCE: encoding phase (discard outputs),
||| then output phase (feed zeros, compute fused BCE loss).
export
calculateLossTwoPhaseVarBce : {i, o, n : Nat} -> {hs : List Nat} ->
                              Network i hs o Variable ->
                              Vect n (TwoPhaseDataPoint i o Variable) -> Variable
calculateLossTwoPhaseVarBce model dataPoints =
  let perSequence : TwoPhaseDataPoint i o Variable -> List Variable
      perSequence dp =
        let zeroInput : Vector i Variable
            zeroInput = map (const (fromDouble 0.0)) zeros
            outputInputs = Data.List.replicate (length (targets dp)) zeroInput
            model' = resetNetworkState model
            encResult = forwardRecurrentVar model' (encodingInputs dp)
            outResult = forwardRecurrentVar (fst encResult) outputInputs
        in zipWith bceWithLogitsVar (snd outResult) (targets dp)
      losses = map perSequence dataPoints
  in mean . VTensor $ map (STensor . mean) losses


----------------------------------------------------------------------
-- Automatic Parameter Naming
----------------------------------------------------------------------

autoNameAny : {i, o : Nat} -> String -> SortedMap String Nat -> AnyLayer i o Variable ->
              (SortedMap String Nat, AnyLayer i o Variable)
autoNameAny scope counts (MkAnyLayer l @{dict} layer) =
  let pfx = layerPrefix @{dict} layer
  in if pfx == "" then (counts, MkAnyLayer l @{dict} layer)
     else let n = fromMaybe 0 (lookup pfx counts)
              counts' = insert pfx (n + 1) counts
              fullName = scope ++ pfx ++ show n
          in (counts', MkAnyLayer l @{dict} (nameLayer @{dict} fullName layer))

autoNameNetwork : String -> SortedMap String Nat ->
                  {i, o : Nat} -> {hs : List Nat} ->
                  Network i hs o Variable ->
                  (SortedMap String Nat, Network i hs o Variable)
autoNameNetwork scope counts (OutputLayer l) =
  let (counts', l') = autoNameAny scope counts l
  in (counts', OutputLayer l')
autoNameNetwork scope counts (l ~> rest) =
  let (counts', l') = autoNameAny scope counts l
      (counts'', rest') = autoNameNetwork scope counts' rest
  in (counts'', l' ~> rest')

||| Automatically name all parameters using type-based prefixes.
export
autoName : {i, o : Nat} -> {hs : List Nat} ->
           Network i hs o Variable -> Network i hs o Variable
autoName net = snd (autoNameNetwork "" empty net)


----------------------------------------------------------------------
-- Network-Level Operations (sync, deltas, readback)
----------------------------------------------------------------------

export
syncNetworkBuffers : {i, o : Nat} -> {hs : List Nat} ->
                     Network i hs o Variable -> Network i hs o Variable
syncNetworkBuffers (OutputLayer (MkAnyLayer l @{dict} layer)) = OutputLayer (MkAnyLayer l @{dict} (syncBuffers @{dict} layer))
syncNetworkBuffers ((MkAnyLayer l @{dict} layer) ~> rest) = MkAnyLayer l @{dict} (syncBuffers @{dict} layer) ~> syncNetworkBuffers rest

export
applyDeltasAndSyncNetwork : {i, o : Nat} -> {hs : List Nat} ->
                            AnyPtr -> Network i hs o Variable -> Network i hs o Variable
applyDeltasAndSyncNetwork deltas (OutputLayer (MkAnyLayer l @{dict} layer)) =
  OutputLayer (MkAnyLayer l @{dict} (applyDeltasAndSync @{dict} deltas layer))
applyDeltasAndSyncNetwork deltas ((MkAnyLayer l @{dict} layer) ~> rest) =
  MkAnyLayer l @{dict} (applyDeltasAndSync @{dict} deltas layer) ~> applyDeltasAndSyncNetwork deltas rest

export
readFromBuffersNetwork : {i, o : Nat} -> {hs : List Nat} ->
                         Network i hs o Variable -> Network i hs o Variable
readFromBuffersNetwork (OutputLayer (MkAnyLayer l @{dict} layer)) = OutputLayer (MkAnyLayer l @{dict} (readFromBuffers @{dict} layer))
readFromBuffersNetwork ((MkAnyLayer l @{dict} layer) ~> rest) =
  MkAnyLayer l @{dict} (readFromBuffers @{dict} layer) ~> readFromBuffersNetwork rest


----------------------------------------------------------------------
-- Variable -> Double Network Conversion
----------------------------------------------------------------------

toDoubleAny : {i, o : Nat} -> AnyLayer i o Variable -> AnyLayer i o Double
toDoubleAny (MkAnyLayer l @{dict} layer) = MkAnyLayer l @{dict} (toDoubleLayer @{dict} layer)

export
toDoubleNetwork : {i, o : Nat} -> {hs : List Nat} ->
                  Network i hs o Variable -> Network i hs o Double
toDoubleNetwork (OutputLayer l) = OutputLayer (toDoubleAny l)
toDoubleNetwork (l ~> rest) = toDoubleAny l ~> toDoubleNetwork rest


----------------------------------------------------------------------
-- Debug Network Forward
----------------------------------------------------------------------

debugApplyAny : {i, o : Nat} -> AnyLayer i o Double -> Vector i Double ->
                (AnyLayer i o Double, Vector o Double, DebugEntry)
debugApplyAny (MkAnyLayer l @{dict} layer) inp =
  case debugApply @{dict} layer inp of
    (layer', out, entry) => (MkAnyLayer l @{dict} layer', out, entry)

||| Walk the network, collecting debug entries from each layer
export
debugForward : {i, o : Nat} -> {hs : List Nat} ->
               Network i hs o Double -> Vector i Double ->
               (Network i hs o Double, Vector o Double, DebugSnapshot)
debugForward (OutputLayer l) x =
  let (l', output, entry) = debugApplyAny l x
  in (OutputLayer l', output, [entry])
debugForward {hs = h :: _} (l ~> layers) x =
  let (l', layerOutput, entry) = debugApplyAny l x
      (rest', networkOutput, entries) = debugForward layers layerOutput
  in (l' ~> rest', networkOutput, entry :: entries)

||| Recurrent: fold over timesteps, collecting per-timestep snapshots
export
debugForwardRecurrent : {i, o : Nat} -> {hs : List Nat} ->
                        Network i hs o Double -> List (Vector i Double) ->
                        (Network i hs o Double, List (Vector o Double), List DebugSnapshot)
debugForwardRecurrent model inputs = foldl step (model, [], []) inputs
  where
    step : (Network i hs o Double, List (Vector o Double), List DebugSnapshot) -> Vector i Double ->
           (Network i hs o Double, List (Vector o Double), List DebugSnapshot)
    step (m, outs, snaps) inp =
      let (m', out, snap) = debugForward m inp
      in (m', outs ++ [out], snaps ++ [snap])


----------------------------------------------------------------------
-- Network Parameter IDs (for testing)
----------------------------------------------------------------------

anyLayerParamIds : {i, o : Nat} -> AnyLayer i o Variable -> List String
anyLayerParamIds (MkAnyLayer _ @{dict} layer) = getParamIds @{dict} layer

export
networkParamIds : {i, o : Nat} -> {hs : List Nat} ->
                  Network i hs o Variable -> List String
networkParamIds (OutputLayer l) = anyLayerParamIds l
networkParamIds (l ~> rest) = anyLayerParamIds l ++ networkParamIds rest
