||| Pluggable-Device interface. Phase 2.1 of the refactor; see
||| `docs/develop/design-decisions.md` "Pluggable Device via sliced
||| `UserDevice` interfaces" for the design.
|||
||| Phase 2.1 ships the **lifecycle + arithmetic** slice (~20 ops).
||| Later slices (2.2-2.5) extend with linear / NN / conv / tape ops.
|||
||| Users implementing their own backend declare an empty type and
||| an instance:
|||
|||   data MyDev : Type where MD : MyDev
|||
|||   UserDeviceCore MyDev where
|||     primAdd = prim__addMine
|||     ...
|||
||| The built-in `TapeDev` / `TorchDev` / `MlxDev` (in `Device.Tape`,
||| `Device.Torch`, `Device.Mlx`) forward to the per-backend C
||| symbols emitted by Phase 1's rename headers
||| (`tensor_add_tape` / `tensor_add_torch` / `tensor_add_mlx`).
module Device.Core


----------------------------------------------------------------------
-- `Device` kind alias
--
-- `Device` is a 0-quantity alias for `Type`. Tensor's `d` phantom is
-- declared as `(0 d : Device)`, which is exactly `(0 d : Type)`
-- underneath but reads as "d is a device tag" at every kind-binder
-- site. No type-system enforcement: nothing stops a caller writing
-- `Tensor [4] Bool`. But construction (`primCreate*`) and operations
-- (`tadd` etc.) both require `UserDeviceCore d =>`, so non-device
-- `d`s can be declared but never inhabited or operated on.
--
-- See `docs/develop/design-decisions.md` "Open `d` kind: why
-- `Device = Type` instead of a real sub-kind" for the alternatives
-- considered and why we kept it open.
----------------------------------------------------------------------

public export
0 Device : Type
Device = Type


----------------------------------------------------------------------
-- UserDeviceCore — lifecycle + arithmetic slice
----------------------------------------------------------------------

||| Phase 2.1 interface: the ~20 ops needed for tensor lifecycle and
||| elementwise arithmetic. Later phases add `UserDeviceLinear`,
||| `UserDeviceNN`, `UserDeviceConv`, `UserDeviceTape` slices.
public export
interface UserDeviceCore (0 d : Device) where
  ||| Human-readable device tag: "tape", "torch", "mlx", "mybackend".
  ||| Used in logs and `Show Device`-style stringification.
  deviceName : String

  -- Lifecycle ---------------------------------------------------------
  ||| Allocate a 0-rank tensor with the given value and grad flag.
  ||| `requires_grad` is 0 or 1.
  primCreateScalar : Double -> Int -> AnyPtr

  ||| Allocate a multi-dimensional tensor. Args: data buffer, shape
  ||| buffer, rank, requires_grad.
  primCreate : AnyPtr -> AnyPtr -> Int -> Int -> AnyPtr

  ||| Release a tensor handle. No-op on backends that GC.
  primFree : AnyPtr -> ()

  ||| Read a 0-rank tensor's value.
  primItem : AnyPtr -> Double

  ||| Deep-copy a tensor (new handle, same shape and values, fresh
  ||| autograd node).
  primClone : AnyPtr -> AnyPtr

  -- Elementwise arithmetic --------------------------------------------
  primAdd     : AnyPtr -> AnyPtr -> AnyPtr
  primSub     : AnyPtr -> AnyPtr -> AnyPtr
  primMul     : AnyPtr -> AnyPtr -> AnyPtr
  primDiv     : AnyPtr -> AnyPtr -> AnyPtr
  primNeg     : AnyPtr -> AnyPtr
  primAbs     : AnyPtr -> AnyPtr
  primExp     : AnyPtr -> AnyPtr
  primLog     : AnyPtr -> AnyPtr
  primSqrt    : AnyPtr -> AnyPtr
  primPow     : AnyPtr -> AnyPtr -> AnyPtr
  primSigmoid : AnyPtr -> AnyPtr
  primTanh    : AnyPtr -> AnyPtr

  -- Scalar arithmetic -------------------------------------------------
  primAddScalar : AnyPtr -> Double -> AnyPtr
  primMulScalar : AnyPtr -> Double -> AnyPtr
  primClampMin  : AnyPtr -> Double -> AnyPtr


----------------------------------------------------------------------
-- UserDeviceLinear — matmul + reductions + reshape + indexing slice
----------------------------------------------------------------------

||| The second slice. Covers linear algebra (mv, matmul, linear,
||| dot, outer, bmm), reductions (sum, mean, min, max, sumDim),
||| reshape / select (view, reshape, select, unsqueeze, squeeze,
||| stack, narrow, transpose), indexing (gather, scatter_add), and
||| sort/scan (argsort, cumprod). ~30 ops.
|||
||| Subclass of `UserDeviceCore`: an implementer also provides
||| lifecycle + arithmetic ops, so a single `UserDeviceLinear d =>`
||| constraint in scope is enough to use both slices' methods. The
||| convention scales as later slices (`UserDeviceNN`, `Conv`,
||| `Tape`) layer on top.
public export
interface UserDeviceCore d => UserDeviceLinear (0 d : Device) where
  -- Linear algebra ----------------------------------------------------
  primMv          : AnyPtr -> AnyPtr -> AnyPtr
  primMatmul      : AnyPtr -> AnyPtr -> AnyPtr
  primLinear      : AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr
  primDot         : AnyPtr -> AnyPtr -> AnyPtr
  primOuter       : AnyPtr -> AnyPtr -> AnyPtr
  primBmm         : AnyPtr -> AnyPtr -> AnyPtr
  primLinear2d    : AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr

  -- Reductions --------------------------------------------------------
  primSum         : AnyPtr -> AnyPtr
  primMean        : AnyPtr -> AnyPtr
  primTensorMin   : AnyPtr -> AnyPtr
  primTensorMax   : AnyPtr -> AnyPtr
  primSumDim      : AnyPtr -> Int -> Int -> AnyPtr

  -- Shape / view / reshape -------------------------------------------
  primSelect      : AnyPtr -> Int -> Int -> AnyPtr
  primUnsqueeze   : AnyPtr -> Int -> AnyPtr
  primSqueeze     : AnyPtr -> Int -> AnyPtr
  primStack       : AnyPtr -> Int -> Int -> AnyPtr
  primView1d      : AnyPtr -> Int -> AnyPtr
  primView2d      : AnyPtr -> Int -> Int -> AnyPtr
  primReshape2d   : AnyPtr -> Int -> Int -> AnyPtr
  primReshape3d   : AnyPtr -> Int -> Int -> Int -> AnyPtr
  primReshape4d   : AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr
  primNarrow      : AnyPtr -> Int -> Int -> Int -> AnyPtr
  primTransposeLast2 : AnyPtr -> AnyPtr
  primTranspose2d : AnyPtr -> AnyPtr

  -- Concatenation -----------------------------------------------------
  primCat         : AnyPtr -> Int -> Int -> AnyPtr
  primCat2        : AnyPtr -> AnyPtr -> AnyPtr
  primConcat2dAxis1 : AnyPtr -> AnyPtr -> AnyPtr

  -- Indexing ----------------------------------------------------------
  primGather      : AnyPtr -> AnyPtr -> Int -> AnyPtr
  primScatterAdd  : AnyPtr -> AnyPtr -> Int -> AnyPtr

  -- Sort / scan -------------------------------------------------------
  primArgsort     : AnyPtr -> Int -> Int -> AnyPtr
  primCumprod    : AnyPtr -> Int -> AnyPtr


----------------------------------------------------------------------
-- UserDeviceNN — activations + softmax + norms + losses + recurrent
-- cells + embedding + attention slice
----------------------------------------------------------------------

||| The third slice. Adds non-linearities, normalizations, recurrent
||| cells, embeddings, and the loss surfaces. Subclass of
||| `UserDeviceLinear` (transitively `UserDeviceCore`).
public export
interface UserDeviceLinear d => UserDeviceNN (0 d : Device) where
  -- Activations -------------------------------------------------------
  primGelu        : AnyPtr -> AnyPtr
  primLeakyRelu   : AnyPtr -> Double -> AnyPtr
  primSilu        : AnyPtr -> AnyPtr
  primSoftplus    : AnyPtr -> AnyPtr

  -- Softmax family ----------------------------------------------------
  primSoftmax     : AnyPtr -> Int -> AnyPtr
  primLogSoftmax  : AnyPtr -> Int -> AnyPtr
  primSoftmax2d   : AnyPtr -> AnyPtr
  primLogSoftmax2d : AnyPtr -> AnyPtr
  primSoftmax3d   : AnyPtr -> AnyPtr

  -- Masking -----------------------------------------------------------
  primMaskedFill  : AnyPtr -> AnyPtr -> Double -> AnyPtr
  primExpandMask  : AnyPtr -> Int -> AnyPtr
  primCausalMask  : Int -> AnyPtr

  -- Norms / dropout ---------------------------------------------------
  primLayerNorm2d : AnyPtr -> AnyPtr -> AnyPtr -> Double -> AnyPtr
  primBatchNorm   : AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr ->
                     Int -> Int -> Int -> Double -> Double -> AnyPtr
  primDropout     : AnyPtr -> Double -> Int -> Int -> AnyPtr

  -- Embedding / similarity / attention -------------------------------
  primEmbedding      : AnyPtr -> AnyPtr -> Int -> Int -> AnyPtr
  primCosineSimilarity : AnyPtr -> AnyPtr -> Int -> AnyPtr
  primCrossAttention : AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr -> Double -> AnyPtr

  -- Loss --------------------------------------------------------------
  primBceWithLogits : AnyPtr -> AnyPtr -> AnyPtr

  -- Recurrent cells ---------------------------------------------------
  primGruCell        : AnyPtr -> AnyPtr -> AnyPtr -> Int -> AnyPtr
  primLstmGatesPair  : AnyPtr -> AnyPtr -> Int -> AnyPtr
  primPairFirst      : AnyPtr -> AnyPtr
  primPairSecond     : AnyPtr -> AnyPtr


----------------------------------------------------------------------
-- UserDeviceConv — convolution + pooling slice
----------------------------------------------------------------------

||| The fourth slice. Covers 1D and 2D convolution + pooling (~9
||| ops). Subclass of `UserDeviceNN` (transitively Linear + Core).
public export
interface UserDeviceNN d => UserDeviceConv (0 d : Device) where
  -- 1D conv + pool
  primConv1d         : AnyPtr -> AnyPtr -> AnyPtr -> Int -> Int -> AnyPtr
  primConv1dCircular : AnyPtr -> AnyPtr -> AnyPtr
  primAvgPool1d      : AnyPtr -> Int -> Int -> AnyPtr
  primMaxPool1d      : AnyPtr -> Int -> Int -> AnyPtr
  -- 2D conv + pool
  primConv2d         : AnyPtr -> AnyPtr -> AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr
  primConv2dBatched  : AnyPtr -> AnyPtr -> AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr
  primAvgPool2d      : AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr
  primMaxPool2d      : AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr
  primMaxPool2dBatched : AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr


----------------------------------------------------------------------
-- UserDeviceTape — autograd + param registry + IO + misc slice
----------------------------------------------------------------------

||| The fifth and final slice. Closes the chain
||| `Core <- Linear <- NN <- Conv <- Tape`. Covers autograd
||| (requiresGrad, noGradBegin/End, detach, withGrad), the param
||| registry that the optimizer reads from, param + state allocation,
||| and small IO helpers (allocDoubles, readDouble, writeDouble,
||| print, seq).
public export
interface UserDeviceConv d => UserDeviceTape (0 d : Device) where
  -- Autograd flag --------------------------------------------------
  primRequiresGrad      : AnyPtr -> Int
  primSetRequiresGrad   : AnyPtr -> Int -> PrimIO ()
  primNoGradBegin       : PrimIO ()
  primNoGradEnd         : PrimIO ()
  primDetach            : AnyPtr -> AnyPtr
  primWithGrad          : AnyPtr -> AnyPtr

  -- Shape / info queries -------------------------------------------
  primTensorDim         : AnyPtr -> Int
  primTensorSizeAt      : AnyPtr -> Int -> Int

  -- Param registry (optimizer-side) --------------------------------
  --
  -- Note: `primParamClear` is intentionally `PrimIO ()` (not bare
  -- `()`). A zero-arg method of unit type gets eagerly evaluated at
  -- instance-dictionary construction time and would silently call
  -- `param_clear` every time `UserDeviceTape d =>` is brought into
  -- scope — wiping the param registry mid-training. `PrimIO ()` is a
  -- thunk; the side effect only fires when `primIO` runs it.
  primParamRegister     : String -> AnyPtr -> AnyPtr
  primParamClear        : PrimIO ()
  primParamCount        : () -> Int
  primParamName         : Int -> String
  primParamGradItem     : Int -> Double
  primParamGradItemAt   : Int -> Int -> Double
  primParamGradItemAndZero : Int -> Double
  primParamZeroAllGrads : Int -> Int
  primParamSubtractDelta : Int -> Double -> ()

  -- Param + state creation -----------------------------------------
  primCreateParam1d     : Int -> AnyPtr -> AnyPtr
  primCreateParam2d     : Int -> Int -> AnyPtr -> AnyPtr
  primCreateParam3d     : Int -> Int -> Int -> AnyPtr -> AnyPtr
  primCreateState1d     : Int -> AnyPtr -> AnyPtr
  primCreateState2d     : Int -> Int -> AnyPtr -> AnyPtr

  -- Doubles array helpers ------------------------------------------
  primAllocDoubles      : Int -> AnyPtr
  primReadDouble        : AnyPtr -> Int -> Double
  primWriteDouble       : AnyPtr -> Int -> Double -> ()

  -- Misc -----------------------------------------------------------
  primPrint             : AnyPtr -> ()
