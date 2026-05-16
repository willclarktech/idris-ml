||| Phase 0b prototype — sliced UserDevice interfaces at realistic
||| width (~150 methods across 5 sub-interfaces), to measure whether
||| Idris-2's instance resolution scales to the full Tensor.idr FFI
||| surface (164 ops today).
|||
||| Slicing matches the Phase 2.1..2.5 staging in the project plan:
||| Core / Linear / NN / Conv / Tape — each slice is its own interface
||| so a backend that doesn't implement (say) Conv simply can't be used
||| with conv layers, and that's a *type error*. This is the "ops
||| depend on device" pitch the refactor exists to deliver.
|||
||| Method signatures here are STUBS — varied shapes (`AnyPtr -> AnyPtr`,
||| `AnyPtr -> AnyPtr -> AnyPtr`, `AnyPtr -> Int -> AnyPtr`, etc.) to
||| exercise the typechecker realistically, but bodies just return
||| their first argument or 0 / "". For Phase 0b the question is:
||| does adding 150 interface methods + 5 instances bloat
||| `idris2 --check` time materially? Acceptance: < 20% delta on the
||| ipkg-wide clean typecheck.
|||
||| Delete this module after Phase 0c writes the design decision and
||| the real Phase 2.x refactor begins.
module Device.ProtoWide


----------------------------------------------------------------------
-- Slice 1: UserDeviceCore — lifecycle + arithmetic (~30 ops)
----------------------------------------------------------------------

public export
interface UserDeviceCore (0 d : Type) where
  wcName     : String
  wcScalar   : Double -> Int -> AnyPtr
  wcCreate1d : Int -> AnyPtr -> Int -> AnyPtr
  wcCreate2d : Int -> Int -> AnyPtr -> Int -> AnyPtr
  wcFree     : AnyPtr -> ()
  wcItem     : AnyPtr -> Double
  wcItem1d   : AnyPtr -> Int -> Double
  wcItem2d   : AnyPtr -> Int -> Int -> Double
  wcClone    : AnyPtr -> AnyPtr
  wcDetach   : AnyPtr -> AnyPtr
  wcAdd      : AnyPtr -> AnyPtr -> AnyPtr
  wcSub      : AnyPtr -> AnyPtr -> AnyPtr
  wcMul      : AnyPtr -> AnyPtr -> AnyPtr
  wcDiv      : AnyPtr -> AnyPtr -> AnyPtr
  wcNeg      : AnyPtr -> AnyPtr
  wcAbs      : AnyPtr -> AnyPtr
  wcExp      : AnyPtr -> AnyPtr
  wcLog      : AnyPtr -> AnyPtr
  wcSqrt     : AnyPtr -> AnyPtr
  wcPow      : AnyPtr -> AnyPtr -> AnyPtr
  wcSigmoid  : AnyPtr -> AnyPtr
  wcTanh     : AnyPtr -> AnyPtr
  wcAddScalar : AnyPtr -> Double -> AnyPtr
  wcMulScalar : AnyPtr -> Double -> AnyPtr
  wcClampMin  : AnyPtr -> Double -> AnyPtr
  wcSubScalarInplace : AnyPtr -> Double -> AnyPtr
  wcDim      : AnyPtr -> Int
  wcSizeAt   : AnyPtr -> Int -> Int
  wcNumel    : AnyPtr -> Int
  wcDeviceTag : AnyPtr -> String
  wcToDevice : AnyPtr -> String -> AnyPtr


----------------------------------------------------------------------
-- Slice 2: UserDeviceLinear — matmul + reductions + reshape (~30 ops)
----------------------------------------------------------------------

public export
interface UserDeviceLinear (0 d : Type) where
  wlMv         : AnyPtr -> AnyPtr -> AnyPtr
  wlLinear     : AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr
  wlDot        : AnyPtr -> AnyPtr -> AnyPtr
  wlOuter      : AnyPtr -> AnyPtr -> AnyPtr
  wlMatmul     : AnyPtr -> AnyPtr -> AnyPtr
  wlBmm        : AnyPtr -> AnyPtr -> AnyPtr
  wlBmm3       : AnyPtr -> AnyPtr -> AnyPtr
  wlTranspose  : AnyPtr -> AnyPtr
  wlSum        : AnyPtr -> AnyPtr
  wlSumDim     : AnyPtr -> Int -> Int -> AnyPtr
  wlMean       : AnyPtr -> AnyPtr
  wlMin        : AnyPtr -> AnyPtr
  wlMax        : AnyPtr -> AnyPtr
  wlSelect     : AnyPtr -> Int -> Int -> AnyPtr
  wlUnsqueeze  : AnyPtr -> Int -> AnyPtr
  wlSqueeze    : AnyPtr -> Int -> AnyPtr
  wlStack      : AnyPtr -> Int -> Int -> AnyPtr
  wlView1d     : AnyPtr -> Int -> AnyPtr
  wlView2d     : AnyPtr -> Int -> Int -> AnyPtr
  wlReshape3d  : AnyPtr -> Int -> Int -> Int -> AnyPtr
  wlReshape4d  : AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr
  wlExpandMask : AnyPtr -> Int -> AnyPtr
  wlCat        : AnyPtr -> Int -> Int -> AnyPtr
  wlCat2       : AnyPtr -> AnyPtr -> AnyPtr
  wlConcatAxis1 : AnyPtr -> AnyPtr -> AnyPtr
  wlNarrow     : AnyPtr -> Int -> Int -> Int -> AnyPtr
  wlGather     : AnyPtr -> AnyPtr -> Int -> AnyPtr
  wlScatterAdd : AnyPtr -> AnyPtr -> Int -> AnyPtr
  wlArgsort    : AnyPtr -> Int -> Int -> AnyPtr
  wlCumprod    : AnyPtr -> Int -> AnyPtr


----------------------------------------------------------------------
-- Slice 3: UserDeviceNN — activations + softmax + norms (~30 ops)
----------------------------------------------------------------------

public export
interface UserDeviceNN (0 d : Type) where
  wnGelu        : AnyPtr -> AnyPtr
  wnLeakyRelu   : AnyPtr -> Double -> AnyPtr
  wnSilu        : AnyPtr -> AnyPtr
  wnSoftplus    : AnyPtr -> AnyPtr
  wnSoftmax     : AnyPtr -> Int -> AnyPtr
  wnLogSoftmax  : AnyPtr -> Int -> AnyPtr
  wnSoftmax3d   : AnyPtr -> AnyPtr
  wnLayerNorm2d : AnyPtr -> AnyPtr -> AnyPtr -> Int -> Int -> Double -> AnyPtr
  wnRmsNorm     : AnyPtr -> AnyPtr -> Int -> Int -> Double -> AnyPtr
  wnBatchNorm   : AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr
               -> Int -> Int -> Int -> Double -> Double -> AnyPtr
  wnDropout     : AnyPtr -> Double -> Int -> Int -> AnyPtr
  wnEmbedding   : AnyPtr -> AnyPtr -> Int -> Int -> AnyPtr
  wnOneHot      : AnyPtr -> Int -> Int -> AnyPtr
  wnCosineSim   : AnyPtr -> AnyPtr -> Int -> AnyPtr
  wnCrossAttn   : AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr -> Double -> AnyPtr
  wnBceLogits   : AnyPtr -> AnyPtr -> AnyPtr
  wnNllLoss     : AnyPtr -> AnyPtr -> AnyPtr
  wnSoftXent    : AnyPtr -> AnyPtr -> AnyPtr
  wnGruCell     : AnyPtr -> AnyPtr -> AnyPtr -> Int -> AnyPtr
  wnLstmCell    : AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr -> ()
  wnLstmGates   : AnyPtr -> AnyPtr -> Int -> AnyPtr
  wnPairFirst   : AnyPtr -> AnyPtr
  wnPairSecond  : AnyPtr -> AnyPtr
  wnResidualAdd : AnyPtr -> AnyPtr -> AnyPtr
  wnScaledDot   : AnyPtr -> AnyPtr -> AnyPtr -> Double -> AnyPtr
  wnAttnMask    : AnyPtr -> AnyPtr -> AnyPtr
  wnGeluBack    : AnyPtr -> AnyPtr -> AnyPtr
  wnSigmoidBack : AnyPtr -> AnyPtr -> AnyPtr
  wnTanhBack    : AnyPtr -> AnyPtr -> AnyPtr
  wnReluMask    : AnyPtr -> AnyPtr -> AnyPtr


----------------------------------------------------------------------
-- Slice 4: UserDeviceConv — conv + pooling (~30 ops)
----------------------------------------------------------------------

public export
interface UserDeviceConv (0 d : Type) where
  wvConv1d          : AnyPtr -> AnyPtr -> AnyPtr -> Int -> Int -> AnyPtr
  wvConv1dCircular  : AnyPtr -> AnyPtr -> AnyPtr
  wvConv2d          : AnyPtr -> AnyPtr -> AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr
  wvConv2dBatched   : AnyPtr -> AnyPtr -> AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr
  wvAvgPool1d       : AnyPtr -> Int -> Int -> AnyPtr
  wvAvgPool2d       : AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr
  wvMaxPool1d       : AnyPtr -> Int -> Int -> AnyPtr
  wvMaxPool2d       : AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr
  wvMaxPool2dBatch  : AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr
  wvConv1dBack      : AnyPtr -> AnyPtr -> AnyPtr -> Int -> Int -> AnyPtr
  wvConv2dBack      : AnyPtr -> AnyPtr -> AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr
  wvAvgPool1dBack   : AnyPtr -> Int -> Int -> AnyPtr
  wvAvgPool2dBack   : AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr
  wvMaxPool1dBack   : AnyPtr -> Int -> Int -> AnyPtr
  wvMaxPool2dBack   : AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr
  wvIm2Col          : AnyPtr -> Int -> Int -> Int -> AnyPtr
  wvCol2Im          : AnyPtr -> Int -> Int -> Int -> AnyPtr
  wvUnfold1d        : AnyPtr -> Int -> Int -> AnyPtr
  wvFold1d          : AnyPtr -> Int -> Int -> AnyPtr
  wvUnfold2d        : AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr
  wvFold2d          : AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr
  wvUpsample        : AnyPtr -> Int -> Int -> AnyPtr
  wvDownsample      : AnyPtr -> Int -> Int -> AnyPtr
  wvBilinear        : AnyPtr -> Int -> Int -> AnyPtr
  wvPad1d           : AnyPtr -> Int -> Int -> AnyPtr
  wvPad2d           : AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr
  wvCrop1d          : AnyPtr -> Int -> Int -> AnyPtr
  wvCrop2d          : AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr
  wvDilate1d        : AnyPtr -> Int -> AnyPtr
  wvDilate2d        : AnyPtr -> Int -> Int -> AnyPtr


----------------------------------------------------------------------
-- Slice 5: UserDeviceTape — autograd + param registry + IO (~30 ops)
----------------------------------------------------------------------

public export
interface UserDeviceTape (0 d : Type) where
  wtReqGrad      : AnyPtr -> Int
  wtSetReqGrad   : AnyPtr -> Int -> PrimIO ()
  wtNoGradBegin  : PrimIO ()
  wtNoGradEnd    : PrimIO ()
  wtBackward     : AnyPtr -> PrimIO ()
  wtBackwardWith : AnyPtr -> AnyPtr -> PrimIO ()
  wtRetain       : AnyPtr -> PrimIO ()
  wtZeroGrad     : AnyPtr -> PrimIO ()
  wtTapeReset    : PrimIO ()
  wtTapeMark     : Int
  wtTapeRewind   : Int -> PrimIO ()
  wtParamReg     : String -> AnyPtr -> AnyPtr
  wtParamClear   : ()
  wtParamCount   : Int
  wtParamName    : Int -> String
  wtParamGradItem : Int -> Double
  wtParamGradItemAt : Int -> Int -> Double
  wtParamZeroAll : Int -> Int
  wtParamSubDelta : Int -> Double -> ()
  wtParamGradItemAndZero : Int -> Double
  wtCreateParam1d : Int -> AnyPtr -> AnyPtr
  wtCreateParam2d : Int -> Int -> AnyPtr -> AnyPtr
  wtCreateParam3d : Int -> Int -> Int -> AnyPtr -> AnyPtr
  wtCreateState1d : Int -> AnyPtr -> AnyPtr
  wtCreateState2d : Int -> Int -> AnyPtr -> AnyPtr
  wtAllocDoubles  : Int -> AnyPtr
  wtReadDouble    : AnyPtr -> Int -> Double
  wtWriteDouble   : AnyPtr -> Int -> Double -> ()
  wtPrint         : AnyPtr -> ()
  wtSeq           : AnyPtr -> AnyPtr -> AnyPtr


----------------------------------------------------------------------
-- TapeDevWide instance — stub bodies just return the first AnyPtr arg
-- or a zero value. Real Phase 2.x instances will forward to live FFI.
----------------------------------------------------------------------

public export
data TapeDevWide : Type where MkTapeDevWide : TapeDevWide

-- A single shared dummy pointer (NULL via prim__getString hack — we
-- never deref these in Phase 0b; the typechecker is what's under test).
%foreign "C:tensor_create_scalar,libidrisml"
prim__nullScalar : Double -> Int -> AnyPtr

dummyPtr : AnyPtr
dummyPtr = prim__nullScalar 0.0 0

public export
UserDeviceCore TapeDevWide where
  wcName              = "tape-wide"
  wcScalar _ _        = dummyPtr
  wcCreate1d _ _ _    = dummyPtr
  wcCreate2d _ _ _ _  = dummyPtr
  wcFree _            = ()
  wcItem _            = 0.0
  wcItem1d _ _        = 0.0
  wcItem2d _ _ _      = 0.0
  wcClone p           = p
  wcDetach p          = p
  wcAdd a _           = a
  wcSub a _           = a
  wcMul a _           = a
  wcDiv a _           = a
  wcNeg p             = p
  wcAbs p             = p
  wcExp p             = p
  wcLog p             = p
  wcSqrt p            = p
  wcPow a _           = a
  wcSigmoid p         = p
  wcTanh p            = p
  wcAddScalar p _     = p
  wcMulScalar p _     = p
  wcClampMin p _      = p
  wcSubScalarInplace p _ = p
  wcDim _             = 0
  wcSizeAt _ _        = 0
  wcNumel _           = 0
  wcDeviceTag _       = "tape-wide"
  wcToDevice p _      = p

public export
UserDeviceLinear TapeDevWide where
  wlMv a _            = a
  wlLinear a _ _      = a
  wlDot a _           = a
  wlOuter a _         = a
  wlMatmul a _        = a
  wlBmm a _           = a
  wlBmm3 a _          = a
  wlTranspose p       = p
  wlSum p             = p
  wlSumDim p _ _      = p
  wlMean p            = p
  wlMin p             = p
  wlMax p             = p
  wlSelect p _ _      = p
  wlUnsqueeze p _     = p
  wlSqueeze p _       = p
  wlStack p _ _       = p
  wlView1d p _        = p
  wlView2d p _ _      = p
  wlReshape3d p _ _ _ = p
  wlReshape4d p _ _ _ _ = p
  wlExpandMask p _    = p
  wlCat p _ _         = p
  wlCat2 a _          = a
  wlConcatAxis1 a _   = a
  wlNarrow p _ _ _    = p
  wlGather p _ _      = p
  wlScatterAdd p _ _  = p
  wlArgsort p _ _     = p
  wlCumprod p _       = p

public export
UserDeviceNN TapeDevWide where
  wnGelu p            = p
  wnLeakyRelu p _     = p
  wnSilu p            = p
  wnSoftplus p        = p
  wnSoftmax p _       = p
  wnLogSoftmax p _    = p
  wnSoftmax3d p       = p
  wnLayerNorm2d p _ _ _ _ _ = p
  wnRmsNorm p _ _ _ _ = p
  wnBatchNorm p _ _ _ _ _ _ _ _ _ = p
  wnDropout p _ _ _   = p
  wnEmbedding p _ _ _ = p
  wnOneHot p _ _      = p
  wnCosineSim a _ _   = a
  wnCrossAttn p _ _ _ _ = p
  wnBceLogits a _     = a
  wnNllLoss a _       = a
  wnSoftXent a _      = a
  wnGruCell p _ _ _   = p
  wnLstmCell _ _ _ _ _ _ _ _ _ = ()
  wnLstmGates p _ _   = p
  wnPairFirst p       = p
  wnPairSecond p      = p
  wnResidualAdd a _   = a
  wnScaledDot p _ _ _ = p
  wnAttnMask a _      = a
  wnGeluBack a _      = a
  wnSigmoidBack a _   = a
  wnTanhBack a _      = a
  wnReluMask a _      = a

public export
UserDeviceConv TapeDevWide where
  wvConv1d p _ _ _ _      = p
  wvConv1dCircular a _    = a
  wvConv2d p _ _ _ _ _ _  = p
  wvConv2dBatched p _ _ _ _ _ _ = p
  wvAvgPool1d p _ _       = p
  wvAvgPool2d p _ _ _ _   = p
  wvMaxPool1d p _ _       = p
  wvMaxPool2d p _ _ _ _   = p
  wvMaxPool2dBatch p _ _ _ _ = p
  wvConv1dBack p _ _ _ _  = p
  wvConv2dBack p _ _ _ _ _ _ = p
  wvAvgPool1dBack p _ _   = p
  wvAvgPool2dBack p _ _ _ _ = p
  wvMaxPool1dBack p _ _   = p
  wvMaxPool2dBack p _ _ _ _ = p
  wvIm2Col p _ _ _        = p
  wvCol2Im p _ _ _        = p
  wvUnfold1d p _ _        = p
  wvFold1d p _ _          = p
  wvUnfold2d p _ _ _ _    = p
  wvFold2d p _ _ _ _      = p
  wvUpsample p _ _        = p
  wvDownsample p _ _      = p
  wvBilinear p _ _        = p
  wvPad1d p _ _           = p
  wvPad2d p _ _ _ _       = p
  wvCrop1d p _ _          = p
  wvCrop2d p _ _ _ _      = p
  wvDilate1d p _          = p
  wvDilate2d p _ _        = p

noopPrim : PrimIO ()
noopPrim w = MkIORes () w

public export
UserDeviceTape TapeDevWide where
  wtReqGrad _           = 0
  wtSetReqGrad _ _      = noopPrim
  wtNoGradBegin         = noopPrim
  wtNoGradEnd           = noopPrim
  wtBackward _          = noopPrim
  wtBackwardWith _ _    = noopPrim
  wtRetain _            = noopPrim
  wtZeroGrad _          = noopPrim
  wtTapeReset           = noopPrim
  wtTapeMark            = 0
  wtTapeRewind _        = noopPrim
  wtParamReg _ p        = p
  wtParamClear          = ()
  wtParamCount          = 0
  wtParamName _         = ""
  wtParamGradItem _     = 0.0
  wtParamGradItemAt _ _ = 0.0
  wtParamZeroAll _      = 0
  wtParamSubDelta _ _   = ()
  wtParamGradItemAndZero _ = 0.0
  wtCreateParam1d _ p   = p
  wtCreateParam2d _ _ p = p
  wtCreateParam3d _ _ _ p = p
  wtCreateState1d _ p   = p
  wtCreateState2d _ _ p = p
  wtAllocDoubles _      = dummyPtr
  wtReadDouble _ _      = 0.0
  wtWriteDouble _ _ _   = ()
  wtPrint _             = ()
  wtSeq a _             = a


----------------------------------------------------------------------
-- Resolution exerciser — forces Idris to look up every slice's
-- methods on TapeDevWide. The point is to make sure the typechecker
-- doesn't skip resolution work when the instance is never called.
----------------------------------------------------------------------

public export
exerciseResolution : IO ()
exerciseResolution = do
  let _ = wcName {d = TapeDevWide}
  let _ = wlMv {d = TapeDevWide} dummyPtr dummyPtr
  let _ = wnGelu {d = TapeDevWide} dummyPtr
  let _ = wvConv1d {d = TapeDevWide} dummyPtr dummyPtr dummyPtr 1 1
  let _ = wtParamCount {d = TapeDevWide}
  pure ()
