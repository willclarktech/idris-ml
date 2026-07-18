||| Training-surface instance slices: Autograd, ParamRegistry,
||| Optimizer, Serialize, MemoryHygiene, Diagnostics, Profiling, TensorCreate.
module Ml.Executor.Tape.Training

import Ml.BackendLib
import Ml.DType.Core
import Ml.Executor.Core
import public Ml.Executor.Tape.Nn
import Ml.Hardware
import Ml.Preset

public export
UserExecutorAutograd TapeExecutor where
  -- >>> GENERATED FROM ffi_manifest.py — gen-executor-instances.py >>>
  primBackward        = prim__backwardTape
  primDetach          = prim__detachTape
  primNoGradBegin     = prim__noGradBeginTape
  primNoGradEnd       = prim__noGradEndTape
  primRequiresGrad    = prim__requiresGradTape
  primSetRequiresGrad = prim__setRequiresGradTape
  primWithGrad        = prim__withGradTape
  -- <<< END GENERATED <<<

public export
UserExecutorParamRegistry TapeExecutor where
  -- >>> GENERATED FROM ffi_manifest.py — gen-executor-instances.py >>>
  primParamCount          = prim__paramCountTape
  primParamEraseByPrefix  = prim__paramEraseByPrefixTape
  primParamGradItemAt     = prim__paramGradItemAtTape
  primParamIsBuffer       = prim__paramIsBufferTape
  primParamName           = prim__paramNameTape
  primParamRegister       = prim__paramRegisterTape
  primParamRegisterBuffer = prim__paramRegisterBufferTape
  primParamZeroAll        = prim__paramZeroAllTape
  -- <<< END GENERATED <<<

public export
UserExecutorOptimizer TapeExecutor where
  -- >>> GENERATED FROM ffi_manifest.py — gen-executor-instances.py >>>
  primNativeTrainStep        = prim__nativeTrainStepTape
  primNativeTrainStepScaled  = prim__nativeTrainStepScaledTape
  primOptimizerCreateAdam    = prim__optimizerCreateAdamTape
  primOptimizerCreateAdamW   = prim__optimizerCreateAdamWTape
  primOptimizerCreateRmsprop = prim__optimizerCreateRmspropTape
  primOptimizerCreateSgd     = prim__optimizerCreateSgdTape
  primOptimizerOwnParam      = prim__optimizerOwnParamTape
  primOptimizerSetLr         = prim__optimizerSetLrTape
  primOptimizerSetParamLr    = prim__optimizerSetParamLrTape
  -- <<< END GENERATED <<<

public export
UserExecutorSerialize TapeExecutor where
  -- >>> GENERATED FROM ffi_manifest.py — gen-executor-instances.py >>>
  primOptimizerLoad          = prim__optimizerLoadTape
  primOptimizerSave          = prim__optimizerSaveTape
  primParamLoad              = prim__paramLoadTape
  primParamLoadRenamed       = prim__paramLoadRenamedTape
  primParamLoadWithPolicy    = prim__paramLoadWithPolicyTape
  primParamLoadWithPrefix    = prim__paramLoadWithPrefixTape
  primParamSave              = prim__paramSaveTape
  primParamSaveByName        = prim__paramSaveByNameTape
  primParamSaveByNameRenamed = prim__paramSaveByNameRenamedTape
  -- <<< END GENERATED <<<

public export
UserExecutorMemoryHygiene TapeExecutor where
  -- >>> GENERATED FROM ffi_manifest.py — gen-executor-instances.py >>>
  primEpochBegin           = prim__epochBeginTape
  primEpochEnd             = prim__epochEndTape
  primReleaseAllPersistent = prim__releaseAllPersistentTape
  primResetForEval         = prim__resetForEvalTape
  -- <<< END GENERATED <<<

public export
UserExecutorDiagnostics TapeExecutor where
  -- >>> GENERATED FROM ffi_manifest.py — gen-executor-instances.py >>>
  primLiveCount     = prim__liveCountTape
  primPeakLiveCount = prim__peakLiveCountTape
  primPerfOpCount   = prim__perfOpCountTape
  -- <<< END GENERATED <<<

public export
UserExecutorProfiling TapeExecutor where
  -- >>> GENERATED FROM ffi_manifest.py — gen-executor-instances.py >>>
  primPerfReset     = prim__perfResetTape
  primProfileReport = prim__profileReportTape
  primProfileReset  = prim__profileResetTape
  -- <<< END GENERATED <<<

public export
UserExecutorTensorCreate TapeExecutor where
  -- >>> GENERATED FROM ffi_manifest.py — gen-executor-instances.py >>>
  primCastStreamed          = prim__castStreamedTape
  primCreate1dStreamed      = prim__create1dStreamedTape
  primCreate2dStreamed      = prim__create2dStreamedTape
  primCreateParam1dStreamed = prim__createParam1dStreamedTape
  primCreateParam2dStreamed = prim__createParam2dStreamedTape
  primCreateParam3dStreamed = prim__createParam3dStreamedTape
  primCreateParam4dStreamed = prim__createParam4dStreamedTape
  primCreateScalarStreamed  = prim__createScalarStreamedTape
  primCreateState1dStreamed = prim__createState1dStreamedTape
  primCreateState2dStreamed = prim__createState2dStreamedTape
  primCreateStreamed        = prim__createStreamedTape
  primItem2d                = prim__item2dTape
  primOneHot                = prim__oneHotTape
  primSetInitSeedStreamed   = prim__setInitSeedStreamedTape
  primTensorDim             = prim__tensorDimTape
  primTensorSizeAt          = prim__tensorSizeAtTape
  -- <<< END GENERATED <<<
