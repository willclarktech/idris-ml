||| Training-surface instance slices: Autograd, ParamRegistry,
||| Optimizer, Serialize, MemoryHygiene, Diagnostics, Profiling, TensorCreate.
module Ml.Executor.Mlx.Training

import Ml.BackendLib
import Ml.DType.Core
import Ml.Executor.Core
import public Ml.Executor.Mlx.Nn
import Ml.Hardware
import Ml.Preset

public export
{s : MlxStream} -> UserExecutorAutograd (MlxExecutor s) where
  -- >>> GENERATED FROM ffi_manifest.py — gen-executor-instances.py >>>
  primBackward        = prim__backwardMlx
  primDetach a0       = prim__detachMlxStreamed a0 (streamTag s)
  primNoGradBegin     = prim__noGradBeginMlx
  primNoGradEnd       = prim__noGradEndMlx
  primRequiresGrad    = prim__requiresGradMlx
  primSetRequiresGrad = prim__setRequiresGradMlx
  primWithGrad a0     = prim__withGradMlxStreamed a0 (streamTag s)
  -- <<< END GENERATED <<<

public export
{s : MlxStream} -> UserExecutorParamRegistry (MlxExecutor s) where
  -- >>> GENERATED FROM ffi_manifest.py — gen-executor-instances.py >>>
  primParamCount          = prim__paramCountMlx
  primParamEraseByPrefix  = prim__paramEraseByPrefixMlx
  primParamGradItemAt     = prim__paramGradItemAtMlx
  primParamIsBuffer       = prim__paramIsBufferMlx
  primParamName           = prim__paramNameMlx
  primParamRegister       = prim__paramRegisterMlx
  primParamRegisterBuffer = prim__paramRegisterBufferMlx
  primParamZeroAll        = prim__paramZeroAllMlx
  -- <<< END GENERATED <<<

public export
{s : MlxStream} -> UserExecutorOptimizer (MlxExecutor s) where
  -- >>> GENERATED FROM ffi_manifest.py — gen-executor-instances.py >>>
  primNativeTrainStep        = prim__nativeTrainStepMlx
  primNativeTrainStepScaled  = prim__nativeTrainStepScaledMlx
  primOptimizerCreateAdam    = prim__optimizerCreateAdamMlx
  primOptimizerCreateAdamW   = prim__optimizerCreateAdamWMlx
  primOptimizerCreateRmsprop = prim__optimizerCreateRmspropMlx
  primOptimizerCreateSgd     = prim__optimizerCreateSgdMlx
  primOptimizerOwnParam      = prim__optimizerOwnParamMlx
  primOptimizerSetLr         = prim__optimizerSetLrMlx
  primOptimizerSetParamLr    = prim__optimizerSetParamLrMlx
  -- <<< END GENERATED <<<

public export
{s : MlxStream} -> UserExecutorSerialize (MlxExecutor s) where
  -- >>> GENERATED FROM ffi_manifest.py — gen-executor-instances.py >>>
  primOptimizerLoad          = prim__optimizerLoadMlx
  primOptimizerSave          = prim__optimizerSaveMlx
  primParamLoad              = prim__paramLoadMlx
  primParamLoadRenamed       = prim__paramLoadRenamedMlx
  primParamLoadWithPolicy    = prim__paramLoadWithPolicyMlx
  primParamLoadWithPrefix    = prim__paramLoadWithPrefixMlx
  primParamSave              = prim__paramSaveMlx
  primParamSaveByName        = prim__paramSaveByNameMlx
  primParamSaveByNameRenamed = prim__paramSaveByNameRenamedMlx
  -- <<< END GENERATED <<<

public export
{s : MlxStream} -> UserExecutorMemoryHygiene (MlxExecutor s) where
  -- >>> GENERATED FROM ffi_manifest.py — gen-executor-instances.py >>>
  primEpochBegin           = prim__epochBeginMlx
  primEpochEnd             = prim__epochEndMlx
  primReleaseAllPersistent = prim__releaseAllPersistentMlx
  primResetForEval         = prim__resetForEvalMlx
  -- <<< END GENERATED <<<

public export
{s : MlxStream} -> UserExecutorDiagnostics (MlxExecutor s) where
  -- >>> GENERATED FROM ffi_manifest.py — gen-executor-instances.py >>>
  primLiveCount     = prim__liveCountMlx
  primPeakLiveCount = prim__peakLiveCountMlx
  primPerfOpCount   = prim__perfOpCountMlx
  -- <<< END GENERATED <<<

public export
{s : MlxStream} -> UserExecutorProfiling (MlxExecutor s) where
  -- >>> GENERATED FROM ffi_manifest.py — gen-executor-instances.py >>>
  primPerfReset     = prim__perfResetMlx
  primProfileReport = prim__profileReportMlx
  primProfileReset  = prim__profileResetMlx
  -- <<< END GENERATED <<<

public export
{s : MlxStream} -> UserExecutorTensorCreate (MlxExecutor s) where
  -- >>> GENERATED FROM ffi_manifest.py — gen-executor-instances.py >>>
  primCastStreamed          = prim__castStreamedMlx
  primCreate1dStreamed      = prim__create1dStreamedMlx
  primCreate2dStreamed      = prim__create2dStreamedMlx
  primCreateParam1dStreamed = prim__createParam1dStreamedMlx
  primCreateParam2dStreamed = prim__createParam2dStreamedMlx
  primCreateParam3dStreamed = prim__createParam3dStreamedMlx
  primCreateParam4dStreamed = prim__createParam4dStreamedMlx
  primCreateScalarStreamed  = prim__createScalarStreamedMlx
  primCreateState1dStreamed = prim__createState1dStreamedMlx
  primCreateState2dStreamed = prim__createState2dStreamedMlx
  primCreateStreamed        = prim__createStreamedMlx
  primItem2d                = prim__item2dMlx
  primOneHot                = prim__oneHotMlx
  primSetInitSeedStreamed   = prim__setInitSeedStreamedMlx
  primTensorDim             = prim__tensorDimMlx
  primTensorSizeAt          = prim__tensorSizeAtMlx
  -- <<< END GENERATED <<<
