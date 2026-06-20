||| Training-surface instance slices: Autograd, ParamRegistry,
||| Optimizer, Serialize, MemoryHygiene, Diagnostics, Profiling, TensorCreate.
module Executor.Torch.Training

import BackendLib
import DType.Core
import Executor.Core
import public Executor.Torch.Nn
import Hardware
import Preset

public export
{d : TorchHwDev} -> UserExecutorAutograd (TorchExecutor d) where
  -- >>> GENERATED FROM ffi_manifest.py — gen-executor-instances.py >>>
  primBackward        = prim__backwardTorch
  primDetach          = prim__detachTorch
  primNoGradBegin     = prim__noGradBeginTorch
  primNoGradEnd       = prim__noGradEndTorch
  primRequiresGrad    = prim__requiresGradTorch
  primSetRequiresGrad = prim__setRequiresGradTorch
  primWithGrad        = prim__withGradTorch
  -- <<< END GENERATED <<<

public export
{d : TorchHwDev} -> UserExecutorParamRegistry (TorchExecutor d) where
  -- >>> GENERATED FROM ffi_manifest.py — gen-executor-instances.py >>>
  primParamCount          = prim__paramCountTorch
  primParamEraseByPrefix  = prim__paramEraseByPrefixTorch
  primParamGradItemAt     = prim__paramGradItemAtTorch
  primParamIsBuffer       = prim__paramIsBufferTorch
  primParamName           = prim__paramNameTorch
  primParamRegister       = prim__paramRegisterTorch
  primParamRegisterBuffer = prim__paramRegisterBufferTorch
  primParamZeroAll        = prim__paramZeroAllTorch
  -- <<< END GENERATED <<<

public export
{d : TorchHwDev} -> UserExecutorOptimizer (TorchExecutor d) where
  -- >>> GENERATED FROM ffi_manifest.py — gen-executor-instances.py >>>
  primNativeTrainStep        = prim__nativeTrainStepTorch
  primNativeTrainStepScaled  = prim__nativeTrainStepScaledTorch
  primOptimizerCreateAdam    = prim__optimizerCreateAdamTorch
  primOptimizerCreateAdamW   = prim__optimizerCreateAdamWTorch
  primOptimizerCreateRmsprop = prim__optimizerCreateRmspropTorch
  primOptimizerCreateSgd     = prim__optimizerCreateSgdTorch
  primOptimizerOwnParam      = prim__optimizerOwnParamTorch
  primOptimizerSetLr         = prim__optimizerSetLrTorch
  primOptimizerSetParamLr    = prim__optimizerSetParamLrTorch
  -- <<< END GENERATED <<<

public export
{d : TorchHwDev} -> UserExecutorSerialize (TorchExecutor d) where
  -- >>> GENERATED FROM ffi_manifest.py — gen-executor-instances.py >>>
  primOptimizerLoad          = prim__optimizerLoadTorch
  primOptimizerSave          = prim__optimizerSaveTorch
  primParamLoad              = prim__paramLoadTorch
  primParamLoadRenamed       = prim__paramLoadRenamedTorch
  primParamLoadWithPolicy    = prim__paramLoadWithPolicyTorch
  primParamLoadWithPrefix    = prim__paramLoadWithPrefixTorch
  primParamSave              = prim__paramSaveTorch
  primParamSaveByName        = prim__paramSaveByNameTorch
  primParamSaveByNameRenamed = prim__paramSaveByNameRenamedTorch
  -- <<< END GENERATED <<<

public export
{d : TorchHwDev} -> UserExecutorMemoryHygiene (TorchExecutor d) where
  -- >>> GENERATED FROM ffi_manifest.py — gen-executor-instances.py >>>
  primEpochBegin           = prim__epochBeginTorch
  primEpochEnd             = prim__epochEndTorch
  primReleaseAllPersistent = prim__releaseAllPersistentTorch
  primResetForEval         = prim__resetForEvalTorch
  -- <<< END GENERATED <<<

public export
{d : TorchHwDev} -> UserExecutorDiagnostics (TorchExecutor d) where
  -- >>> GENERATED FROM ffi_manifest.py — gen-executor-instances.py >>>
  primLiveCount     = prim__liveCountTorch
  primPeakLiveCount = prim__peakLiveCountTorch
  primPerfOpCount   = prim__perfOpCountTorch
  -- <<< END GENERATED <<<

public export
{d : TorchHwDev} -> UserExecutorProfiling (TorchExecutor d) where
  -- >>> GENERATED FROM ffi_manifest.py — gen-executor-instances.py >>>
  primPerfReset     = prim__perfResetTorch
  primProfileReport = prim__profileReportTorch
  primProfileReset  = prim__profileResetTorch
  -- <<< END GENERATED <<<

public export
{d : TorchHwDev} -> UserExecutorTensorCreate (TorchExecutor d) where
  -- >>> GENERATED FROM ffi_manifest.py — gen-executor-instances.py >>>
  primCastStreamed          = prim__castStreamedTorch
  primCreate1dStreamed      = prim__create1dStreamedTorch
  primCreate2dStreamed      = prim__create2dStreamedTorch
  primCreateParam1dStreamed = prim__createParam1dStreamedTorch
  primCreateParam2dStreamed = prim__createParam2dStreamedTorch
  primCreateParam3dStreamed = prim__createParam3dStreamedTorch
  primCreateParam4dStreamed = prim__createParam4dStreamedTorch
  primCreateScalarStreamed  = prim__createScalarStreamedTorch
  primCreateState1dStreamed = prim__createState1dStreamedTorch
  primCreateState2dStreamed = prim__createState2dStreamedTorch
  primCreateStreamed        = prim__createStreamedTorch
  primItem2d                = prim__item2dTorch
  primOneHot                = prim__oneHotTorch
  primSetInitSeedStreamed   = prim__setInitSeedStreamedTorch
  primTensorDim             = prim__tensorDimTorch
  primTensorSizeAt          = prim__tensorSizeAtTorch
  -- <<< END GENERATED <<<
