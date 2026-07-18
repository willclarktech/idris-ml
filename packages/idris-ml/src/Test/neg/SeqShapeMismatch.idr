||| NEGATIVE test: this file MUST NOT type-check.
|||
||| A `Seq` chain whose hidden dims don't line up — `fc1`'s out-dim (256)
||| never matches `fc2`'s in-dim (128). The `ChainFits` witness on `(::)`
||| exists so this fails with an error that NAMES BOTH DIMS:
|||
|||   Can't find an implementation for ChainFits 256 128.
|||
||| (Pre-ChainFits this surfaced as the opaque `Can't find an implementation
||| for Module ?l` — higher-order unification postpones the layer-constructor
||| inversion, and the failed `Module` search won error reporting.)
||| Gate: scripts/check-seq-shape-gate.sh via
||| `make test-integration-typegate-seq-shape`.
module SeqShapeMismatch

import Control.Linear.LIO

import Ml.Executor
import Ml.GradMode
import Ml.Nn.Init
import Ml.Nn.Linear
import Ml.Nn.Module
import Ml.Nn.Seq
import Ml.Tensor

Model : Type
Model = Seq 784 10 TapeExecutor F64 WithGrad

mkModel : Init Model
mkModel = do
  fc1 <- linear {i=784} {o=256}
  fc2 <- linear {i=128} {o=10}
  pure (fc1 :: fc2 :: Nil)
