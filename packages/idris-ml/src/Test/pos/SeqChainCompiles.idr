||| Positive compile test: the well-sized counterpart of
||| `Test/neg/SeqShapeMismatch.idr`. The same chain shape with the hidden
||| dims lined up (784 -> 256 -> 10, plus an identity activation whose dims
||| are inferred through two `ChainFits` ties). This file MUST compile — it
||| proves the negative test fails for the *right* reason (the ChainFits
||| dim mismatch), not because the `Seq`/`ChainFits` surface itself broke.
module SeqChainCompiles

import Control.Linear.LIO

import Ml.Executor
import Ml.GradMode
import Ml.Nn.Activation
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
  fc2 <- linear {i=256} {o=10}
  pure (fc1 :: reluA :: fc2 :: Nil)
