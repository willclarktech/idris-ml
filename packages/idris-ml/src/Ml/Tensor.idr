||| Umbrella re-export of the Tensor surface, split into cohesive
||| sub-modules (Handle / Core / Ops / Construct / Quant / Index /
||| Optimizer / Linear). The public API is unchanged: `import Tensor`
||| still brings the whole autograd-handle surface into scope.
module Ml.Tensor

import public Ml.DType.Core
import public Ml.GradMode
import public Ml.Init
import public Ml.Tensor.Construct
import public Ml.Tensor.Core
import public Ml.Tensor.Handle
import public Ml.Tensor.Index
import public Ml.Tensor.Internal
import public Ml.Tensor.Linear
import public Ml.Tensor.Ops
import public Ml.Tensor.Optimizer
import public Ml.Tensor.Quant
