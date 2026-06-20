||| Umbrella re-export of the Tensor surface, split into cohesive
||| sub-modules (Handle / Core / Ops / Construct / Quant / Index /
||| Optimizer / Linear). The public API is unchanged: `import Tensor`
||| still brings the whole autograd-handle surface into scope.
module Tensor

import public DType.Core
import public GradMode
import public Init
import public Tensor.Construct
import public Tensor.Core
import public Tensor.Handle
import public Tensor.Index
import public Tensor.Internal
import public Tensor.Linear
import public Tensor.Ops
import public Tensor.Optimizer
import public Tensor.Quant
