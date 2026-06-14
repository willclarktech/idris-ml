||| `ML` — the single-import user surface. `import ML` brings the daily
||| toolkit: the autograd `Tensor` (+ operator aliases `+.`/`-.`/`*.`/`*:`,
||| `tgather`/`tgatherRows`/`tmaxRows`, the loss vocabulary), the `Nn` model
||| library (Module/Params/Seq/Init/Group/Recurrent + all layers),
||| optimizers, the `Dataset`/`DataStream` data surface, the `fit` driver +
||| engine pieces, checkpoint save/load, and the `Backend` constraint
||| bundle. `import ML.Simple` additionally pins `(Ex, F)` to the build's
||| default cell so example/tutorial code needs zero `{ex=}`.
module ML

import public Tensor
import public Nn
import public Nn.Init
import public Optimizer
import public Dataset
import public DataStream
import public Fit
import public Train.Engine
import public Checkpoint
import public Backend
