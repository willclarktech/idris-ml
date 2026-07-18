||| `Ml` — the single-import user surface. `import Ml` brings the daily
||| toolkit: the autograd `Tensor` (+ operator aliases `+.`/`-.`/`*.`/`*:`,
||| `tgather`/`tgatherRows`/`tmaxRows`, the loss vocabulary), the `Nn` model
||| library (Module/Params/Seq/Init/Group/Recurrent + all layers),
||| optimizers + the typed-scope surface (`Train.Freeze`'s
||| `restrictTo`/`freezeGroup`/`setGroupLR`/`namesMatching`, fed exact names
||| from `Nn.Group`'s `groupOf`/`reflectNames`), the `Dataset`/`DataStream`
||| data surface, the `fit` driver + engine pieces, checkpoint save/load, and
||| the `Backend` constraint bundle. `import Ml.Simple` additionally pins
||| `(Ex, F)` to the build's default cell so example/tutorial code needs zero
||| `{ex=}`.
module Ml

import public Ml.Backend
import public Ml.Checkpoint
import public Ml.DataStream
import public Ml.Dataset
import public Ml.Fit
import public Ml.Nn
import public Ml.Nn.Init
import public Ml.Optimizer
import public Ml.Tensor
import public Ml.Train.Engine
import public Ml.Train.Freeze
