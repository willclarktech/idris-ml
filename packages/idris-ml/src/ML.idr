||| `ML` — the single-import user surface. `import ML` brings the daily
||| toolkit: the autograd `Tensor` (+ operator aliases `+.`/`-.`/`*.`/`*:`,
||| `tgather`/`tgatherRows`/`tmaxRows`, the loss vocabulary), the `Nn` model
||| library (Module/Params/Seq/Init/Group/Recurrent + all layers),
||| optimizers + the typed-scope surface (`Train.Freeze`'s
||| `restrictTo`/`freezeGroup`/`setGroupLR`/`namesMatching`, fed exact names
||| from `Nn.Group`'s `groupOf`/`reflectNames`), the `Dataset`/`DataStream`
||| data surface, the `fit` driver + engine pieces, checkpoint save/load, and
||| the `Backend` constraint bundle. `import ML.Simple` additionally pins
||| `(Ex, F)` to the build's default cell so example/tutorial code needs zero
||| `{ex=}`.
module ML

import public Backend
import public Checkpoint
import public DataStream
import public Dataset
import public Fit
import public Nn
import public Nn.Init
import public Optimizer
import public Tensor
import public Train.Engine
import public Train.Freeze
