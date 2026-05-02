||| Hyperparameter optimization tooling — fastai-inspired.
|||
||| Re-exports the public APIs from `Hpo.*` so that consumers can
|||
|||   import Hpo
|||
||| and have everything HPO-related in scope.
|||
||| Currently exports:
|||   * `Hpo.LrFinder` — LR-range test (`lr_find`), `LrFindConfig`,
|||     `LrFindResult`, `defaultLrFindConfig`.
|||
||| Companion APIs in other modules:
|||   * `Schedule.idr` — `cosineAnnealing`, `oneCycle`, `withWarmup` etc.
|||   * `Train.idr` — `applySchedule` binds a `Schedule` to a
|||     `NativeOptimizer` for the `beforeEpoch` hook.
module Hpo

import public Hpo.LrFinder
