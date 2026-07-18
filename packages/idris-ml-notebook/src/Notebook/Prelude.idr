-- | Notebook Prelude: re-exports all idris-ml modules for interactive use.
-- |
-- | Loaded automatically by the Jupyter kernel so users don't need
-- | manual :module imports for common operations.

module Notebook.Prelude

import public Control.Linear.LIO
import public Data.Linear.Notation
import public Data.List
import public Data.String
import public Data.Vect
import public Decidable.Equality
import public System
import public System.Random

import public Ml.Array
import public Ml.Checkpoint
import public Ml.DataStream
import public Ml.Dataset
import public Ml.Executor
import public Ml.Fit
import public Ml.Floating
import public Ml.Hpo
import public Ml.HwExecutors
import public Ml.Init
import public Ml.Math
import public Ml.Nn
import public Ml.Optimizer
import public Ml.RL.Gae
import public Ml.RL.ReplayBuffer
import public Ml.Sampler
import public Ml.Schedule
import public Ml.Tensor
import public Ml.Train
import public Ml.Util
