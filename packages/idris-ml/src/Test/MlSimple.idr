module Test.MlSimple

import Control.Linear.LIO
import Data.Linear.Notation
import Data.Vect

import ML.Simple
import Test.Harness

-- `import ML.Simple` alone brings Tensor + Nn + Init + the build's (Ex, F)
-- pin. A Linear is constructed and run with ZERO `{ex=}` spellings — the
-- types are pinned only by annotating results to `Ex`/`F`. (The raw value
-- read uses `{ex=Ex}` — that's a test-internal probe, not user API.)
mlSimpleNoExSpelling : IO Bool
mlSimpleNoExSpelling = do
  lin <- runInit (the (Init (Linear 3 2 Ex F WithGrad)) linear)
  x   <- the (IO (Tensor [2, 3] Ex F NoGrad)) (tensor (Const 1.0))
  out <- Control.Linear.LIO.run (do
           (MkBang o # m') <- forward {b=2} lin (retypeGrad x)
           discard m'
           pure o)
  let v = primItem2d {ex=Ex} out.tensorPtr 0 0
  check "ML.Simple: import + Ex/F build & run a Linear with no {ex=}" (v == v)

export
tests : List (IO Bool)
tests = [mlSimpleNoExSpelling]
