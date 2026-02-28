module Test.Init

import Harness
import Init


tol : Double
tol = 1.0e-6

export
tests : List (IO Bool)
tests =
  [ -- xavierInit
    checkClose "xavierInit 10 10" (Prelude.sqrt (6.0 / 20.0)) (xavierInit 10 10) tol

  -- heInit
  , checkClose "heInit 10 10" (Prelude.sqrt (6.0 / 10.0)) (heInit 10 10) tol

  -- lecunInit
  , checkClose "lecunInit 10 10" (Prelude.sqrt (3.0 / 10.0)) (lecunInit 10 10) tol

  -- uniformInit returns constant
  , checkClose "uniformInit 1.0" 1.0 (uniformInit 1.0 5 10) tol
  ]
