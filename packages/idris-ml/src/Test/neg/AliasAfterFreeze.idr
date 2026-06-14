||| Negative compile test: confirms that the linear-types discipline
||| on `freezeNetwork` rejects reuse of the original WithGrad-typed
||| Network reference after freezing. This file MUST NOT type-check.
|||
||| The bug class this catches: a user freezes a network, retains the
||| original `Network ... WithGrad` reference, and feeds it back into
||| training. The C-side `requires_grad=false` flags on the params
||| make the training silently no-op — same bug class the entire
||| GradMode refactor exists to prevent, just dressed up as freezing.
|||
||| Linear `freezeNetwork : (1 _ : Network ... g) -> IO (Network ... NoGrad)`
||| consumes the input reference. Trying to use the same name after the
||| call is a linearity error at compile time. If this file ever starts
||| to compile, the aliasing footgun has come back.

module AliasAfterFreeze

import Executor
import Layer
import Tensor

-- A parameter-free Network (single tanh activation) so we don't need
-- RNG to construct it.
buildNet : Network 4 [] 4 TapeExecutor dt WithGrad
buildNet = OutputLayer (the (AnyLayer 4 4 TapeExecutor dt WithGrad) tanhLayerAny)

-- ^^^ EXPECTED COMPILE ERROR: after `freezeNetwork net`, the variable
-- `net` has 0 remaining uses (consumed linearly). The `forwardVar net
-- ...` call below tries to use it a second time — Idris reports a
-- linearity violation. Error message includes "linear" or "There are
-- 0 uses" or similar.
brokenReuse : (input : Tensor [4] TapeExecutor dt WithGrad) -> IO ()
brokenReuse input = do
  let net = buildNet
  _ <- freezeNetwork net           -- consumes `net`
  let (_, _) = forwardVar net input -- ERROR: net already consumed
  pure ()
