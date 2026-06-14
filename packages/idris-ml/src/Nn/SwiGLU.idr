||| `SwiGLU` — Llama's gated-SiLU MLP: `down(silu(gate(x)) * up(x))`, three
||| bias-free projections. Params-only (the forward is 1-D `tmv`-based; the
||| projections are bias-free so they don't go through `tlinear2d`). Its
||| `(hidden, intermediate)` indices match `Params`' kind directly, so it
||| gets `Params` + the generic `freeze` for free; composed inside a
||| transformer block at the example level.
module Nn.SwiGLU

import Data.Vect

import Executor
import Tensor
import Nn.Init
import Nn.Module

%default total

||| Gated-SiLU MLP with three weight tensors (gate/up: [intermediate,
||| hidden]; down: [hidden, intermediate]). No bias (Llama convention).
public export
record SwiGLU (hidden : Nat) (intermediate : Nat) (0 ex : Executor) (0 dt : DType) (0 g : GradMode) where
  constructor MkSwiGLU
  gateW : Tensor [intermediate, hidden] ex dt g
  upW   : Tensor [intermediate, hidden] ex dt g
  downW : Tensor [hidden, intermediate] ex dt g

public export
Params SwiGLU where
  params (MkSwiGLU g u d) = [toParam g, toParam u, toParam d]
  castGrad (MkSwiGLU g u d) = MkSwiGLU (retypeGrad g) (retypeGrad u) (retypeGrad d)

||| 1-D SwiGLU forward: `down(silu(gate·x) * (up·x))`.
export
swigluForward : {0 ex : Executor} -> Backend ex dt => {0 g : GradMode} -> {hidden, intermediate : Nat} ->
                SwiGLU hidden intermediate ex dt g -> Tensor [hidden] ex dt g -> IO (Tensor [hidden] ex dt g)
swigluForward (MkSwiGLU gateW upW downW) input = do
  gate <- tmv gateW input
  up   <- tmv upW   input
  sg   <- tsilu gate
  mid  <- tmul sg up
  tmv downW mid

||| Construct a `SwiGLU hidden intermediate` inside an `Init` derivation.
||| Registers HF-style `<scope>.swiglu_<n>.{gate,up,down}_proj.weight`,
||| each ~ N(0, 1/√fan_in).
export
swiglu : {0 ex : Executor} -> Backend ex dt => {hidden, intermediate : Nat} ->
         Init (SwiGLU hidden intermediate ex dt WithGrad)
swiglu = do
  name <- freshChild "swiglu"
  let stdH = 1.0 / sqrt (cast {to=Double} hidden)
      stdI = 1.0 / sqrt (cast {to=Double} intermediate)
  g <- liftIO $ tparam2dNormal {ex} {dt} {o=intermediate} {i=hidden}       (name ++ ".gate_proj.weight") 0.0 stdH
  u <- liftIO $ tparam2dNormal {ex} {dt} {o=intermediate} {i=hidden}       (name ++ ".up_proj.weight")   0.0 stdH
  d <- liftIO $ tparam2dNormal {ex} {dt} {o=hidden}       {i=intermediate} (name ++ ".down_proj.weight") 0.0 stdI
  pure (MkSwiGLU g u d)
