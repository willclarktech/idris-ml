module Layer.SwiGLU

import Data.Vect

import Executor
import Layer.Core
import Tensor


----------------------------------------------------------------------
-- SwiGLU — Llama's gated SiLU MLP block
----------------------------------------------------------------------
--
-- Composite of three (bias-free) linear projections and one SiLU:
--
--   gate = gate_proj(x)       # [intermediate]
--   up   = up_proj(x)         # [intermediate]
--   mid  = silu(gate) * up    # [intermediate], elementwise
--   out  = down_proj(mid)     # [hidden]
--
-- Llama (modeling_llama.py) ships `gate_proj` / `up_proj` /
-- `down_proj` as `nn.Linear(..., bias=False)`, so the state record
-- holds three plain weight tensors with no bias — matches HF's
-- on-disk shape (CONVENTIONS rule 2 for HfLlama, which re-uses this
-- layer at the typed surface).
--
-- For LayerLike's `applyVar`-shape (i = o = hidden), the
-- `intermediate` dim is a layer-internal knob (typically 4 × hidden
-- in classic transformers, 8/3 × hidden in Llama with the FFN-mult
-- adjustment to keep param count roughly equal to a 4 × hidden GeLU
-- FFN). Not exposed at the LayerLike level — pin it at construction.

public export
record SwiGLUState
        (hidden : Nat) (intermediate : Nat)
        (0 d : Executor) (0 dt : DType) (0 g : GradMode) where
  constructor MkSwiGLU
  gateW : Tensor [intermediate, hidden] d dt g
  upW   : Tensor [intermediate, hidden] d dt g
  downW : Tensor [hidden, intermediate] d dt g


----------------------------------------------------------------------
-- Forward
----------------------------------------------------------------------

%default partial

||| 1D forward — single input vector `[hidden]` to single output vector
||| `[hidden]`. Used by `applyVar`. The `applyVarBatch` path can build
||| on this manually for now (a 2D batched version would compose
||| `tlinear2d` + `tsilu` + `tmul` + `tlinear2d` instead of `tmv`).
export
applySwiGLU : {0 d : Executor} -> UserExecutorTraining d => UserExecutorCore d =>
              {hidden, intermediate : Nat} ->
              SwiGLUState hidden intermediate d dt g ->
              Tensor [hidden] d dt g ->
              IO (SwiGLUState hidden intermediate d dt g, Tensor [hidden] d dt g)
applySwiGLU st@(MkSwiGLU gateW upW downW) input = do
  gate <- tmv gateW input               -- [intermediate]
  up   <- tmv upW   input               -- [intermediate]
  sg   <- tsilu gate                    -- [intermediate]
  mid  <- tmul sg up                    -- [intermediate] elementwise
  out  <- tmv downW mid                 -- [hidden]
  pure (st, out)


----------------------------------------------------------------------
-- Constructor
----------------------------------------------------------------------

||| Build a `SwiGLUState hidden intermediate` with all three weight
||| tensors initialised N(0, 1/sqrt(fan_in)) — PyTorch nn.Linear's
||| default re-expressed as a normal distribution (was xavier-uniform
||| pre-P3; switched in lockstep with `linearLayer`'s default). Registers
||| as `<prefix>_gate_weight`, `<prefix>_up_weight`, `<prefix>_down_weight`.
||| HF-aligned modules (HfLlama) re-bind at construction to e.g.
||| `model.layers.{i}.mlp.gate_proj.weight`.
export
swigluLayer : UserExecutorTraining d => RuntimeDType dt => Linked d => Compatible d dt =>
              {hidden, intermediate : Nat} -> (paramPrefix : String) ->
              IO (SwiGLUState hidden intermediate d dt WithGrad)
swigluLayer paramPrefix = do
  -- fan_in is hidden for gate/up (W: [intermediate, hidden]) and
  -- intermediate for down (W: [hidden, intermediate]).
  let stdH = 1.0 / sqrt (cast {to=Double} hidden)
      stdI = 1.0 / sqrt (cast {to=Double} intermediate)
  gateW <- tparam2dNormal {o=intermediate} {i=hidden}       (paramPrefix ++ "_gate_weight") 0.0 stdH
  upW   <- tparam2dNormal {o=intermediate} {i=hidden}       (paramPrefix ++ "_up_weight")   0.0 stdH
  downW <- tparam2dNormal {o=hidden}       {i=intermediate} (paramPrefix ++ "_down_weight") 0.0 stdI
  pure (MkSwiGLU gateW upW downW)


----------------------------------------------------------------------
-- LayerLike instance
----------------------------------------------------------------------

-- LayerLike is parameterised by a state type with signature
-- `Nat -> Nat -> Executor -> DType -> GradMode -> Type`. SwiGLUState
-- carries an additional `intermediate` knob; collapse it via a
-- wrapper that pins one of the two "input/output" Nats to be the
-- intermediate and exposes only (hidden, hidden) to LayerLike — same
-- shape as Dropout / LayerNorm / RmsNorm which all use i = o = n.
public export
data SwiGLUStateAnyI : (hidden : Nat) -> (sameHidden : Nat) ->
                      (0 d : Executor) -> (0 dt : DType) -> (0 g : GradMode) -> Type where
  MkSwiGLUAnyI : (intermediate : Nat) ->
                 SwiGLUState hidden intermediate d dt g ->
                 SwiGLUStateAnyI hidden hidden d dt g


public export
LayerLike SwiGLUStateAnyI where
  applyVar (MkSwiGLUAnyI intermediate sw) input = do
    (sw', out) <- applySwiGLU sw input
    pure (MkSwiGLUAnyI intermediate sw', out)
  layerPrefix _ = "swiglu"

  freezeLayer (MkSwiGLUAnyI intermediate (MkSwiGLU g u d)) = do
    g' <- weakenGrad g
    u' <- weakenGrad u
    d' <- weakenGrad d
    pure (MkSwiGLUAnyI intermediate (MkSwiGLU g' u' d'))

  unfreezeLayer (MkSwiGLUAnyI intermediate (MkSwiGLU g u dn)) = do
    primIO (primSetRequiresGrad {d} g.tensorPtr 1)
    primIO (primSetRequiresGrad {d} u.tensorPtr 1)
    primIO (primSetRequiresGrad {d} dn.tensorPtr 1)
    pure (MkSwiGLUAnyI intermediate (MkSwiGLU (retypeGrad g) (retypeGrad u) (retypeGrad dn)))

||| Wrap a SwiGLU in `AnyLayer`. The `intermediate` knob is fixed at
||| construction; the LayerLike surface only sees (hidden, hidden).
export
swigluLayerAny : UserExecutorTraining d => RuntimeDType dt => Linked d => Compatible d dt =>
                 {hidden, intermediate : Nat} -> (paramPrefix : String) ->
                 IO (AnyLayer hidden hidden d dt WithGrad)
swigluLayerAny {intermediate} pid = do
  sw <- swigluLayer {hidden} {intermediate} pid
  pure (MkAnyLayer SwiGLUStateAnyI (MkSwiGLUAnyI intermediate sw))
