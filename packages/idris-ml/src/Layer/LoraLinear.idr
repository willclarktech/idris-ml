module Layer.LoraLinear

import Data.Vect

import Executor
import Layer.Linear
import Tensor


----------------------------------------------------------------------
-- LoraLinear — Low-rank adaptation wrapper (Hu et al. 2021)
----------------------------------------------------------------------
--
-- Wraps a base `LinearState i o` with two trainable low-rank
-- adapters: `A : [r, i]` (Gaussian init, std = 1/sqrt(r) per peft's
-- canonical convention) and `B : [o, r]` (zero init). Forward is
--
--     y = W·x + b + (α/r) · B · (A · x)
--
-- where (W, b) are the base linear's weight + bias and (A, B) are
-- trained. At t=0, B = 0 so the LoRA contribution is identically
-- zero — the wrapped layer is bit-identical to the bare base. As
-- training progresses, the LoRA branch accumulates a rank-r update
-- to the effective weight without ever materialising W + (α/r)·B·A.
--
-- The base `LinearState` is reused AS-IS (its weight + bias keep
-- their existing paramIds in the C-side registry). `mkLoraLinear`
-- only registers `A` and `B` under `<paramPrefix>.lora_A` /
-- `<paramPrefix>.lora_B`, matching peft's adapter-key convention
-- (modulo the `base_model.model.*.default.weight` peft decorations,
-- which the L4 adapter-IO layer adds at safetensors write time).
--
-- Freezing the base for LoRA-only training is the caller's
-- responsibility — typically `freezeByPrefix opt "<basePrefix>"`
-- (after L2's `freezeBySuffix`/`unfreezeBySuffix` lands, the
-- canonical LoRA setup is `freezeByPrefix opt "bert."` followed
-- by `unfreezeBySuffix opt "lora_A"` + `unfreezeBySuffix opt
-- "lora_B"` to keep adapters trainable).
--
-- Note (L1): no `LayerLike` instance. The interface takes two Nat
-- params (`l : Nat -> Nat -> ... -> Type`); LoraLinear has three
-- (i, o, r). LoraLinear is used by composition inside HfBert's
-- attention forward (L3), not as a node in a generic `Network`
-- chain — so the 3-arity fit doesn't bite here. A
-- `LoraLayer i o ex dt g` newtype with an existentially-packed
-- rank could be added later if a generic-network use case appears.

public export
record LoraLinearState (i : Nat) (o : Nat)
                       (0 ex : Executor) (0 dt : DType) (0 g : GradMode) where
  constructor MkLoraLinear
  {rank      : Nat}                       -- implicit, runtime-available
  baseLinear : LinearState i o ex dt g
  loraA      : Tensor [rank, i] ex dt g
  loraB      : Tensor [o, rank] ex dt g
  alpha      : Double


----------------------------------------------------------------------
-- Forward
----------------------------------------------------------------------

%default partial

||| LoRA forward: y = W·x + b + (α/r) · B · (A · x).
|||
||| Computed as four FFI hops (`tlinear` + `tmv` + `tmv` +
||| `tmulScalar` + `tadd`). The LoRA branch keeps A and B split so
||| total work is `r·(i+o)` rather than the `o·i` of a materialised
||| W + (α/r)·B·A — the rank-r savings are what make LoRA cheap on
||| large attention projections.
export
applyLoraLinear : {0 ex : Executor} -> Backend ex dt => {0 g : GradMode} -> {i, o : Nat}
              -> LoraLinearState i o ex dt g
              -> Tensor [i] ex dt g
              -> IO (Tensor [o] ex dt g)
applyLoraLinear (MkLoraLinear {rank} base loraA loraB alpha) input = do
  baseOut <- tlinear base.weightT input base.biasT
  aOut    <- tmv loraA input
  bOut    <- tmv loraB aOut
  scaled  <- tmulScalar bOut (alpha / cast rank)
  tadd baseOut scaled


----------------------------------------------------------------------
-- Constructor
----------------------------------------------------------------------

||| Wrap an already-registered `LinearState i o` with trainable LoRA
||| adapters. Registers two new params under
|||
|||     <paramPrefix>.lora_A    (Gaussian, std = 1/sqrt(rank))
|||     <paramPrefix>.lora_B    (zero)
|||
||| The zero B init is load-bearing: it makes the t=0 LoRA delta
||| identically zero, so the wrapped layer's forward is bit-identical
||| to the bare base at initialisation. The Gaussian A std matches
||| peft's `LoraConfig` default (`init_lora_weights="default"`,
||| which uses Kaiming uniform; the equivalent N(0, 1/sqrt(r)) is
||| approximately equivalent for `r=8..16` and matches the original
||| LoRA paper's stated init).
|||
||| The base `LinearState` is taken AS-IS — its weight + bias keep
||| their existing paramIds in the C-side registry. Do NOT pass a
||| fresh Linear here unless the base really should re-register
||| (per `feedback_param_registry_dedup`, `param_register` REPLACES
||| existing entries by name).
export
mkLoraLinear : {0 ex : Executor} -> Backend ex dt
            => {i, o : Nat}
            -> (paramPrefix : String)
            -> (rank : Nat)
            -> (alpha : Double)
            -> (baseLinear : LinearState i o ex dt WithGrad)
            -> IO (LoraLinearState i o ex dt WithGrad)
mkLoraLinear pfx rank alpha base = do
  a <- tparam2dNormal {ex} {dt} {o=rank} {i=i}
                      (pfx ++ ".lora_A") 0.0 (1.0 / sqrt (cast rank))
  b <- tparam2dConst  {ex} {dt} {o=o} {i=rank}
                      (pfx ++ ".lora_B") 0.0
  pure (MkLoraLinear {rank} base a b alpha)


----------------------------------------------------------------------
-- Param-name helpers
----------------------------------------------------------------------

||| The two paramIds registered by `mkLoraLinear pfx ...`. Useful for
||| callers that want to freeze / unfreeze / inspect adapter params
||| by exact name (rather than via `bySuffix`-style filtering).
export
loraParamNames : (paramPrefix : String) -> (String, String)
loraParamNames pfx = (pfx ++ ".lora_A", pfx ++ ".lora_B")
