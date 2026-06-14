||| `Nn.PosEncoding` — sinusoidal positional encoding (Vaswani et al 2017).
|||
||| Like `Nn.RoPE`, sinusoidal PE has NO learnable parameters, so this is a
||| parameter-free free function rather than a `Module`/`Params` instance.
||| It builds a `[seqLen, dModel]` table once (host-side Double math, uploaded
||| as a persistent-state tensor via `dtCreateState2d` so it survives
||| `tape_reset` between forward passes) and the caller adds it to the token
||| embeddings.
|||
||| The table is polymorphic in `GradMode`: it carries no paramId so the
||| optimizer never touches it whatever `g` is, but the result must share the
||| embedding's `g` to feed `tadd` — pin `g = WithGrad` at the call site to add
||| it to learnable embeddings. Relocated from `Layer.Transformer`'s private
||| `posEncVal`/`writePE` (which die with `Layer/` at the migration sweep).
module Nn.PosEncoding

import Control.Linear.LIO as LIO
import Data.Vect

import Executor
import Tensor

----------------------------------------------------------------------
-- Sinusoidal positional encoding (host-side Double math)
----------------------------------------------------------------------

||| The sinusoidal value at `(pos, dim)` for model width `dModel`:
|||   PE[pos, 2i]   = sin(pos / 10000^(2i/dModel))
|||   PE[pos, 2i+1] = cos(pos / 10000^(2i/dModel))
|||
||| Casts `Nat` args to `Int` before `div`/`mod`: the stdlib
||| `Data.Nat.divNat`/`modNatNZ` compile to recursive Peano walks even
||| though `Nat` is `Integer` underneath (the `posEncVal` perf incident,
||| `docs/develop/gotchas.md`). Plain `Int div`/`mod` are single CPU ops.
export
posEncVal : (dModel : Nat) -> (pos : Nat) -> (dim : Nat) -> Double
posEncVal dModel pos dim =
  let dimI  = the Int (cast dim)
      p     = cast {to=Double} pos
      i     = cast {to=Double} (dimI `div` 2)
      dm    = cast {to=Double} dModel
      angle = p / pow 10000.0 (2.0 * i / dm)
  in if (dimI `mod` 2) == 0 then sin angle else cos angle

-- Fill a row-major [sLen, dMod] buffer with the sinusoidal table.
-- Recursive over (pos, dim) — Int-indexed → partial.
partial
writePE : (dModel : Nat) -> AnyPtr -> (pos, dim, sLen, dMod : Int) -> AnyPtr
writePE dModel buf pos dim sLen dMod =
  if pos >= sLen then buf
  else if dim >= dMod then writePE dModel buf (pos + 1) 0 sLen dMod
  else let val  = posEncVal dModel (cast pos) (cast dim)
           buf' = prim__setDouble buf (pos * dMod + dim) val
       in writePE dModel buf' pos (dim + 1) sLen dMod

----------------------------------------------------------------------
-- Table builder
----------------------------------------------------------------------

||| Build the `[seqLen, dModel]` sinusoidal positional-encoding table as a
||| persistent-state tensor (`dtCreateState2d`, so it survives `tape_reset`
||| between forwards). Carries no paramId — invisible to the optimizer —
||| and is polymorphic in `g`; pin `g = WithGrad` to add it to learnable
||| token embeddings via `tadd`.
export partial
sinusoidalPE : {0 ex : Executor} -> Backend ex dt => {seqLen, dModel : Nat} ->
               IO (Tensor [seqLen, dModel] ex dt g)
sinusoidalPE = ioRerun (\_ =>
  let sI   = cast {to=Int} seqLen
      dI   = cast {to=Int} dModel
      buf  = prim__allocDoubles (sI * dI)
      buf' = writePE dModel buf 0 0 sI dI
      ptr  = dtCreateState2d {ex} {t=dt} sI dI buf' (deviceStreamTag {ex})
  in MkTensor ptr Nothing)

||| `L IO` twin of `sinusoidalPE`, for building the PE table inside a model
||| `forward` / `runInitL` block without a `liftIO1` seam at the call site.
||| Same deferral semantics (`liftIO1` over the `ioRerun`-deferred body =
||| `ioRerunL` of that body); kept a free function, not a `Module` — the
||| table holds no learnable parameter.
export partial
sinusoidalPEL : {0 ex : Executor} -> Backend ex dt => {seqLen, dModel : Nat} ->
                LIO.L IO (Tensor [seqLen, dModel] ex dt g)
sinusoidalPEL = liftIO1 sinusoidalPE
