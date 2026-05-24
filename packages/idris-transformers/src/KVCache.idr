||| Per-layer KV cache for incremental decoder-only generation.
|||
||| In greedy decoding without a cache, each step re-runs the forward
||| on the *full growing sequence*, which means K and V are re-projected
||| from input embeddings for positions [0..len-1] on every step. That's
||| O(n²) compute over a generation budget of N tokens. With a per-layer
||| KV cache holding `K, V : Tensor [len, kvOut]`, step k computes only
||| the new token's K and V (shape `[1, kvOut]`), appends them to the
||| cached pair, and attention runs with Q `[1, kvOut]` against the
||| full-history K/V — O(n) compute per step.
|||
||| Storage shape: flat 2D `[len, kvOut]` (where `kvOut = numKvHeads *
||| headDim`) to match `applyAttention`'s K/V projection outputs. SDPA
||| consumes the same layout, so no rank-change is needed at the
||| consumer side.
|||
||| Append strategy: functional concat via `tconcat2dAxis0`. Each step
||| allocates a fresh `[len + s, kvOut]` tensor; the old `[len, kvOut]`
||| becomes garbage. For a 64-token generation that's ~2k alloc-free
||| pairs of K/V tensors but only ~25 MB total churn on Llama 3.2 1B
||| (kvOut=512 × 4B × 64 ≈ 130 KB per step, summed). Cheap. A future
||| `primSliceWrite` could pre-allocate `[maxLen, kvOut]` once and write
||| into slots, but that's optimization on top of correctness.
|||
||| This is the first stateful inference cache in the codebase; the API
||| is deliberately minimal (Empty / Filled, two functions) so future
||| adapters (HfGpt2, HfBitNet) can adopt the same shape without a
||| refactor. K and V keep separate handles rather than a single
||| concatenated tensor — that matches PyTorch's `past_key_values`
||| layout and gives the user-facing genLoop a natural place to release
||| both on early-stop.
module KVCache

import Data.Vect

import Device
import Tensor


----------------------------------------------------------------------
-- KVCache — Empty / Filled tag union
----------------------------------------------------------------------

||| Per-layer K/V cache. `Empty` is the seed state (no tokens cached
||| yet); `Filled` carries the current cached prefix's K and V plus
||| its length. The `len` field is duplicated in the Tensor's first
||| dim — it's tracked alongside as a value-level Nat so the cache-
||| consumer can read it without pattern-matching the Tensor shape
||| (and so `appendKV` can compute the next-step `len + s` without
||| inspecting the new Tensor's implicit `s`).
public export
data KVCache : (kvOut : Nat) -> (0 d : Device) -> (0 dt : DType) -> Type where
  ||| Seed state: no tokens cached. Constructed at the start of
  ||| generation (one per layer).
  Empty  : KVCache kvOut d dt
  ||| Populated cache: K and V each `[len, kvOut]`, with `len > 0` by
  ||| construction (callers always reach this via `appendKV`).
  Filled : (len : Nat) ->
           (k : Tensor [len, kvOut] d dt NoGrad) ->
           (v : Tensor [len, kvOut] d dt NoGrad) ->
           KVCache kvOut d dt


||| Empty cache constructor — used when seeding a per-layer cache at
||| the start of generation. The `kvOut` parameter is left to be
||| inferred from context (e.g. a `Vect numLayers (KVCache kvOut d dt)`
||| ascription).
public export
emptyKVCache : KVCache kvOut d dt
emptyKVCache = Empty


||| Current cached prefix length. Returns 0 for `Empty`, the stored
||| `len` for `Filled`.
public export
cacheLen : KVCache kvOut d dt -> Nat
cacheLen Empty            = 0
cacheLen (Filled len _ _) = len


||| Append new K and V chunks to the cache along axis 0 (the sequence
||| axis). For `Empty` this simply wraps the new K/V; for `Filled` it
||| allocates fresh `[len + s, kvOut]` tensors via `tconcat2dAxis0` and
||| returns the new `Filled` state. The previous cache's K/V tensors
||| become garbage and are reclaimed by the next handle drain.
|||
||| Caller is responsible for ensuring `newK` and `newV` have already
||| had RoPE applied at the correct position offset (= `cacheLen`
||| before append). The cache stores the post-RoPE values; reading
||| them back during SDPA is the consumer's path.
public export
appendKV : {0 d : Device} -> UserDeviceTraining d =>
           {s, kvOut : Nat} ->
           KVCache kvOut d dt ->
           (newK : Tensor [s, kvOut] d dt NoGrad) ->
           (newV : Tensor [s, kvOut] d dt NoGrad) ->
           IO (KVCache kvOut d dt)
appendKV Empty newK newV = pure (Filled s newK newV)
appendKV (Filled len k v) newK newV = do
  k' <- tconcat2dAxis0 k newK
  v' <- tconcat2dAxis0 v newV
  pure (Filled (len + s) k' v')
