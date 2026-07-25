||| Distributions over a `Source` — the shapes a caller actually wants, built
||| from raw uniforms so every one of them replays.
module Random.Dist

import Random.Source

%default total

||| Uniform on [lo, hi).
export
uniform : Source -> (lo : Double) -> (hi : Double) -> (Double, Source)
uniform s lo hi =
  let (d, s') = nextDouble s
  in (lo + d * (hi - lo), s')

||| The uniform that would have produced `x` on [lo, hi).
|||
||| `uniform` is affine, so it inverts exactly. That is what lets a recording
||| be made from observed *values* rather than from the draws behind them —
||| useful when the values came from somewhere else entirely, such as another
||| implementation's generator.
export
uniformInverse : (x : Double) -> (lo : Double) -> (hi : Double) -> Double
uniformInverse x lo hi = if hi == lo then 0.0 else (x - lo) / (hi - lo)

||| An index in [0, n), by scaling a uniform. Biased only by the rounding of
||| `floor`, which is below what any consumer here can observe.
export
boundedNat : Source -> (n : Nat) -> (Nat, Source)
boundedNat s Z       = (Z, s)
boundedNat s n@(S _) =
  let (d, s') = nextDouble s
      scaled  = cast {to = Integer} (floor (d * cast n))
      capped  = if scaled >= cast n then cast n - 1 else scaled
  in (cast {to = Nat} capped, s')

||| Standard normal, N(0, 1), by Box-Muller. Consumes two uniforms.
export
normal : Source -> (Double, Source)
normal s =
  let (u1raw, s1) = nextDouble s
      (u2, s2)    = nextDouble s1
      -- log 0 is -inf; clamp the low end rather than let one unlucky draw
      -- poison the result.
      u1          = if u1raw < 1.0e-10 then 1.0e-10 else u1raw
      z           = prim__doubleSqrt (-2.0 * prim__doubleLog u1)
                      * prim__doubleCos (2.0 * 3.141592653589793 * u2)
  in (z, s2)

||| N(mu, sigma).
export
normalWith : Source -> (mu : Double) -> (sigma : Double) -> (Double, Source)
normalWith s mu sigma =
  let (z, s') = normal s
  in (mu + sigma * z, s')

||| Index of a categorical outcome, by inverse CDF over `probs`.
|||
||| Walks the cumulative sum and takes the first bucket the draw falls in. The
||| last index absorbs any shortfall from probabilities that do not quite sum
||| to one, so this is total for every input.
export
categorical : Source -> List Double -> (Nat, Source)
categorical s probs =
  let (u, s') = nextDouble s
  in (pick 0 0.0 probs u, s')
  where
    pick : Nat -> Double -> List Double -> Double -> Nat
    pick idx _     []          _ = idx
    pick idx _     [_]         _ = idx
    pick idx cumul (p :: rest) u =
      let cumul' = cumul + p
      in if u < cumul' then idx else pick (S idx) cumul' rest u
