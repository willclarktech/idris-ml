||| Generalized Advantage Estimation (Schulman et al. 2015).
|||
||| Used by A2C and PPO. Pure function — no IO, no tensors (just Doubles on
||| the value/reward level). Caller feeds per-step (reward, value, done)
||| triples plus the bootstrap value V(s_T) of the post-trajectory state
||| (zero if the trajectory terminates naturally, else the critic's
||| estimate at the final next-state).
|||
||| Recursion:
|||   δ_t      = r_t + γ · V(s_{t+1}) · mask_t − V(s_t)
|||   A_t      = δ_t + γ · λ · mask_t · A_{t+1}
|||   G_t      = A_t + V(s_t)           -- return target for the critic
||| where mask_t = 1 − done_t zeros out the contribution from steps after
||| a terminal state. A_T = 0 by convention.
module RL.Gae

||| Compute GAE advantages and return targets from a trajectory.
|||
||| @gamma           discount factor (e.g. 0.99)
||| @lam             GAE lambda (e.g. 0.95)
||| @bootstrapValue  V(s_T) at the post-trajectory state (0 if terminated)
||| @steps           per-step (reward, value, done) in forward order
|||
||| Returns per-step (advantage, return_target) in forward order.
export
gae : (gamma : Double) -> (lam : Double) ->
      (bootstrapValue : Double) ->
      List (Double, Double, Bool) -> List (Double, Double)
gae gamma lam bootstrapValue steps =
  reverse (go (reverse steps) bootstrapValue 0.0)
  where
    go : List (Double, Double, Bool) -> Double -> Double -> List (Double, Double)
    go [] _ _ = []
    go ((r, v, d) :: rest) nextV nextA =
      let mask : Double
          mask = if d then 0.0 else 1.0
          delta  = r + gamma * nextV * mask - v
          a      = delta + gamma * lam * mask * nextA
          retT   = a + v
      in (a, retT) :: go rest v a
