module Test.RL.Gae

import Harness
import RL.Gae


-- Reference values computed by hand.
-- Inputs: rewards=[1,1,1], values=[0.5,0.5,0.5], dones=[F,F,T], bootstrap=0,
--         gamma=0.99, lambda=0.95.
--
-- t=2 (done): delta = 1 + 0.99*0*0 - 0.5 = 0.5; A = 0.5; G = 1.0.
-- t=1:        delta = 1 + 0.99*0.5 - 0.5 = 0.995;
--             A = 0.995 + 0.99*0.95*1*0.5 = 0.995 + 0.47025 = 1.46525;
--             G = 1.96525.
-- t=0:        delta = 0.995; A = 0.995 + 0.99*0.95*1*1.46525 = 2.3729 approx;
--             G = 2.8729 approx.

tol : Double
tol = 1.0e-6


export
tests : List (IO Bool)
tests =
  [ check "gae on empty trajectory returns empty" $
      gae 0.99 0.95 0.0 [] == []

  , -- 3-step trajectory, last step terminal
    let trajectory = [(1.0, 0.5, False), (1.0, 0.5, False), (1.0, 0.5, True)]
        result = gae 0.99 0.95 0.0 trajectory
    in check "gae trajectory length preserved" (length result == 3)

  , -- Terminal step: advantage = r - v when done, no bootstrap contribution
    let trajectory = [(1.0, 0.5, False), (1.0, 0.5, False), (1.0, 0.5, True)]
        result = gae 0.99 0.95 0.0 trajectory
    in case result of
         [_, _, (a, g)] =>
           check "terminal-step advantage and return"
             (abs (a - 0.5) < tol && abs (g - 1.0) < tol)
         _ => check "unexpected trajectory length" False

  , -- Middle step: delta with bootstrap of next value
    let trajectory = [(1.0, 0.5, False), (1.0, 0.5, False), (1.0, 0.5, True)]
        result = gae 0.99 0.95 0.0 trajectory
    in case result of
         [_, (a, g), _] =>
           check "middle-step GAE advantage"
             (abs (a - 1.46525) < tol && abs (g - 1.96525) < tol)
         _ => check "unexpected trajectory length" False

  , -- First step: discounted + λ-weighted advantage propagation
    let trajectory = [(1.0, 0.5, False), (1.0, 0.5, False), (1.0, 0.5, True)]
        result = gae 0.99 0.95 0.0 trajectory
    in case result of
         [(a, g), _, _] =>
           let expectedA = 0.995 + 0.99 * 0.95 * 1.46525
               expectedG = expectedA + 0.5
           in check "first-step GAE propagation"
                (abs (a - expectedA) < tol && abs (g - expectedG) < tol)
         _ => check "unexpected trajectory length" False

  , -- Bootstrap value: if the trajectory is truncated (not terminated),
    -- the last-step advantage picks up γ·V(s_T) from the bootstrap.
    let trajectory = [(1.0, 0.5, False)]
        result = gae 0.99 0.95 2.0 trajectory
    in case result of
         [(a, g)] =>
           -- delta = 1 + 0.99*2 - 0.5 = 2.48; A = 2.48 (no nextA yet); G = 2.98
           check "bootstrap value applied on truncation"
             (abs (a - 2.48) < tol && abs (g - 2.98) < tol)
         _ => check "unexpected trajectory length" False

  , -- Zero rewards + zero values + terminated = zero advantages
    let trajectory = [(0.0, 0.0, False), (0.0, 0.0, True)]
        result = gae 0.99 0.95 0.0 trajectory
        allZero = all (\(a, g) => abs a < tol && abs g < tol) result
    in check "zero inputs produce zero advantages" allZero

  , -- lambda = 1 (MC returns): A_t = G_t - V_t, where G_t = Σ γ^k r_{t+k}
    -- rewards=[1,1,1], γ=1, λ=1, values=[0,0,0], terminal done at end.
    -- G_2 = 1, G_1 = 2, G_0 = 3. A_t = G_t - 0 = G_t.
    let trajectory = [(1.0, 0.0, False), (1.0, 0.0, False), (1.0, 0.0, True)]
        result = gae 1.0 1.0 0.0 trajectory
    in case result of
         [(a0, g0), (a1, g1), (a2, g2)] =>
           check "λ=1, γ=1: advantages match MC returns"
             (abs (a0 - 3.0) < tol && abs (a1 - 2.0) < tol && abs (a2 - 1.0) < tol &&
              abs (g0 - 3.0) < tol && abs (g1 - 2.0) < tol && abs (g2 - 1.0) < tol)
         _ => check "unexpected trajectory length" False

  , -- lambda = 0 (TD(0)): A_t = delta_t, no λ-weighted propagation.
    let trajectory = [(1.0, 0.5, False), (1.0, 0.5, True)]
        result = gae 0.99 0.0 0.0 trajectory
    in case result of
         [(a0, _), (a1, _)] =>
           -- delta_1 = 1 - 0.5 = 0.5; delta_0 = 1 + 0.99*0.5 - 0.5 = 0.995
           check "λ=0: advantages are TD residuals"
             (abs (a1 - 0.5) < tol && abs (a0 - 0.995) < tol)
         _ => check "unexpected trajectory length" False
  ]
