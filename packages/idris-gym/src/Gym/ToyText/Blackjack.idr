module Gym.ToyText.Blackjack

import Data.Vect

import Gym.Env
import Gym.Rng

----------------------------------------------------------------------
-- Blackjack-v1 (Gymnasium-compatible, natural=False by default)
--
-- Player plays against dealer:
--   1. Deal 2 cards each (dealer's first card visible).
--   2. Player hits (action 1) or sticks (action 0) until stick or bust.
--   3. Dealer then draws until >= 17 (sticks on soft 17 by default).
--   4. Score: +1 win, -1 lose, 0 draw.
--
-- Card values 1..10, face cards = 10. Ace = 11 if the total stays <=21,
-- else 1 (this is "usable ace" logic).
--
-- Observation: (player_sum, dealer_showing_card, usable_ace_as_0/1).
-- Packed as a 3-element Vect.
----------------------------------------------------------------------

||| Blackjack state. RNG-backed card deck (infinite with replacement).
public export
record BJState where
  constructor MkBJ
  bjPlayer : List Nat
  bjDealer : List Nat   -- first card is visible; whole hand resolved at end
  bjDone   : Bool
  bjSeed   : Seed

||| Draw one card from a uniform 13-card suit (canonical Gymnasium
||| Blackjack-v1 distribution). Values: 1=Ace (1/13), 2..9 (1/13 each),
||| 10 (4/13 from 10, J, Q, K).
drawCard : Seed -> (Nat, Seed)
drawCard s =
  let (n, s') = nextNat s 13
  in case n of
       0 => (1, s')       -- Ace
       1 => (2, s')
       2 => (3, s')
       3 => (4, s')
       4 => (5, s')
       5 => (6, s')
       6 => (7, s')
       7 => (8, s')
       8 => (9, s')
       _ => (10, s')      -- 10, J, Q, K (n=9..12)

sumList : List Nat -> Nat
sumList []        = Z
sumList (x :: xs) = x + sumList xs

-- Promote aces from 1 to 11 while the total stays <= 21.
upgradeAces : Nat -> Nat -> Nat
upgradeAces tot Z     = tot
upgradeAces tot (S a) =
  if tot + 10 <= 21 then upgradeAces (tot + 10) a else tot

-- Sum a hand, treating aces as 11 if it keeps total <= 21.
handSum : List Nat -> Nat
handSum cards = upgradeAces (sumList cards) (length (filter (== 1) cards))

-- Whether the hand has a "usable ace" (one counted as 11).
usableAce : List Nat -> Bool
usableAce cards =
  let naive = foldr (+) Z cards
      aces  = length (filter (== 1) cards)
  in aces > 0 && naive + 10 <= 21

isBust : Nat -> Bool
isBust n = n > 21

||| Deal initial hands (2 cards each). The input Seed both seeds the
||| internal deck and produces an advanced caller-side Seed (the deck
||| post-deal), so successive resets diverge.
export
initBJ : Seed -> (BJState, Seed)
initBJ seed =
  let (p1, s1) = drawCard seed
      (p2, s2) = drawCard s1
      (d1, s3) = drawCard s2
      (d2, s4) = drawCard s3
  in (MkBJ [p1, p2] [d1, d2] False s4, s4)

-- Dealer plays out: hits while sum < 17.
dealerPlay : List Nat -> Seed -> (List Nat, Seed)
dealerPlay dealer seed =
  if handSum dealer >= 17 then (dealer, seed)
  else let (c, seed') = drawCard seed
       in dealerPlay (c :: dealer) seed'

||| One step. Action 0 = stick, 1 = hit.
export
bjStep : BJState -> Nat -> (Double, BJState, Outcome, Info)
bjStep s action =
  if s.bjDone then (0.0, s, Terminated, [])
  else if action == 1 then bjHit s
  else bjStick s

  where
    bjHit : BJState -> (Double, BJState, Outcome, Info)
    bjHit st =
      let (c, seed') = drawCard st.bjSeed
          player' = c :: st.bjPlayer
          newSum  = handSum player'
          st'     = { bjPlayer := player', bjSeed := seed' } st
      in if isBust newSum
           then (-1.0, { bjDone := True } st', Terminated, [])
         else (0.0, st', Continue, [])

    bjStick : BJState -> (Double, BJState, Outcome, Info)
    bjStick st =
      let (dealer', seed') = dealerPlay st.bjDealer st.bjSeed
          st'    = { bjDealer := dealer', bjSeed := seed', bjDone := True } st
          pSum   = handSum st.bjPlayer
          dSum   = handSum dealer'
          reward = if isBust dSum then 1.0
                   else if pSum > dSum then 1.0
                   else if pSum == dSum then 0.0
                   else -1.0
      in (reward, st', Terminated, [])

||| Observation: [player_sum, dealer_showing, usable_ace_as_0_or_1].
export
bjObserve : BJState -> Vect 3 Double
bjObserve s =
  let p = cast {to=Double} (cast {to=Integer} (handSum s.bjPlayer))
      d = case s.bjDealer of
            (c :: _) => cast {to=Double} (cast {to=Integer} c)
            []       => 0.0
      u = if usableAce s.bjPlayer then 1.0 else 0.0
  in [p, d, u]

public export
Env BJState Nat (Vect 3 Double) where
  reset            = initBJ
  step             = bjStep
  observe          = bjObserve
  actionSpace      = Discrete 2
  obsSpace         = Box [4.0, 1.0, 0.0] [32.0, 11.0, 1.0]
  defaultTimeLimit = Nothing
