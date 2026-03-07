module Test.Layer

import Data.List
import Data.String
import Data.Vect
import System.Random

import Harness
import Floating
import Layer
import Memory
import Tensor
import Variable


----------------------------------------------------------------------
-- Helpers
----------------------------------------------------------------------

-- Collect all paramIds from a Tensor of Variables
tensorIds : {dims : Vect rank Nat} -> Tensor dims Variable -> List String
tensorIds = mapMaybe paramId . toList

mutual
  layerIds : {i, o : Nat} -> Layer i o Variable -> List String
  layerIds (LinearLayer w b _) = tensorIds w ++ tensorIds b
  layerIds (RnnLayer iw rw b _ _ _) = tensorIds iw ++ tensorIds rw ++ tensorIds b
  layerIds (LstmLayer iw rw b _ _ _ _) = tensorIds iw ++ tensorIds rw ++ tensorIds b
  layerIds (NtmLayer ctrl mem rh wh ro) =
    networkIds ctrl ++ tensorIds mem
      ++ tensorIds rh.addressingWeights
      ++ tensorIds wh.readHead.addressingWeights
      ++ tensorIds ro
  layerIds _ = []

  networkIds : {i, o : Nat} -> {hs : List Nat} -> Network i hs o Variable -> List String
  networkIds (OutputLayer l) = layerIds l
  networkIds (l ~> r) = layerIds l ++ networkIds r

hasPrefix : String -> List String -> Bool
hasPrefix pfx = any (isPrefixOf pfx)

noDuplicates : List String -> Bool
noDuplicates [] = True
noDuplicates (x :: xs) = not (elem x xs) && noDuplicates xs


----------------------------------------------------------------------
-- Tests
----------------------------------------------------------------------

export
tests : List (IO Bool)
tests =
  [ -- Single linear layer: ll0_weight*, ll0_bias*
    do srand 42
       ll <- linearLayer {i=2, o=3}
       let named = autoName $ OutputLayer ll
       let ids = networkIds named
       r1 <- check "autoName linear: has ll0_weight" (hasPrefix "ll0_weight" ids)
       r2 <- check "autoName linear: has ll0_bias" (hasPrefix "ll0_bias" ids)
       r3 <- check "autoName linear: 9 params" (length ids == 9)
       pure (r1 && r2 && r3)

  -- Two linear layers: ll0 and ll1, no collisions
  , do srand 42
       l1 <- linearLayer {i=2, o=3}
       l2 <- linearLayer {i=3, o=2}
       let named = autoName $ l1 ~> OutputLayer l2
       let ids = networkIds named
       r1 <- check "autoName two linear: has ll0_weight" (hasPrefix "ll0_weight" ids)
       r2 <- check "autoName two linear: has ll1_weight" (hasPrefix "ll1_weight" ids)
       r3 <- check "autoName two linear: no duplicates" (noDuplicates ids)
       r4 <- check "autoName two linear: 17 params" (length ids == 17)
       pure (r1 && r2 && r3 && r4)

  -- RNN layer: rnn0_inputWeight*, rnn0_bias*
  , do srand 42
       rnn <- rnnLayer {i=2, o=3}
       let named = autoName $ OutputLayer rnn
       let ids = networkIds named
       r1 <- check "autoName rnn: has rnn0_inputWeight" (hasPrefix "rnn0_inputWeight" ids)
       r2 <- check "autoName rnn: has rnn0_recurrentWeight" (hasPrefix "rnn0_recurrentWeight" ids)
       r3 <- check "autoName rnn: has rnn0_bias" (hasPrefix "rnn0_bias" ids)
       pure (r1 && r2 && r3)

  -- Mixed: linear ~> sigmoid ~> rnn -> ll0 and rnn0 prefixes
  , do srand 42
       ll <- linearLayer {i=2, o=3}
       rnn <- rnnLayer {i=3, o=2}
       let named = autoName $ ll ~> sigmoidLayer ~> OutputLayer rnn
       let ids = networkIds named
       r1 <- check "autoName mixed: has ll0_weight" (hasPrefix "ll0_weight" ids)
       r2 <- check "autoName mixed: has rnn0_inputWeight" (hasPrefix "rnn0_inputWeight" ids)
       r3 <- check "autoName mixed: no ll1" (not (hasPrefix "ll1_" ids))
       r4 <- check "autoName mixed: no duplicates" (noDuplicates ids)
       pure (r1 && r2 && r3 && r4)

  -- LSTM layer: lstm0_inputWeight*, lstm0_bias*
  , do srand 42
       lstm <- lstmLayer {i=2, o=3}
       let named = autoName $ OutputLayer lstm
       let ids = networkIds named
       r1 <- check "autoName lstm: has lstm0_inputWeight" (hasPrefix "lstm0_inputWeight" ids)
       r2 <- check "autoName lstm: has lstm0_recurrentWeight" (hasPrefix "lstm0_recurrentWeight" ids)
       r3 <- check "autoName lstm: has lstm0_bias" (hasPrefix "lstm0_bias" ids)
       r4 <- check "autoName lstm: no duplicates" (noDuplicates ids)
       pure (r1 && r2 && r3 && r4)

  -- LSTM + linear: no collisions
  , do srand 42
       lstm <- lstmLayer {i=2, o=3}
       ll <- linearLayer {i=3, o=2}
       let named = autoName $ lstm ~> OutputLayer ll
       let ids = networkIds named
       r1 <- check "autoName lstm+linear: has lstm0_" (hasPrefix "lstm0_" ids)
       r2 <- check "autoName lstm+linear: has ll0_" (hasPrefix "ll0_" ids)
       r3 <- check "autoName lstm+linear: no duplicates" (noDuplicates ids)
       pure (r1 && r2 && r3)

  -- NTM: controller layers get scoped names
  , do srand 42
       ctrlH <- linearLayer {i=6, o=4}
       ctrlO <- linearLayer {i=4, o=27}
       let ctrl = ctrlH ~> sigmoidLayer ~> OutputLayer ctrlO
       ntm <- ntmLayer {n=10, w=3} ctrl
       let named = autoName $ ntm ~> OutputLayer logSoftmaxLayer
       let ids = networkIds named
       r1 <- check "autoName ntm: has ntm0_mem" (hasPrefix "ntm0_mem" ids)
       r2 <- check "autoName ntm: has ntm0_rAddr" (hasPrefix "ntm0_rAddr" ids)
       r3 <- check "autoName ntm: has ntm0_wAddr" (hasPrefix "ntm0_wAddr" ids)
       r4 <- check "autoName ntm: has ntm0_rOut" (hasPrefix "ntm0_rOut" ids)
       r5 <- check "autoName ntm: has ntm0_ll0_weight (ctrl hidden)" (hasPrefix "ntm0_ll0_weight" ids)
       r6 <- check "autoName ntm: has ntm0_ll1_weight (ctrl output)" (hasPrefix "ntm0_ll1_weight" ids)
       r7 <- check "autoName ntm: no duplicates" (noDuplicates ids)
       pure (r1 && r2 && r3 && r4 && r5 && r6 && r7)
  ]
