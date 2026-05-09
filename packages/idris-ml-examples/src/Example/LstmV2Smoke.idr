module Example.LstmV2Smoke

import Data.Vect
import System
import Compat.Random

import Device
import Layer.LstmV2
import Variable


-- Path C P3-2c — runtime sanity check for LstmV2.
--
-- Build an LstmV2 with i=2, o=4, run 5 forward steps with random
-- inputs, print the cell-output norms. This proves the typed-surface
-- LSTM actually executes (the smoke gate doesn't run this; it's a
-- standalone manual check).

-- Returns a persistent 1D tensor of `n` random uniform values in [-0.1, 0.1].
-- Threads the buffer pointer through `prim__setDouble` calls so the
-- compiler can't drop the writes (as it would with `let _ = ...`).
randomInput : (n : Nat) -> IO AnyPtr
randomInput n = do
  let nI = cast {to=Int} n
      buf = prim__allocDoubles nI
  buf' <- fillRandom buf 0 nI
  pure (prim__createState1d nI buf')
  where
    fillRandom : AnyPtr -> Int -> Int -> IO AnyPtr
    fillRandom b _ 0 = pure b
    fillRandom b o n = do
      v <- randomRIO (-0.1, 0.1)
      fillRandom (prim__setDouble b o v) (o + 1) (n - 1)

%default partial

main : IO ()
main = do
  srand 42

  putStrLn "=== LstmV2 smoke test (i=2, o=4) ==="
  putStrLn ""

  lstm <- lstmLayerV2 {i = 2} {o = 4} "smoke_lstm"
  putStrLn "  built lstm i=2 o=4 — params: smoke_lstm_iw, smoke_lstm_rw, smoke_lstm_b"
  putStrLn ""

  -- Run 5 forward steps with fresh random input each time.
  -- State threads through the loop; new hidden carries forward.
  go 5 lstm
  putStrLn ""
  putStrLn "Done."

  where
    go : Nat -> LstmStateV2 2 4 CPU -> IO ()
    go Z _ = pure ()
    go (S k) st = do
      inputT <- randomInput 2
      let inputTV = the (TVec 2 CPU) (MkTVar inputT Nothing)
          (st', outTV) = applyLstmV2 st inputTV
          outNorm = prim__item (prim__sum (prim__mul outTV.tensorPtr outTV.tensorPtr))
      putStrLn $ "  step " ++ show (5 `minus` k) ++ ": ||hidden||^2 = " ++ show outNorm
      go k st'
