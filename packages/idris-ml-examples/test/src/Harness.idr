module Harness


export
check : String -> Bool -> IO Bool
check name True  = do putStrLn ("  PASS: " ++ name); pure True
check name False = do putStrLn ("  FAIL: " ++ name); pure False

export
checkClose : String -> Double -> Double -> Double -> IO Bool
checkClose name expected actual tol =
  let diff = abs (expected - actual)
  in if diff <= tol
       then do putStrLn ("  PASS: " ++ name); pure True
       else do putStrLn ("  FAIL: " ++ name ++ " expected=" ++ show expected ++ " actual=" ++ show actual ++ " diff=" ++ show diff)
               pure False

export
runSuite : String -> List (IO Bool) -> IO Nat
runSuite name ts = do
  putStrLn ("[" ++ name ++ "]")
  results <- traverse id ts
  let ct = length results
  let failures = foldl (\acc, b => if b then acc else S acc) 0 results
  putStrLn ("  " ++ show (ct `minus` failures) ++ "/" ++ show ct ++ " passed")
  pure failures

runLoop : Nat -> List (String, List (IO Bool)) -> IO ()
runLoop nFails ss =
  case ss of
    [] => if nFails == 0
             then putStrLn "\nAll tests passed."
             else putStrLn ("\n" ++ show nFails ++ " failure(s).")
    ((name, ts) :: rest) => do
      n <- runSuite name ts
      runLoop (nFails + n) rest

export
runAll : List (String, List (IO Bool)) -> IO ()
runAll = runLoop 0
