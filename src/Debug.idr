module Debug

import Data.List
import Data.List1
import Data.String
import Data.Vect

import Floating
import Layer
import Math
import Memory
import Tensor
import Variable


----------------------------------------------------------------------
-- Types
----------------------------------------------------------------------

||| A debug snapshot for one layer at one timestep
public export
record DebugEntry where
  constructor MkDebugEntry
  layerName : String
  fields : List (String, String)

||| Per-timestep snapshot of the entire network
public export
DebugSnapshot : Type
DebugSnapshot = List DebugEntry


----------------------------------------------------------------------
-- Formatting Helpers
----------------------------------------------------------------------

||| Format a Double to 4 decimal places
export
showF : Double -> String
showF x = if x /= x then "NaN"  -- NaN check
  else sign ++ show w ++ "." ++ fracStr
  where
    neg : Bool
    neg = x < 0.0
    ax : Double
    ax = abs x
    wholeAndFrac : Integer
    wholeAndFrac = cast (ax * 10000.0 + 0.5)
    w : Integer
    w = wholeAndFrac `div` 10000
    f : Integer
    f = wholeAndFrac `mod` 10000
    fracStr : String
    fracStr = padLeft 4 '0' (show f)
    sign : String
    sign = if neg then "-" else ""

||| Format a vector compactly
export
showVec : {n : Nat} -> Vector n Double -> String
showVec (VTensor xs) = "[" ++ go xs ++ "]"
  where
    go : Vect k (Tensor [] Double) -> String
    go [] = ""
    go [STensor x] = showF x
    go (STensor x :: rest) = showF x ++ " " ++ go rest

||| Format a matrix row by row
export
showMat : {r, c : Nat} -> Matrix r c Double -> String
showMat (VTensor rows) = "[" ++ go rows ++ "]"
  where
    go : Vect k (Vector c Double) -> String
    go [] = ""
    go [row] = showVec row
    go (row :: rest) = showVec row ++ "\n " ++ go rest


----------------------------------------------------------------------
-- Local Activation Helpers (matching Memory.idr's internal functions)
----------------------------------------------------------------------

sigD : Double -> Double
sigD x = 1.0 / (1.0 + exp (-x))

softplusD : Double -> Double
softplusD x = log (1.0 + exp x)


----------------------------------------------------------------------
-- Write Head Parameter Extraction Helper
----------------------------------------------------------------------

||| Split write head input into its component parameters.
||| Needs an explicit type signature so the plusAssociative rewrite works.
splitWriteInput : {w : Nat}
               -> Vector (((w + ShiftKernelSize) + 3) + w + w) Double
               -> ( Vector w Double, Vector ShiftKernelSize Double
                  , Vector 1 Double, Vector 1 Double, Vector 1 Double
                  , Vector w Double, Vector w Double )
splitWriteInput {w} inp =
  let inp' = rewrite plusAssociative ((w + ShiftKernelSize) + 3) w w in inp
      (rhInput, remaining) = Tensor.splitAt ((w + ShiftKernelSize) + 3) inp'
      (rawErase, rawAdd) = splitAt w remaining
      (mainInput, prms) = splitAt (w + ShiftKernelSize) rhInput
      (key, shft) = splitAt w mainInput
      (betaRaw, prms') = splitAt 1 prms
      (gRaw, gammaRaw) = splitAt 1 prms'
  in (key, shft, betaRaw, gRaw, gammaRaw, rawErase, rawAdd)


----------------------------------------------------------------------
-- Variable -> Double Network Conversion
----------------------------------------------------------------------

mutual
  ||| Convert a Variable-typed layer to Double by extracting values.
  ||| Activation/normalization functions are reconstructed by name.
  export
  toDoubleLayer : {i, o : Nat} -> Layer i o Variable -> Layer i o Double
  toDoubleLayer (LinearLayer w b _) =
    LinearLayer (map value w) (map value b) Nothing
  toDoubleLayer (RnnLayer iw rw b po _ _) =
    RnnLayer (map value iw) (map value rw) (map value b) (map value po) Nothing Nothing
  toDoubleLayer (LstmLayer iw rw b hs cs _ _) =
    LstmLayer (map value iw) (map value rw) (map value b) (map value hs) (map value cs) Nothing Nothing
  toDoubleLayer (ActivationLayer "sigmoid" _) = sigmoidLayer
  toDoubleLayer (ActivationLayer "tanh" _) = tanhLayer
  toDoubleLayer (ActivationLayer name _) = ActivationLayer name id
  toDoubleLayer (NormalizationLayer "softmax" _) = softmaxLayer
  toDoubleLayer (NormalizationLayer "logSoftmax" _) = logSoftmaxLayer
  toDoubleLayer (NormalizationLayer name _) = NormalizationLayer name id
  toDoubleLayer (NtmLayer lstm rfc wfc ofc mem ra wa ro) =
    NtmLayer (toDoubleLayer lstm) (toDoubleLayer rfc) (toDoubleLayer wfc) (toDoubleLayer ofc)
             (map value mem) (map value ra) (map value wa) (map value ro)

  ||| Convert a Variable-typed network to Double
  export
  toDoubleNetwork : {i, o : Nat} -> {hs : List Nat} -> Network i hs o Variable -> Network i hs o Double
  toDoubleNetwork (OutputLayer layer) = OutputLayer (toDoubleLayer layer)
  toDoubleNetwork (layer ~> rest) = toDoubleLayer layer ~> toDoubleNetwork rest


----------------------------------------------------------------------
-- Debug Layer Forward
----------------------------------------------------------------------

||| Forward + debug: runs the layer and captures internal state
export
debugApplyLayer : {i, o : Nat} -> Layer i o Double -> Vector i Double
               -> (Layer i o Double, Vector o Double, DebugEntry)
debugApplyLayer {i} {o} layer@(LinearLayer _ _ _) inp =
  let (updated, out) = applyLayer layer inp
  in (updated, out, MkDebugEntry ("Linear<" ++ show i ++ ":" ++ show o ++ ">") [])

debugApplyLayer {i} {o} (RnnLayer iw rw b previousOutput iwb rwb) inp =
  let (updated, out) = applyLayer (RnnLayer iw rw b previousOutput iwb rwb) inp
  in (updated, out, MkDebugEntry ("Rnn<" ++ show i ++ ":" ++ show o ++ ">")
       [("hidden", showVec previousOutput)])

debugApplyLayer {i} {o} (LstmLayer iw rw b hiddenState cellState iwb rwb) inp =
  let (updated, out) = applyLayer (LstmLayer iw rw b hiddenState cellState iwb rwb) inp
  in (updated, out, MkDebugEntry ("Lstm<" ++ show i ++ ":" ++ show o ++ ">")
       [("hidden", showVec hiddenState), ("cell", showVec cellState)])

debugApplyLayer layer@(ActivationLayer name _) inp =
  let (updated, out) = applyLayer layer inp
  in (updated, out, MkDebugEntry ("Activation<" ++ name ++ ">") [])

debugApplyLayer layer@(NormalizationLayer name _) inp =
  let (updated, out) = applyLayer layer inp
  in (updated, out, MkDebugEntry ("Normalization<" ++ name ++ ">") [])

debugApplyLayer {i} {o} (NtmLayer {n} {m} {h} lstm readFc writeFc outputFc memory readAddr writeAddr readOutput) inp =
  let
    -- Run forward pass via applyLayer (handles full pipeline)
    layer = NtmLayer lstm readFc writeFc outputFc memory readAddr writeAddr readOutput
    (updatedLayer, output) = applyLayer layer inp

    -- Build debug entry with pre-step state
    entry = MkDebugEntry ("Ntm<" ++ show i ++ ":" ++ show o ++ ", mem=" ++ show n ++ "x" ++ show m ++ ">")
      [ ("readAddr",   showVec readAddr)
      , ("writeAddr",  showVec writeAddr)
      , ("readOutput", showVec readOutput)
      , ("memory",     showMat memory)
      ]
  in (updatedLayer, output, entry)


----------------------------------------------------------------------
-- Debug Network Forward
----------------------------------------------------------------------

||| Walk the network, collecting debug entries from each layer
export
debugForward : {i, o : Nat} -> {hs : List Nat} -> Network i hs o Double -> Vector i Double
            -> (Network i hs o Double, Vector o Double, DebugSnapshot)
debugForward (OutputLayer layer) x =
  let (updatedLayer, output, entry) = debugApplyLayer layer x
  in (OutputLayer updatedLayer, output, [entry])
debugForward {hs = h :: _} (layer ~> layers) x =
  let (updatedLayer, layerOutput, entry) = debugApplyLayer layer x
      (updatedNetwork, networkOutput, entries) = debugForward layers layerOutput
  in (updatedLayer ~> updatedNetwork, networkOutput, entry :: entries)

||| Recurrent: fold over timesteps, collecting per-timestep snapshots
export
debugForwardRecurrent : {i, o : Nat} -> {hs : List Nat} -> Network i hs o Double -> List (Vector i Double)
                     -> (Network i hs o Double, List (Vector o Double), List DebugSnapshot)
debugForwardRecurrent model inputs = foldl step (model, [], []) inputs
  where
    step : (Network i hs o Double, List (Vector o Double), List DebugSnapshot) -> Vector i Double
        -> (Network i hs o Double, List (Vector o Double), List DebugSnapshot)
    step (m, outs, snaps) inp =
      let (m', out, snap) = debugForward m inp
      in (m', outs ++ [out], snaps ++ [snap])


----------------------------------------------------------------------
-- Printing
----------------------------------------------------------------------

||| Print all timesteps for a sequence
export
printDiagnostics : String -> List DebugSnapshot -> IO ()
printDiagnostics label snapshots = do
  putStrLn $ "=== Diagnostics: " ++ label ++ " ==="
  go 0 snapshots
  where
    printEntry : DebugEntry -> IO ()
    printEntry entry = do
      putStrLn $ "  [" ++ entry.layerName ++ "]"
      traverse_ (\(k, v) => putStrLn $ "    " ++ k ++ ": " ++ v) entry.fields

    go : Nat -> List DebugSnapshot -> IO ()
    go _ [] = pure ()
    go t (snap :: rest) = do
      putStrLn $ "--- Timestep " ++ show t ++ " ---"
      traverse_ printEntry snap
      go (t + 1) rest


----------------------------------------------------------------------
-- Parsing Utilities
----------------------------------------------------------------------

||| Parse "[0.12 0.34 ...]" back to List Double
export
parseVec : String -> List Double
parseVec s =
  let cleaned = pack $ filter (\c => c /= '[' && c /= ']') (unpack s)
  in map cast (words cleaned)

||| Parse a scalar string to Double
export
parseScalar : String -> Double
parseScalar = cast

||| Parse a matrix string (showMat format) to list of row vectors
export
parseMat : String -> List (List Double)
parseMat s =
  let rows = lines s
      parseRow : String -> List Double
      parseRow r = let cleaned = pack $ filter (\c => c /= '[' && c /= ']') (unpack r)
                   in map cast (words cleaned)
      isNonEmpty : List a -> Bool
      isNonEmpty [] = False
      isNonEmpty _ = True
  in filter isNonEmpty (map parseRow rows)


----------------------------------------------------------------------
-- NTM Entry Extraction
----------------------------------------------------------------------

||| Find the NTM DebugEntry in a snapshot
export
findNtmEntry : DebugSnapshot -> Maybe DebugEntry
findNtmEntry [] = Nothing
findNtmEntry (e :: es) =
  if isPrefixOf "Ntm<" e.layerName then Just e else findNtmEntry es

||| Look up a field value in a DebugEntry
export
lookupField : String -> DebugEntry -> Maybe String
lookupField key entry = lookup key entry.fields


----------------------------------------------------------------------
-- NTM Summary Types
----------------------------------------------------------------------

public export
record NtmSummary where
  constructor MkNtmSummary
  writeGInput, writeGOutput : Double
  readGInput, readGOutput : Double
  avgWriteBeta, avgReadBeta : Double
  avgWriteGamma, avgReadGamma : Double
  writeAddrEntropy, readAddrEntropy : Double
  writeAddrPeakMass, readAddrPeakMass : Double
  writeMonotonic, readMonotonic : Bool
  writeArgmaxes, readArgmaxes : List Nat
  slotsUsed : Nat
  numSlots : Nat
  seqLen : Nat


----------------------------------------------------------------------
-- Summary Helpers
----------------------------------------------------------------------

avg : List Double -> Double
avg [] = 0.0
avg xs = sum xs / cast (length xs)

listArgmax : List Double -> Nat
listArgmax [] = 0
listArgmax (x :: xs) = go 0 x 1 xs
  where
    go : Nat -> Double -> Nat -> List Double -> Nat
    go bi _ _ [] = bi
    go bi bv ci (y :: ys) =
      if y > bv then go ci y (ci + 1) ys else go bi bv (ci + 1) ys

addrEntropy : List Double -> Double
addrEntropy xs =
  negate $ foldl (\acc, p => if p > 1.0e-12 then acc + p * log p else acc) 0.0 xs

peakMass : List Double -> Double
peakMass [] = 0.0
peakMass (x :: xs) = foldl max x xs

vecNorm : List Double -> Double
vecNorm xs = sqrt (foldl (\acc, x => acc + x * x) 0.0 xs)

countSlotsUsed : List (List Double) -> Nat
countSlotsUsed rows = length (filter (\row => vecNorm row > 0.01) rows)

isStrictlyIncreasing : List Nat -> Bool
isStrictlyIncreasing [] = True
isStrictlyIncreasing [_] = True
isStrictlyIncreasing (x :: y :: rest) = x < y && isStrictlyIncreasing (y :: rest)


----------------------------------------------------------------------
-- Summary Computation
----------------------------------------------------------------------

||| Compute summary metrics from debug snapshots of a single sequence.
||| seqLen = number of input-phase timesteps (half total).
export
computeSummary : Nat -> List DebugSnapshot -> Maybe NtmSummary
computeSummary sl snapshots = do
  ntmEntries <- traverse findNtmEntry snapshots
  let getS = \field, entry => parseScalar (fromMaybe "0" (lookupField field entry))
  let writeGs = map (getS "writeG") ntmEntries
  let readGs = map (getS "readG") ntmEntries
  let writeBetas = map (getS "writeBeta") ntmEntries
  let readBetas = map (getS "readBeta") ntmEntries
  let writeGammas = map (getS "writeGamma") ntmEntries
  let readGammas = map (getS "readGamma") ntmEntries
  let writeAddrs = map (\e => parseVec (fromMaybe "[]" (lookupField "writeAddr" e))) ntmEntries
  let readAddrs = map (\e => parseVec (fromMaybe "[]" (lookupField "readAddr" e))) ntmEntries
  let (writeGsIn, writeGsOut) = splitAt sl writeGs
  let (readGsIn, readGsOut) = splitAt sl readGs
  let wArgmaxes = map listArgmax writeAddrs
  let rArgmaxes = map listArgmax readAddrs
  let (wArgIn, _) = splitAt sl wArgmaxes
  let (_, rArgOut) = splitAt sl rArgmaxes
  let memEntry = case drop sl ntmEntries of
                   (e :: _) => Just e
                   [] => case ntmEntries of
                           (e :: _) => Just e
                           [] => Nothing
  let memRows = case memEntry >>= lookupField "memory" of
                  Just s => parseMat s
                  Nothing => []
  pure $ MkNtmSummary
    (avg writeGsIn) (avg writeGsOut)
    (avg readGsIn) (avg readGsOut)
    (avg writeBetas) (avg readBetas)
    (avg writeGammas) (avg readGammas)
    (avg (map addrEntropy writeAddrs))
    (avg (map addrEntropy readAddrs))
    (avg (map peakMass writeAddrs))
    (avg (map peakMass readAddrs))
    (isStrictlyIncreasing wArgIn)
    (isStrictlyIncreasing rArgOut)
    wArgmaxes rArgmaxes
    (countSlotsUsed memRows) (length memRows) sl

||| Average multiple NtmSummary values (for aggregate comparison)
export
avgSummaries : List NtmSummary -> Maybe NtmSummary
avgSummaries [] = Nothing
avgSummaries ss@(s :: _) =
  let avgF : (NtmSummary -> Double) -> Double
      avgF f = avg (map f ss)
  in pure $ MkNtmSummary
    (avgF writeGInput) (avgF writeGOutput)
    (avgF readGInput) (avgF readGOutput)
    (avgF avgWriteBeta) (avgF avgReadBeta)
    (avgF avgWriteGamma) (avgF avgReadGamma)
    (avgF writeAddrEntropy) (avgF readAddrEntropy)
    (avgF writeAddrPeakMass) (avgF readAddrPeakMass)
    (all (\x => x.writeMonotonic) ss) (all (\x => x.readMonotonic) ss)
    s.writeArgmaxes s.readArgmaxes
    s.slotsUsed s.numSlots s.seqLen


----------------------------------------------------------------------
-- Summary Printing
----------------------------------------------------------------------

showDelta : Double -> String
showDelta d =
  let sign = if d >= 0.0 then "+" else ""
      flag = if abs d > 0.15 then " !" else ""
  in "  (" ++ sign ++ showF d ++ flag ++ ")"

showBool : Bool -> String
showBool True = "YES"
showBool False = "NO"

slotGrid : Nat -> Nat -> String
slotGrid ns ai =
  "[" ++ pack (map (\i => if i == ai then '#' else '.') [0..pred ns]) ++ "]"

showTimesteps : Nat -> Nat -> List Nat -> String
showTimesteps ns startIdx argmaxes = go startIdx argmaxes
  where
    go : Nat -> List Nat -> String
    go _ [] = ""
    go t [a] = "t" ++ show t ++ slotGrid ns a
    go t (a :: rest) = "t" ++ show t ++ slotGrid ns a ++ " " ++ go (t + 1) rest

showArgmaxList : List Nat -> String
showArgmaxList xs = "[" ++ go xs ++ "]"
  where
    go : List Nat -> String
    go [] = ""
    go [x] = show x
    go (x :: rest) = show x ++ "," ++ go rest

||| Print a compact summary for one sequence
export
printSummary : String -> NtmSummary -> IO ()
printSummary label s = do
  putStrLn $ "=== NTM Summary: " ++ label ++ " ==="
  putStrLn $ "  Gate (g: 1=content, 0=location):"
  putStrLn $ "    Write:  input=" ++ showF s.writeGInput ++ "  output=" ++ showF s.writeGOutput
  putStrLn $ "    Read:   input=" ++ showF s.readGInput ++ "  output=" ++ showF s.readGOutput
  putStrLn $ "  Beta:  write=" ++ showF s.avgWriteBeta ++ "  read=" ++ showF s.avgReadBeta
  putStrLn $ "  Gamma: write=" ++ showF s.avgWriteGamma ++ "  read=" ++ showF s.avgReadGamma
  putStrLn $ "  Focus: write entropy=" ++ showF s.writeAddrEntropy
           ++ " peak=" ++ showF s.writeAddrPeakMass
           ++ " | read entropy=" ++ showF s.readAddrEntropy
           ++ " peak=" ++ showF s.readAddrPeakMass
  putStrLn $ "  Memory: " ++ show s.slotsUsed ++ "/" ++ show s.numSlots ++ " slots used"
  putStrLn $ "  Sequential: write=" ++ showArgmaxList (fst (splitAt s.seqLen s.writeArgmaxes))
           ++ " " ++ showBool s.writeMonotonic
           ++ " | read=" ++ showArgmaxList (snd (splitAt s.seqLen s.readArgmaxes))
           ++ " " ++ showBool s.readMonotonic

||| Print addressing grid showing argmax per timestep
export
printAddrGrid : NtmSummary -> IO ()
printAddrGrid s = do
  let ns = s.numSlots
  let sl = s.seqLen
  let (wIn, wOut) = splitAt sl s.writeArgmaxes
  let (rIn, rOut) = splitAt sl s.readArgmaxes
  putStrLn "  Addressing grid:"
  putStrLn $ "    Write: " ++ showTimesteps ns 0 wIn ++ "  (input)"
  putStrLn $ "           " ++ showTimesteps ns sl wOut ++ "  (output)"
  putStrLn $ "    Read:  " ++ showTimesteps ns 0 rIn ++ "  (input)"
  putStrLn $ "           " ++ showTimesteps ns sl rOut ++ "  (output)"

padRight : Nat -> String -> String
padRight n s = s ++ pack (replicate (minus n (length s)) ' ')

||| Print train vs test comparison with deltas and diagnostic flags
export
printComparison : NtmSummary -> NtmSummary -> IO ()
printComparison train test = do
  putStrLn "=== Train vs Test Comparison (averaged) ==="
  putStrLn $ "                          Train    Test     Delta"
  let row : String -> Double -> Double -> IO ()
      row label tv testv = putStrLn $ "  " ++ padRight 22 label
        ++ showF tv ++ "   " ++ showF testv ++ showDelta (testv - tv)
  row "Gate g (write/in):" train.writeGInput test.writeGInput
  row "Gate g (write/out):" train.writeGOutput test.writeGOutput
  row "Gate g (read/in):" train.readGInput test.readGInput
  row "Gate g (read/out):" train.readGOutput test.readGOutput
  row "Beta (write):" train.avgWriteBeta test.avgWriteBeta
  row "Beta (read):" train.avgReadBeta test.avgReadBeta
  row "Gamma (write):" train.avgWriteGamma test.avgWriteGamma
  row "Gamma (read):" train.avgReadGamma test.avgReadGamma
  row "Entropy (write):" train.writeAddrEntropy test.writeAddrEntropy
  row "Entropy (read):" train.readAddrEntropy test.readAddrEntropy
  row "Peak mass (write):" train.writeAddrPeakMass test.writeAddrPeakMass
  row "Peak mass (read):" train.readAddrPeakMass test.readAddrPeakMass
  putStrLn $ "  " ++ padRight 22 "Write monotonic:"
    ++ padRight 9 (showBool train.writeMonotonic)
    ++ showBool test.writeMonotonic
  putStrLn $ "  " ++ padRight 22 "Read monotonic:"
    ++ padRight 9 (showBool train.readMonotonic)
    ++ showBool test.readMonotonic
  putStrLn ""
  putStrLn "  Interpretation guide:"
  putStrLn "  | Observation                 | Diagnosis                            |"
  putStrLn "  |------------------------------|--------------------------------------|"
  putStrLn "  | Train g low, test g high     | Memorization (content fallback)      |"
  putStrLn "  | Both g high                  | Never learned location addressing    |"
  putStrLn "  | g low, monotonic=NO          | Shift broken (wrong direction)       |"
  putStrLn "  | Slots used << seq length     | Memory collapse                      |"
