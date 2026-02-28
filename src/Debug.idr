module Debug

import Data.List
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
showF x = sign ++ show w ++ "." ++ fracStr
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

sig : Double -> Double
sig x = 1.0 / (1.0 + exp (-x))

softplusD : Double -> Double
softplusD x = log (1.0 + exp x)


----------------------------------------------------------------------
-- Write Head Parameter Extraction Helper
----------------------------------------------------------------------

||| Split write head input into its component parameters.
||| Needs an explicit type signature so the plusAssociative rewrite works.
splitWriteInput : {n, w : Nat}
               -> Vector (((w + n) + 3) + w + w) Double
               -> ( Vector w Double, Vector n Double
                  , Vector 1 Double, Vector 1 Double, Vector 1 Double
                  , Vector w Double, Vector w Double )
splitWriteInput {n} {w} inp =
  let inp' = rewrite plusAssociative ((w + n) + 3) w w in inp
      (rhInput, remaining) = Tensor.splitAt ((w + n) + 3) inp'
      (rawErase, rawAdd) = splitAt w remaining
      (mainInput, prms) = splitAt (w + n) rhInput
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
  toDoubleLayer (ActivationLayer "sigmoid" _) = sigmoidLayer
  toDoubleLayer (ActivationLayer "tanh" _) = tanhLayer
  toDoubleLayer (ActivationLayer name _) = ActivationLayer name id
  toDoubleLayer (NormalizationLayer "softmax" _) = softmaxLayer
  toDoubleLayer (NormalizationLayer "logSoftmax" _) = logSoftmaxLayer
  toDoubleLayer (NormalizationLayer name _) = NormalizationLayer name id
  toDoubleLayer (NtmLayer controller mem rh wh ro) =
    NtmLayer (toDoubleNetwork controller)
             (map value mem) (map value rh) (map value wh) (map value ro)

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

debugApplyLayer layer@(ActivationLayer name _) inp =
  let (updated, out) = applyLayer layer inp
  in (updated, out, MkDebugEntry ("Activation<" ++ name ++ ">") [])

debugApplyLayer layer@(NormalizationLayer name _) inp =
  let (updated, out) = applyLayer layer inp
  in (updated, out, MkDebugEntry ("Normalization<" ++ name ++ ">") [])

debugApplyLayer {i} (NtmLayer {n} {hs} controller memory readHead writeHead readHeadOutput) inp =
  let
    -- Run controller
    (newController, controllerOutput) = forward controller (readHeadOutput ++ inp)

    -- Split controller output
    (readHeadInput, controllerOutput') = Tensor.splitAt (ReadHeadInputWidth n i) controllerOutput
    (writeHeadInput, networkOutput) = Tensor.splitAt (WriteHeadInputWidth n i) controllerOutput'

    -- Extract read head parameters
    (rMainInput, rPrms) = splitAt (i + n) readHeadInput
    (rKey, rShift) = splitAt i rMainInput
    (rBetaRaw, rPrms2) = splitAt 1 rPrms
    (rGRaw, rGammaRaw) = splitAt 1 rPrms2
    rBeta = softplusD (sum rBetaRaw)
    rG = sig (sum rGRaw)
    rGamma = 1.0 + 4.0 * sig (sum rGammaRaw)

    -- Extract write head parameters via helper
    (wKey, wShift, wBetaRaw, wGRaw, wGammaRaw, wRawErase, wRawAdd) = splitWriteInput writeHeadInput
    wEraseVec = map sig wRawErase
    wAddVec = map (\x => 2.0 * sig (2.0 * x) - 1.0) wRawAdd
    wBeta = softplusD (sum wBetaRaw)
    wG = sig (sum wGRaw)
    wGamma = 1.0 + 4.0 * sig (sum wGammaRaw)

    -- Run actual forward step
    (newReadHead, newReadHeadOutput) = forwardReadHead memory readHead readHeadInput
    (newWriteHead, newMemory) = forwardWriteHead memory writeHead writeHeadInput
    newLayer = NtmLayer newController newMemory newReadHead newWriteHead newReadHeadOutput

    -- Build debug entry
    entry = MkDebugEntry ("Ntm<" ++ show i ++ ", mem=" ++ show n ++ ">")
      [ ("readAddr",   showVec readHead.addressingWeights)
      , ("writeAddr",  showVec writeHead.readHead.addressingWeights)
      , ("readOutput", showVec readHeadOutput)
      , ("memory",     showMat memory)
      , ("readKey",    showVec rKey)
      , ("readShift",  showVec rShift)
      , ("readBeta",   showF rBeta)
      , ("readG",      showF rG)
      , ("readGamma",  showF rGamma)
      , ("writeKey",   showVec wKey)
      , ("writeShift", showVec wShift)
      , ("writeBeta",  showF wBeta)
      , ("writeG",     showF wG)
      , ("writeGamma", showF wGamma)
      , ("eraseVec",   showVec wEraseVec)
      , ("addVec",     showVec wAddVec)
      ]
  in (newLayer, networkOutput, entry)


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
