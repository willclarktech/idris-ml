module Test.TensorExpr

import Data.List
import Data.Vect

import Array
import Executor
import Tensor
import Test.Config
import Test.Harness

-- Small fixed tensors for the expression-op suite. ioRerun pins FFI
-- ordering (pure-typed bulkToTensor* reorders across sibling
-- let-bindings otherwise; see gotchas).

row2 : Double -> Double -> Vector 2 Double
row2 a b = VArray [SArray a, SArray b]

mk32 : (Double, Double) -> (Double, Double) -> (Double, Double) ->
       IO (Tensor [3, 2] TestExecutor TestDType WithGrad)
mk32 (a, b) (c, d) (e, f) = ioRerun (\_ =>
  MkTensor (bulkToTensor2d {ex=TestExecutor} {dt=TestDType} [row2 a b, row2 c d, row2 e f]) Nothing)

mkIdx : Vect 3 Double -> IO (Tensor [3] TestExecutor TestDType NoGrad)
mkIdx xs = ioRerun (\_ =>
  MkTensor (bulkToTensor {ex=TestExecutor} {dt=TestDType} (VArray (map SArray xs))) Nothing)

read3 : Tensor [3] TestExecutor TestDType g -> (Double, Double, Double)
read3 t = ( primItem1d {ex=TestExecutor} t.tensorPtr 0
          , primItem1d {ex=TestExecutor} t.tensorPtr 1
          , primItem1d {ex=TestExecutor} t.tensorPtr 2 )

gatherRowsPicks : IO Bool
gatherRowsPicks = do
  q <- mk32 (1, 2) (3, 4) (5, 6)
  ix <- mkIdx [1, 0, 1]
  r <- tgatherRows q ix
  let (x, y, z) = read3 r
  check "tgatherRows [[1,2],[3,4],[5,6]] @ [1,0,1] = (2,3,6)"
        (x == 2 && y == 3 && z == 6)

maxRowsValues : IO Bool
maxRowsValues = do
  q <- mk32 (1, 5) (7, 2) (3, 4)
  r <- tmaxRows q
  let (x, y, z) = read3 r
  check "tmaxRows [[1,5],[7,2],[3,4]] = (5,7,4)" (x == 5 && y == 7 && z == 4)

operatorAliases : IO Bool
operatorAliases = do
  a <- mk32 (1, 2) (3, 4) (5, 6)
  b <- mk32 (10, 20) (30, 40) (50, 60)
  viaOps   <- a +. !(2.0 *: b)
  viaNames <- tadd a !(tmulScalar b 2.0)
  let ok = all (\i => primItem1d {ex=TestExecutor} viaOps.tensorPtr i
                    == primItem1d {ex=TestExecutor} viaNames.tensorPtr i)
               (the (List Int) [0, 1, 2, 3, 4, 5])
  check "(+.)/(*:) with bang = named ops" ok

----------------------------------------------------------------------
-- DQN TD-loss acceptance (api-critique S3 exhibit)
----------------------------------------------------------------------
--
-- The hand-rolled per-sample form below is ported from
-- Example/Dqn.idr's perSampleLoss + meanScalarLoss as the in-test
-- oracle (Dqn.idr itself is untouched until the migration sweep).
-- The expression-ops form reproduces its loss value and gradients in
-- a handful of lines — the PyTorch equivalent is
-- F.mse_loss(q.gather(1, a).squeeze(), r + gamma * q_next.max(1).values).

gamma : Double
gamma = 0.9

qVals : List (Double, Double)
qVals = [(0.2, 0.8), (0.5, 0.1), (0.9, 0.3), (0.4, 0.6)]

actionsH : List Double
actionsH = [1, 0, 1, 0]

rewardsH : List Double
rewardsH = [1.0, 0.5, 0.0, 2.0]

qNextH : List (Double, Double)
qNextH = [(0.3, 0.7), (0.6, 0.2), (0.1, 0.4), (0.8, 0.5)]

packBuf : List Double -> AnyPtr -> Int -> AnyPtr
packBuf [] b _ = b
packBuf (x :: xs) b o = packBuf xs (prim__setDouble b o x) (o + 1)

mkParam42 : String -> IO (Tensor [4, 2] TestExecutor TestDType WithGrad)
mkParam42 name = do
  buf <- ioRerun (\_ => packBuf (concatMap (\(a, b) => [a, b]) qVals) (prim__allocDoubles 8) 0)
  tparam2d {ex=TestExecutor} name buf

mkVec4 : List Double -> IO (Tensor [4] TestExecutor TestDType WithGrad)
mkVec4 xs = ioRerun (\_ =>
  MkTensor (dtCreate1d {ex=TestExecutor} {t=TestDType} 4 (packBuf xs (prim__allocDoubles 4) 0) 0
              (deviceStreamTag {ex=TestExecutor})) Nothing)

mkMat42 : List (Double, Double) -> IO (Tensor [4, 2] TestExecutor TestDType WithGrad)
mkMat42 rows = ioRerun (\_ =>
  MkTensor (dtCreate2d {ex=TestExecutor} {t=TestDType} 4 2
              (packBuf (concatMap (\(a, b) => [a, b]) rows) (prim__allocDoubles 8) 0) 0
              (deviceStreamTag {ex=TestExecutor})) Nothing)

-- Oracle: Dqn.idr's shape — per-sample row/elem selects against a
-- host-computed constant target (Dqn's computeTargetVal materialises
-- max(qNext_i) host-side), hand-summed, mean-scaled.
perSample : Tensor [4, 2] TestExecutor TestDType WithGrad ->
            (row : Int) -> (action : Int) -> (reward : Double) -> (maxQNext : Double) ->
            IO (Tensor [] TestExecutor TestDType WithGrad)
perSample q row action reward maxQNext = do
  r <- trowSelect q row
  qa <- telemSelect r action
  tgt <- tconstScalar {ex=TestExecutor} {dt=TestDType} (reward + gamma * maxQNext)
  diff <- tsub qa tgt
  tmul diff diff

oracleLoss : Tensor [4, 2] TestExecutor TestDType WithGrad ->
             IO (Tensor [] TestExecutor TestDType WithGrad)
oracleLoss q = do
  l0 <- perSample q 0 1 1.0 0.7
  l1 <- perSample q 1 0 0.5 0.6
  l2 <- perSample q 2 1 0.0 0.4
  l3 <- perSample q 3 0 2.0 0.8
  s01 <- tadd l0 l1
  s23 <- tadd l2 l3
  totalLoss <- tadd s01 s23
  tmulScalar totalLoss 0.25

-- Expression-ops form: the whole TD loss in four lines.
exprLoss : Tensor [4, 2] TestExecutor TestDType WithGrad ->
           IO (Tensor [] TestExecutor TestDType WithGrad)
exprLoss q = do
  actions <- mkVec4 actionsH
  rewards <- mkVec4 rewardsH
  qNext <- mkMat42 qNextH
  pred <- tgatherRows q actions
  tgt <- rewards +. !(gamma *: !(tmaxRows qNext))
  loss <- tmseLoss pred tgt
  tmulScalar loss 0.25

findParam : String -> IO (Maybe Int)
findParam name = do
  n <- primIO (primParamCount {ex=TestExecutor})
  go 0 n
  where
    go : Int -> Int -> IO (Maybe Int)
    go i n = if i >= n then pure Nothing else do
      nm <- primIO (primParamName {ex=TestExecutor} i)
      if nm == name then pure (Just i) else go (i + 1) n

gradsAt : Int -> IO (List Double)
gradsAt pIdx = traverse (\j => primIO (primParamGradItemAt {ex=TestExecutor} pIdx j))
                        (the (List Int) [0, 1, 2, 3, 4, 5, 6, 7])

tdLossAcceptance : IO Bool
tdLossAcceptance = do
  wOr <- mkParam42 "tdl_w_or"
  wNew <- mkParam42 "tdl_w_new"
  lOr <- oracleLoss wOr
  lNew <- exprLoss wNew
  let vOr = tensorItem lOr
  let vNew = tensorItem lNew
  Just iOr <- findParam "tdl_w_or" | Nothing => check "tdl_w_or registered" False
  Just iNew <- findParam "tdl_w_new" | Nothing => check "tdl_w_new registered" False
  primIO (primParamZeroAll {ex=TestExecutor})
  runBackward lOr
  gOr <- gradsAt iOr
  primIO (primParamZeroAll {ex=TestExecutor})
  runBackward lNew
  gNew <- gradsAt iNew
  let closeV = abs (vOr - vNew) < 1.0e-12
  let closeG = all (\(a, b) => abs (a - b) < 1.0e-12) (zip gOr gNew)
  check ("TD loss: expr form == 37-line oracle (loss " ++ show vNew
         ++ " vs " ++ show vOr ++ "; grads match)") (closeV && closeG)

-- Batched NLL oracle: with all-equal logits, logSoftmax is uniform
-- (log(1/C) per class), so for one-hot targets tnllLossMean collapses to
-- -(1/(b*C)) * sum_rows log(1/C) = log(C)/C, independent of the logit
-- value and the batch size. C=3 → log(3)/3 ≈ 0.3662. Pins both the
-- axis=1 (row-wise) softmax and the 1/(b*C) scaling against PyTorch's
-- nll_loss(log_softmax(logits,-1), target).
row3 : Double -> Double -> Double -> Vector 3 Double
row3 a b c = VArray [SArray a, SArray b, SArray c]

nllLossMeanOracle : IO Bool
nllLossMeanOracle = do
  pred0 <- tensor {ex=TestExecutor} {dt=TestDType} {dims=[2, 3]} (Const 0.0)
  tgt0  <- the (IO (Tensor [2, 3] TestExecutor TestDType NoGrad)) $
             ioRerun (\_ => MkTensor (bulkToTensor2d {ex=TestExecutor} {dt=TestDType}
                                        [row3 1 0 0, row3 0 1 0]) Nothing)
  let pred = the (Tensor [2, 3] TestExecutor TestDType WithGrad) (retypeGrad pred0)
  let tgt  = the (Tensor [2, 3] TestExecutor TestDType WithGrad) (retypeGrad tgt0)
  l <- tnllLossMean {b=2} {n=3} pred tgt
  let v = tensorItem l
  let expected = log 3.0 / 3.0
  check ("tnllLossMean uniform-logits oracle = log(3)/3 (got " ++ show v ++ ")")
        (abs (v - expected) < 1.0e-9)

export
tests : List (IO Bool)
tests = [gatherRowsPicks, maxRowsValues, operatorAliases, tdLossAcceptance, nllLossMeanOracle]
