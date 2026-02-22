module Variable

import Data.List
import Data.Maybe
import Data.SortedMap
import Data.SortedSet
import System.Random

import Floating
import Util


----------------------------------------------------------------------
-- Global Node ID Counter (FFI)
----------------------------------------------------------------------

%foreign "scheme:(lambda (dummy) (when (not (top-level-bound? 'idris-ml-nid)) (set-top-level-value! 'idris-ml-nid 0)) (set-top-level-value! 'idris-ml-nid (+ (top-level-value 'idris-ml-nid) 1)) (top-level-value 'idris-ml-nid))"
prim__nextNodeId : Double -> Int

export
%noinline
nextNodeId : Double -> Nat
nextNodeId x = cast (prim__nextNodeId x)


----------------------------------------------------------------------
-- Record Definition
----------------------------------------------------------------------

public export
record Variable where
  constructor Var
  nodeId : Nat
  paramId : Maybe String
  value : Double
  grad : Double
  back : Double -> List Double
  children : List Variable

----------------------------------------------------------------------
-- Instances
----------------------------------------------------------------------

export
Show Variable where
  show v =
    "Var" ++
    (case v.paramId of (Just pid) => "<" ++ pid ++ ">"; Nothing => "") ++
    "(" ++ show v.value ++ ":" ++ show v.grad ++ ")"
    -- ++ " [" ++ concat (intersperse ", " (map show v.children)) ++ "]" -- NOTE: Toggle this for children

public export
implementation Eq Variable where
  v1 == v2 = v1.value == v2.value

public export
implementation Ord Variable where
  v1 < v2 = v1.value < v2.value

public export
implementation FromDouble Variable where
  fromDouble n =
    Var
      { nodeId = nextNodeId n,
        paramId = Nothing,
        value = n,
        grad = 0,
        back = const [],
        children = []
      }

public export
implementation Cast Variable Double where
  cast v = v.value

public export
implementation Cast Double Variable where
  cast = fromDouble

public export
implementation Random Variable where
  randomIO = map fromDouble randomIO
  randomRIO (lo, hi) = map fromDouble (randomRIO (lo.value, hi.value))

----------------------------------------------------------------------
-- Construction Helpers
----------------------------------------------------------------------

unaryOp : Double -> (Double -> Double) -> Variable -> Variable
unaryOp val bk x = Var (nextNodeId val) Nothing val 0 (\g => [bk g]) [x]

binaryOp : Double -> (Double -> Double) -> (Double -> Double) -> Variable -> Variable -> Variable
binaryOp val bkL bkR x y = Var (nextNodeId val) Nothing val 0 (\g => [bkL g, bkR g]) [x, y]

public export
implementation Num Variable where
  v1 + v2 = binaryOp (v1.value + v2.value) id id v1 v2
  v1 * v2 = binaryOp (v1.value * v2.value) (* v2.value) (* v1.value) v1 v2
  fromInteger v = let val = fromInteger v in Var (nextNodeId val) Nothing val 0 (const []) []

public export
implementation Neg Variable where
  v1 - v2 = binaryOp (v1.value - v2.value) id negate v1 v2
  negate v = unaryOp (negate v.value) negate v

public export
implementation Abs Variable where
  abs v = unaryOp (abs v.value) (* signum v.value) v

public export
implementation Fractional Variable where
  v1 / v2 = binaryOp (v1.value / v2.value) (/ v2.value) (\g => -g * v1.value / pow v2.value 2) v1 v2

public export
implementation Floating Variable where
  exp v  = unaryOp (exp v.value) (* exp v.value) v
  log v  = unaryOp (log v.value) (/ v.value) v
  pow v1 v2 = binaryOp (pow v1.value v2.value)
    (* v2.value * pow v1.value (v2.value - 1))
    (\g => g * pow v1.value v2.value * log v1.value) v1 v2
  sqrt v = unaryOp (sqrt v.value) (/ (2 * sqrt v.value)) v

----------------------------------------------------------------------
-- Parameter Naming
----------------------------------------------------------------------

export
param : String -> Double -> Variable
param pid = { paramId := Just pid } . fromDouble

export
nameParam : String -> Nat -> Variable -> Variable
nameParam prefx i p = { paramId := Just (prefx ++ show i) } p

----------------------------------------------------------------------
-- Backpropagation
----------------------------------------------------------------------

topoSort : Variable -> List Variable
topoSort root = snd $ go empty root
  where
    go : SortedSet Nat -> Variable -> (SortedSet Nat, List Variable)
    go visited v =
      if contains v.nodeId visited
        then (visited, [])
        else
          let visited' = insert v.nodeId visited
              (visited'', childNodes) =
                foldl (\(vis, acc), c =>
                  let (vis', nodes) = go vis c
                  in (vis', acc ++ nodes))
                  (visited', []) v.children
          in (visited'', childNodes ++ [v])

export
collectGrads : Double -> Variable -> SortedMap String Double
collectGrads initGrad root =
  let sorted = reverse $ topoSort root
      nodeGrads = singleton root.nodeId initGrad
  in snd $ foldl propagateOne (nodeGrads, empty) sorted
  where
    propagateOne :
      (SortedMap Nat Double, SortedMap String Double) ->
      Variable ->
      (SortedMap Nat Double, SortedMap String Double)
    propagateOne (nodeGrads, paramGrads) v =
      let g = fromMaybe 0 $ lookup v.nodeId nodeGrads
          childGs = v.back g
          nodeGrads' = foldl (\m, (c, cg) =>
            mergeWith (+) m (singleton c.nodeId cg))
            nodeGrads (zip v.children childGs)
          paramGrads' = case v.paramId of
            Just pid => mergeWith (+) paramGrads (singleton pid g)
            Nothing  => paramGrads
      in (nodeGrads', paramGrads')
