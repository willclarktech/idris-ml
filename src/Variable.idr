module Variable

import Data.List
import Math
import System.Random


signum : Double -> Double
signum x = case compare x 0 of
  GT => 1.0
  EQ => 0.0
  LT => -1.0

public export
record Variable where
  constructor Var
  paramId : Maybe String
  value : Double
  grad : Double
  back : Double -> List Double
  children : List Variable

export
partial
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
      { paramId = Nothing,
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

public export
implementation Num Variable where
  v1 + v2 =
    Var
      { paramId = Nothing,
        value = v1.value + v2.value,
        grad = 0,
        back = \g => [g, g],
        children = [v1, v2]
      }
  v1 * v2 =
    Var
      { paramId = Nothing,
        value = v1.value * v2.value,
        grad = 0,
        back = \g => [g * v2.value, g * v1.value],
        children = [v1, v2]
      }
  fromInteger v =
    Var
      { paramId = Nothing,
        value = fromInteger v,
        grad = 0,
        back = const [],
        children = []
      }

public export
implementation Neg Variable where
  v1 - v2 =
    Var
      { paramId = Nothing,
        value = v1.value - v2.value,
        grad = 0,
        back = \g => [g, -g],
        children = [v1, v2]
      }
  negate v =
    Var
      { paramId = Nothing,
        value = negate $ v.value,
        grad = 0,
        back = \g => [-g],
        children = [v]
      }

public export
implementation Abs Variable where
  abs v =
    Var
      { paramId = Nothing,
        value = abs $ v.value,
        grad = 0,
        back = \g => [g * signum (v.value)],
        children = [v]
      }

public export
implementation Fractional Variable where
  v1 / v2 =
    Var
      { paramId = Nothing,
        value = v1.value / v2.value,
        grad = 0,
        back = \g => [g / v2.value, -g * v1.value / (pow v2.value 2)],
        children = [v1, v2]
      }

public export
implementation Floating Variable where
  exp v =
      Var
        { paramId = Nothing,
          value = exp (v.value),
          grad = 0,
          back = \g => [g * exp (v.value)],
          children = [v]
        }
  log v =
    Var
      { paramId = Nothing,
        value = log (v.value),
        grad = 0,
        back = \g => [g / v.value],
        children = [v]
      }
  pow v1 v2 =
    Var {
      paramId = Nothing,
      value = pow v1.value v2.value,
      grad = 0,
      back = \g => [
          g * (pow v1.value v2.value) * (log v1.value),
          g * (pow v1.value v2.value) * (v1.value / v2.value)
      ],
      children = [v1, v2]
    }

export
param : String -> Double -> Variable
param paramId = { paramId := Just paramId } . fromDouble

export
nameParam : String -> Nat -> Variable -> Variable
nameParam prefx i p = { paramId := Just (prefx ++ show i) } p

export
backwardVariable : Double -> Variable -> Variable
backwardVariable g v =
    { grad $= (g +),
      children $= zipWith backwardVariable (v.back g)
    } v

export
gradMap : Variable -> List (String, Double)
gradMap v = case v.paramId of
  (Just pid) => (pid, v.grad) :: concatMap gradMap v.children
  Nothing => concatMap gradMap v.children
