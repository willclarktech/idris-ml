# Getting started

A text walkthrough from a first tensor to a trained model — a Jupyter-independent
companion to the [notebook tutorials](../../packages/jupyter/README.md). Every snippet here
is the real current API; the complete, compiling version of the training example is
[`packages/idris-ml-examples/src/Example/Supervised.idr`](../../packages/idris-ml-examples/src/Example/Supervised.idr).

New to the *why*? Read [Why idris-ml](why-idris-ml.md) first.

## 0. Build and run

Inside the dev shell (`nix develop`, or direnv-loaded):

```bash
make backend             # build the C tape backend (no external deps)
make example-supervised  # train the 3-class classifier end-to-end
```

`make backend` builds `libidrisml.dylib`; `make install` installs the core lib + gym so
examples and tests can link. For the optional backends: `make BACKEND=torch backend` /
`make BACKEND=mlx backend`.

Examples don't hardcode device or dtype — they reference `ExampleDevice` / `ExampleDType`
(aliased `Ex` / `F` below) from the build-generated `BuildConfig.idr`. A `tape` build
gives you `TapeExecutor` + `F64`; switching backends is just a different `make install`.

## 1. A first tensor

Shape, executor, dtype, and grad-mode all live in the type:
`Tensor (dims : Vect rank Nat) (0 ex : Executor) (0 dt : DType) (0 g : GradMode)`. Build
one from an `InitSpec`:

```idris
x <- tensor {dims=[2,3]} (Const 0.5)            -- Tensor [2,3] Ex F NoGrad, all 0.5
v <- tensor {dims=[3]}   (FromVect [1.0, 2.0, 3.0])
s <- tadd v v                                    -- elementwise; shapes must match
```

`FromVect`'s length is tied to the shape at compile time, so a data/shape mismatch is a
type error — not a runtime surprise. A learnable parameter uses `param` (it registers with
the optimizer):

```idris
w <- param "w" (Normal 0.0 0.02)                -- registered under the name "w"
```

## 2. A first model

Models are records of `Nn` layers. A `Seq` chains `Module`s with `~~>`; its type pins only
the endpoints, hidden dims are existential. Layers are built in the `Init` monad:

```idris
Model : Type
Model = Seq 2 3 Ex F WithGrad

mkModel : Init Model
mkModel = do
  l1 <- linear {i=2} {o=10}
  l2 <- linear {i=10} {o=3}
  pure (l1 ~~> reluA ~~> l2 ~~> Nil)            -- 10 (l1 out) must unify with 10 (l2 in)
```

`runInitL mkModel` populates the C parameter registry, deriving names from the scope path
(the PyTorch `state_dict` convention) — there's no `autoName`, and no paramId to pass.

## 3. A first training run

The model is a **single-owner linear resource** threaded through `Control.Linear.LIO.L IO`.
The `fit` driver owns the epoch loop, early stopping, checkpointing, and NaN handling; you
hand it a loss function that consumes-and-threads the model. Data flows through a
`DataStream` of batched `(input, target)` tensors:

```idris
-- collate samples into one [b, i] / [b, o] batch (full-batch here, b=5):
buildStream : IO (DataStream (Tensor [5,2] Ex F NoGrad, Tensor [5,3] Ex F NoGrad))
buildStream = do
  s <- stream NoShuffle (fromIndexed 5 sampleAt)   -- sampleAt : Nat -> IO (x, y)
  pure (batched {b=5} {i=2} {o=3} s)

-- the loss fn: consume the linear model, forward it, return the scalar loss
-- (banged) beside the rebuilt model so fit can thread it through the epoch:
lossFn : (1 _ : Model) -> (Tensor [5,2] Ex F NoGrad, Tensor [5,3] Ex F NoGrad)
      -> L IO {use=1} (LPair (!* (Tensor [] Ex F WithGrad)) Model)
lossFn model (x, tgt) = do
  (MkBang out # model') <- forwardSeq {b=5} model (retypeGrad x)
  loss <- tnllLossMeanL {b=5} {n=3} out (retypeGrad tgt)
  pure1 (MkBang loss # model')
```

Then wire it together (this runs in `L IO`; `main : IO` re-enters via `run`):

```idris
model <- runInitL mkModel
opt   <- liftIO1 (adam 0.01 defaultOpts)
bs    <- liftIO1 buildStream
(MkBang (epochs, finalLoss) # trained) <-
  fitSupervised opt lossFn bs (simpleConfig 1000) model
```

`fitSupervised` runs zero_grad → backward → clip → step each epoch — you never call
`trainStep` yourself for the supervised case. `adam`/`sgd`/`adamW`/`rmsprop` are the four
optimizer constructors; `defaultOpts` carries PyTorch's defaults (record-update to change
`beta1`/`eps`/`clip`).

## 4. Evaluate

`eval` consumes the trained model and retypes it `WithGrad → NoGrad`, so the result runs
genuinely tape-free and can't be fed back into training (compile error):

```idris
infer <- eval trained
ein   <- liftIO1 evalInput                        -- Tensor [5,2] Ex F NoGrad
(MkBang predB # infer') <- forwardSeq {b=5} infer ein
discard infer'
-- read predictions off predB (e.g. argmax per row)
```

## Where to next

- [Why idris-ml](why-idris-ml.md) — the full safety story (shape, device, multi-backend,
  grad-mode, dtype) vs PyTorch / TF1+JAX / Haskell.
- [PyTorch mapping](pytorch-mapping.md) — concept-by-concept translation table.
- [idris-transformers](idris-transformers.md) — load real HuggingFace BERT / GPT-2 /
  Llama checkpoints with `fromPretrained`.
- [Notebook tutorials](../../packages/jupyter/README.md) — the same path, interactively.
