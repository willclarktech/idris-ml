# idris-ml

This is a learning project as part of my quest to find deep learning tools that don't infuriate me.

## Tensor

`Tensor` is a recursively-defined data type which holds a scalar value or a fixed-length vector of `Tensor`s. `Scalar`, `Vector`, and `Matrix` are type aliases for `Tensor` with different ranks.

## Math

This module defines various mathematical functions, including tensor operations.

## Layer

`Layer` is a data type which can represent various sorts of neural network layers. This module also defines how to perform a forward pass for each sort of layer.

## Network

`Network` is a data type composed of one or more layers. This module also defines how to perform a forward pass through the entire network, as well as how to calculate a loss value given some data and a loss function.

## Variable

`Variable` represents a value which can form a computational graph, store information about gradients, and backpropagate those gradients. It is an instance of `Num`, `Fractional` etc so any computations consisting of mathematical functions defined for those interfaces (as well as `exp`, `log`, and `pow`) can be backpropagated via automatic differentation.

## Backprop

This module defines the training loop for supervised learning using backpropagation to successively adjust the parameters of a neural network.
