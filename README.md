# feedforward-neural-network
A basic implementation of a Multi-Layer Perceptron in C++.

The purpose of this project is to understand the working principle of neural networks by building a network from scratch and feeding it with custom data.

Please check source files for more details:

- You can find the implementation of various parts of the neural network, going from a simple `neuron`, to a `layer`, to the whole `network`.
- Both weight and bias neurons are implemented.
- The topology (depth of the network, along with the number of neurons in each layer) could be specified in the `Data.txt` file.
- The data generator actually simulates the behavior of a XOR gate and outputs training data in a non-generic format (Not CSV).

Ressources:
- https://www.youtube.com/watch?v=KkwX7FkLfug
