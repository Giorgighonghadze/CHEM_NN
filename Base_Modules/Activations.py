import numpy as np
from Base_Modules.Layers import layer

class activation_layer(layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        self.input = input
        self.output = self.activation(input)
        return self.output

    def backward(self, gradient, learning_rate):
        return self.activation_prime(self.input) * gradient


def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1 - np.tanh(x) ** 2

def ReLU(x):
    if x.all() <= 0:

        return 0
    else:
        return x

def ReLU_prime(x):
    if x.all() <= 0:
        return 0
    else:
        return 1

class Sigmoid(activation_layer):
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def sigmoid_prime(x):
            s = sigmoid(x)
            return s * (1 - s)

        super().__init__(sigmoid, sigmoid_prime)
