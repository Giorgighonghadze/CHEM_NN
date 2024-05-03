import numpy as np
from scipy import signal

class layer:
    def __init__(self):

        self.input = None
        self.output = None

    def forward(self, inp):
        pass

    def backward(self, gradient, learning_rate):
        pass


class Dense(layer):

    def __init__(self, input_size, output_size, pred_bool, weights=[0], biases=[0]):
        if pred_bool == 0:
            self.weights = np.random.randn(input_size, output_size)
            self.biases = np.random.randn(1, output_size)
        else:
            self.weights = weights
            self.biases = biases

    def forward(self, inp):

        self.input = inp
        self.output = np.dot(self.input, self.weights) + self.biases
        return self.output

    def backward(self, gradient, learning_rate):
        input_prime = np.dot(gradient, self.weights.T)
        weights_prime = np.dot(self.input.T, gradient)

        self.weights -= weights_prime * learning_rate
        self.biases -= gradient * learning_rate
        return input_prime


class Convolutional(layer):
    def __init__(self, input_shape, kernel_size, depth, Istest, kernels = [], biases = []):
        if Istest == False:
            input_depth, input_height, input_width = input_shape
            self.depth = depth
            self.input_shape = input_shape
            self.input_depth = input_depth
            self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
            self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)
            self.kernels = np.random.randn(*self.kernels_shape)
            self.biases = np.random.randn(*self.output_shape)
        else:
            input_depth, input_height, input_width = input_shape
            self.depth = depth
            self.input_shape = input_shape
            self.input_depth = input_depth
            self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
            self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)
            self.kernels = kernels
            self.biases = biases

    def forward(self, input):
        self.input = input
        self.output = np.copy(self.biases)
        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output[i] += signal.correlate2d(self.input[j], self.kernels[i, j], "valid")
        return self.output

    def backward(self, gradient, learning_rate):
        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input_shape)

        for i in range(self.depth):
            for j in range(self.input_depth):
                kernels_gradient[i, j] = signal.correlate2d(self.input[j], gradient[i], "valid")
                input_gradient[j] += signal.convolve2d(gradient[i], self.kernels[i, j], "full")

        self.kernels -= learning_rate * kernels_gradient
        self.biases -= learning_rate * gradient
        return input_gradient

class Reshape(layer):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, input):
        return np.reshape(input, self.output_shape)

    def backward(self, gradient, learning_rate):
        return np.reshape(gradient, self.input_shape)


class Den(layer):
    def __init__(self, input_size, output_size, Istest, weights = [], biases = []):
        if Istest == False:
            self.weights = np.random.randn(output_size, input_size)
            self.bias = np.random.randn(output_size, 1)
        else:
            self.weights = weights
            self.bias = biases

    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias

    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, output_gradient)
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        return input_gradient