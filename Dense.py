from Layers import layer
import numpy as np

class dense(layer):
    
    def __init__(self, input_size, output_size, pred_bool, weights=[0], biases=[0]):
        if pred_bool == 0:
            self.weights = np.random.randn(input_size,output_size)
            self.biases = np.random.randn(1, output_size)
        else:
            self.weights = weights
            self.biases = biases

    def forward(self, input):

        self.input = input
        self.output = np.dot(self.input, self.weights) + self.biases
        return self.output


    def backward(self, gradient, learning_rate):
        input_prime = np.dot(gradient, self.weights.T)
        weights_prime = np.dot(self.input.T, gradient)

        self.weights -= weights_prime * learning_rate
        self.biases -= gradient * learning_rate
        return input_prime


