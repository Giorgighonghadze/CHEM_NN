import numpy as np
from Dense import dense
from Network import network
from Activations import activation_layer, tanh, tanh_prime
import pickle


#XOR PREDICTION

x_train = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])



with open("data.XORweights1", "rb") as file:
    weights1 = pickle.load(file)
with open("data.XORweights2", "rb") as file:
    weights2 = pickle.load(file)
with open("data.XORbiases1", "rb") as file:
    biases1 = pickle.load(file)
with open("data.XORbiases2", "rb") as file:
    biases2 = pickle.load(file)

net = network()
net.add(dense(2, 3,True, weights1, biases1))
net.add(activation_layer(tanh, tanh_prime))
net.add(dense(3, 1,True, weights2, biases2))
net.add(activation_layer(tanh, tanh_prime))


out = net.predict(np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]]))
print(out)