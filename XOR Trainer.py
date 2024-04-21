import numpy as np
from Layers import Dense
from Network import network
from Activations import activation_layer, tanh, tanh_prime, ReLU, ReLU_prime
from Loss import mse, mse_prime
import pickle


#XOR TRAINER

x_train = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

net = network()
net.add(Dense(2, 3, False))
net.add(activation_layer(tanh, tanh_prime))
net.add(Dense(3, 1, False))
net.add(activation_layer(tanh, tanh_prime))


net.use(mse, mse_prime)
net.train(x_train, y_train, epochs=1000, learning_rate=0.1)

weights1 = net.layers[0].weights
weights2 = net.layers[2].weights
biases1 = net.layers[0].biases
biases2 = net.layers[2].biases


print("\n", weights1, "\n\n", weights2, "\n\n", biases1,"\n\n", biases2)


with open("data.XORweights1", "wb") as file:
    pickle.dump(net.layers[0].weights, file)
with open("data.XORweights2", "wb") as file:
    pickle.dump(net.layers[2].weights, file)
with open("data.XORbiases1", "wb") as file:
    pickle.dump(net.layers[0].biases, file)
with open("data.XORbiases2", "wb") as file:
    pickle.dump(net.layers[2].biases, file)
