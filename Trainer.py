import numpy as np
from Dense import dense
from Network import network
from Activations import activation_layer, tanh, tanh_prime, ReLU_prime, ReLU
from Loss import mse, mse_prime
from keras.datasets import mnist
import keras.utils as np_utils
from matplotlib import pyplot as plt
import pickle


(x_train1, y_train1), (x_test1, y_test1) = mnist.load_data()

# reshape and normalize input data
x_train = x_train1.reshape(x_train1.shape[0], 1, 28*28)
x_train = x_train.astype('float32')
x_train /= 255
# encode output which is a number in range [0,9] into a vector of size 10
# e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
y_train = np_utils.to_categorical(y_train1)

# same for test data : 10000 samples
x_test = x_test1.reshape(x_test1.shape[0], 1, 28*28)
x_test = x_test.astype('float32')
x_test /= 255
y_test = np_utils.to_categorical(y_test1)

net = network()
net.add(dense(28 * 28, 10, False))
net.add(activation_layer(tanh, tanh_prime))
net.add(dense(10, 10, False))
net.add(activation_layer(tanh, tanh_prime))


net.use(mse, mse_prime)
net.train(x_train[0:1000], y_train[0:1000], epochs=100, learning_rate=0.1)

with open("data.weights1", "wb") as file:
    pickle.dump(net.layers[0].weights, file)
with open("data.weights2", "wb") as file:
    pickle.dump(net.layers[2].weights, file)
with open("data.biases1", "wb") as file:
    pickle.dump(net.layers[0].biases, file)
with open("data.biases1", "wb") as file:
    pickle.dump(net.layers[2].biases, file)


with open("data.weights1", "rb") as file:
    weights1 = pickle.load(file)
with open("data.weights2", "rb") as file:
    weights2 = pickle.load(file)
with open("data.biases1", "rb") as file:
    biases1 = pickle.load(file)
with open("data.biases1", "rb") as file:
    biases2 = pickle.load(file)

print(weights1, weights2, biases1, biases2)



#XOR TRAINER

# x_train = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
# y_train = np.array([[[0]], [[1]], [[1]], [[0]]])
#
# net = network()
# net.add(dense(2, 3, False))
# net.add(activation_layer(tanh, tanh_prime))
# net.add(dense(3, 1, False))
# net.add(activation_layer(tanh, tanh_prime))
#
#
# net.use(mse, mse_prime)
# net.train(x_train, y_train, epochs=1000, learning_rate=0.1)
#
# weights1 = net.layers[0].weights
# weights2 = net.layers[2].weights
# biases1 = net.layers[0].biases
# biases2 = net.layers[2].biases
#
#
# print("\n", weights1, "\n\n", weights2, "\n\n", biases1,"\n\n", biases2)


