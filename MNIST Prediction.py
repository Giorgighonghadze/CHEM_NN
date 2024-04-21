import numpy as np
from Dense import dense
from Network import network
from Activations import activation_layer, tanh, tanh_prime
from keras.datasets import mnist
import keras.utils as np_utils
from matplotlib import pyplot as plt
import pickle



#MNIST Prediction

(x_train1, y_train1), (x_test1, y_test1) = mnist.load_data()

x_test = x_test1.reshape(x_test1.shape[0], 1, 28*28)
x_test = x_test.astype('float32')
x_test /= 255
y_test = np_utils.to_categorical(y_test1)


with open("data.MNISTweights1", "rb") as file:
    weights1 = pickle.load(file)
with open("data.MNISTweights2", "rb") as file:
    weights2 = pickle.load(file)
with open("data.MNISTbiases1", "rb") as file:
    biases1 = pickle.load(file)
with open("data.MNISTbiases2", "rb") as file:
    biases2 = pickle.load(file)

net = network()
net.add(dense(28*28, 10,True, weights1, biases1))
net.add(activation_layer(tanh, tanh_prime))
net.add(dense(10, 10,True, weights2, biases2))
net.add(activation_layer(tanh, tanh_prime))

out = net.predict(x_test[0:10])
print(out)

for i in range(0,10):
    plt.imshow(x_test1[i])
    plt.show()


