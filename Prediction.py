from Base_Modules.Layers import Dense
from Base_Modules.Network import network
from Base_Modules.Activations import activation_layer, tanh, tanh_prime
from keras.datasets import mnist
import keras.utils as np_utils
from matplotlib import pyplot as plt
import pickle

with open("data.weights1", "rb") as file:
    weights_1 = pickle.load(file)
with open("data.weights2", "rb") as file:
    weights_2 = pickle.load(file)
with open("data.biases1", "rb") as file:
    biases_1 = pickle.load(file)
with open("data.biases2", "rb") as file:
    biases_2 = pickle.load(file)


(x_train1, y_train1), (x_test1, y_test1) = mnist.load_data()

x_test = x_test1.reshape(x_test1.shape[0], 1, 28*28)
x_test = x_test.astype('float32')
x_test /= 255
y_test = np_utils.to_categorical(y_test1)

net = network()
net.add(Dense(28*28, 10,True, weights_1, biases_1))
net.add(activation_layer(tanh, tanh_prime))
net.add(Dense(10, 10,True, weights_2, biases_2))
net.add(activation_layer(tanh, tanh_prime))

out = net.predict(x_test[0:10])
print(out)

for i in range(0,10):
    plt.imshow(x_test1[i])
    plt.show()


