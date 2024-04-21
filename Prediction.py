import numpy as np
from Dense import dense
from Network import network
from Activations import activation_layer, tanh, tanh_prime
from Trainer import x_test, x_test1
import pickle
from matplotlib import pyplot as plt

with open("data.weights1", "rb") as file:
    weights1 = pickle.load(file)
with open("data.weights2", "rb") as file:
    weights2 = pickle.load(file)
with open("data.biases1", "rb") as file:
    biases1 = pickle.load(file)
with open("data.biases1", "rb") as file:
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


#XOR PREDICTION

# net = network()
# net.add(dense(2, 3,True, weights1, biases1))
# net.add(activation_layer(tanh, tanh_prime))
# net.add(dense(3, 1,True, weights2, biases2))
# net.add(activation_layer(tanh, tanh_prime))
#
#
# out = net.predict(np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]]))
# print(out)