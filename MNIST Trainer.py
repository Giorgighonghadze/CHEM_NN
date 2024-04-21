from Dense import dense
from Network import network
from Activations import activation_layer, tanh, tanh_prime, ReLU_prime, ReLU
from Loss import mse, mse_prime
from keras.datasets import mnist
import keras.utils as np_utils
import pickle


#MNIST Trainer


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
net.train(x_train[0:5000], y_train[0:5000], epochs=1000, learning_rate=0.1)

with open("data.MNISTweights1", "wb") as file:
    pickle.dump(net.layers[0].weights, file)
with open("data.MNISTweights2", "wb") as file:
    pickle.dump(net.layers[2].weights, file)
with open("data.MNISTbiases1", "wb") as file:
    pickle.dump(net.layers[0].biases, file)
with open("data.MNISTbiases2", "wb") as file:
    pickle.dump(net.layers[2].biases, file)






