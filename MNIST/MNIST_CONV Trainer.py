import numpy as np
from keras.datasets import mnist
import keras.utils as np_utils
from Layers import Den, Convolutional, Reshape
from Activations import Sigmoid
from Loss import mse, mse_prime
from net import train, predict

def preprocess_data(x, y):

    x = x.reshape(len(x), 1, 28, 28)
    x = x.astype("float32") / 255
    y = np_utils.to_categorical(y)
    y = y.reshape(len(y), 10, 1)
    return x, y


# load MNIST from server, limit to 100 images per class since we're not training on GPU
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train[0:1000], y_train[0:1000])
x_test, y_test = preprocess_data(x_test[0:10], y_test[0:10])

print(x_train.shape)
print(y_train.shape)

# neural network
network = [
    Convolutional((1, 28, 28), 3, 5, False),
    Sigmoid(),
    Reshape((5, 26, 26), (5 * 26 * 26, 1)),
    Den(5 * 26 * 26, 100, False),
    Sigmoid(),
    Den(100, 10, False),
    Sigmoid()
]

# train
train(
    network,
    mse,
    mse_prime,
    x_train,
    y_train,
    epochs=20,
    learning_rate=0.1
)


for x, y in zip(x_test, y_test):
    output = predict(network, x)
    print(f"pred: {np.argmax(output)}, true: {np.argmax(y)}")