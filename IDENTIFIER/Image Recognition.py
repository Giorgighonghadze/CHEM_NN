import numpy as np
from Base_Modules.Layers import Den, Convolutional, Reshape
from Base_Modules.Activations import Sigmoid
from Base_Modules.Loss import mse, mse_prime
from Base_Modules.net import train
import glob as glob
import matplotlib.pylab as plt
import pickle
import cv2



data_base = glob.glob('Training Data/TRAIN_SET/Base/*.png')
data_acid = glob.glob('Training Data/TRAIN_SET/Acid/*.png')

train_x, train_y = [], []

for i in range(len(data_base)):
    img = plt.imread(data_base[i])
    img = cv2.resize(img,[50, 50])
    train_y.append(np.array([0,1]))
    train_x.append(np.array(img[:,:,3]))

for i in range(len(data_acid)):
    img = plt.imread(data_acid[i])
    img = cv2.resize(img, [50, 50])
    train_y.append(np.array([1,0]))
    train_x.append(np.array(img[:,:,3]))

def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b

Train_X = np.array(train_x)
Train_Y = np.array(train_y)
x_train, y_train = shuffle_in_unison(Train_X,Train_Y)
x_train = x_train.reshape(221, 1, 50, 50)
y_train = y_train.reshape(221, 2, 1)
print(x_train.shape)
print(y_train.shape)


network = [
    Convolutional((1, 50, 50), 5, 3, False),
    Sigmoid(),
    Reshape((3, 46, 46), (3 * 46 * 46, 1)),
    Den(3 * 46 * 46, 100, False),
    Sigmoid(),
    Den(100, 2, False),
    Sigmoid()
]

# train
train(
    network,
    mse,
    mse_prime,
    x_train,
    y_train,
    epochs=1000,
    learning_rate=0.05
)
print(network[0].kernels.shape)

with open("../data.CHEMweights1", "wb") as file:
    pickle.dump(network[0].kernels, file)
with open("../data.CHEMweights2", "wb") as file:
    pickle.dump(network[3].weights, file)
with open("../data.CHEMweights3", "wb") as file:
    pickle.dump(network[5].weights, file)
with open("../data.CHEMbiases1", "wb") as file:
    pickle.dump(network[0].biases, file)
with open("../data.CHEMbiases2", "wb") as file:
    pickle.dump(network[3].bias, file)
with open("../data.CHEMbiases3", "wb") as file:
    pickle.dump(network[5].bias, file)





