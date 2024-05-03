from Base_Modules.Layers import Den, Convolutional, Reshape
from Base_Modules.Activations import Sigmoid
from Base_Modules.net import predict
import glob as glob
import cv2
import matplotlib.pylab as plt
import pickle



test_structure = glob.glob("../Training Data/Test/16.png")

with open("../data.CHEMweights1", "rb") as file:
    kernels = pickle.load(file)
with open("../data.CHEMweights2", "rb") as file:
    weights2 = pickle.load(file)
with open("../data.CHEMweights3", "rb") as file:
    weights3 = pickle.load(file)
with open("../data.CHEMbiases1", "rb") as file:
    biases1 = pickle.load(file)
with open("../data.CHEMbiases2", "rb") as file:
    biases2 = pickle.load(file)
with open("../data.CHEMbiases3", "rb") as file:
    biases3 = pickle.load(file)


test_structure_org = plt.imread(test_structure[0])
test_structure = cv2.resize(test_structure_org,(50, 50))
test_structure = test_structure[:,:,3]
#test_structure = test_structure / 255
test_structure = test_structure.reshape(1, 50, 50)
print(test_structure.shape)



network = [
    Convolutional((1, 50, 50, ), 5, 3, True, kernels, biases1),
    Sigmoid(),
    Reshape((3, 46, 46), (3 * 46 * 46, 1)),
    Den(3 * 46 * 46, 100, True, weights2, biases2),
    Sigmoid(),
    Den(100, 2, True, weights3, biases3),
    Sigmoid()
]

result = predict(network, test_structure)

print(f"\nAcid is {round((result[0][0]/(result[0][0] + result[1][0])) * 100, 2)}%\n"
      f"Base is {round(result[1][0]/(result[0][0] + result[1][0]) * 100, 2)}%")


plt.imshow(test_structure_org, cmap="Greys")
plt.axis("off")
plt.show()

