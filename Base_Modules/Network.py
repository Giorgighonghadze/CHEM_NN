class network():
    def __init__(self):

        self.layers = []
        self.loss = None
        self.loss_prime = None

    def add(self, layer):
        self.layers.append(layer)

    def use(self, loss, loss_prime):

        self.loss = loss
        self.loss_prime = loss_prime

    def predict(self, input):

        samples = len(input)
        result = []

        for i in range(samples):
            output = input[i]

            for layer in self.layers:
                output = layer.forward(output)
            result.append(output)

        return result

    def train(self, x_train, y_train, epochs, learning_rate):


        for i in range(epochs):
            err = 0

            for j in range(len(x_train)):

                output = x_train[j]

                for layer in self.layers:

                    output = layer.forward(output)

                err += self.loss(y_train[j], output)

                error = self.loss_prime(y_train[j],output)
                for layer in reversed(self.layers):
                    error = layer.backward(error, learning_rate)

            err /= len(x_train)
            print('epoch %d/%d   error = %f' % (i + 1, epochs, err))