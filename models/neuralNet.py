import dill
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score


class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward_propagation(self, input):
        pass

    def backward_propagation(self, output_error, learning_rate):
        pass


class FullyConnectedLayer(Layer):
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.rand(n_inputs, n_neurons) - 0.5
        self.bias = np.random.rand(1, n_neurons) - 0.5

    def forward_propagation(self, input):
        self.input = input
        self.output = np.dot(self.input, self.weights) + self.bias

        return self.output

    def backward_propagation(self, output_error, learning_rate):
        m = self.input.shape[-1]
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.reshape(-1, m).T, output_error)
        bias_error = output_error

        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * bias_error

        return input_error


class ActivationLayer(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward_propagation(self, input):
        self.input = input
        self.output = self.activation(input)
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        return self.activation_prime(self.input) * output_error


class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None
        self.errors = []
        self.accuracy = []

    def add(self, layer):
        self.layers.append(layer)

    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    def predict(self, input):
        result = []
        for i in range(len(input)):
            output = input[i]

            for layer in self.layers:
                output = layer.forward_propagation(output)

            result.append(output[0, 0])

        return np.array(result)

    def fit(self, X, y, epochs, learning_rate, printCallback=None):
        errors = []
        accuracy = []
        for epoch in range(epochs):
            err = 0

            for i in range(len(X)):
                output = X[i]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                err += self.loss(y[i], output)
                error = self.loss_prime(y[i], output)

                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            err /= X.shape[0]

            y_pred = self.predict(X)
            accuracy.append(accuracy_score(y, (y_pred > 0.5).astype(int)))
            errors.append(err)

            if epoch % 100 == 0 or epoch == (epochs-1):
                print('epoch %d/%d   error=%f' % (epoch+1, epochs, err))
                if printCallback:
                    printCallback(epoch, epochs, err)
        self.errors = errors
        self.accuracy = accuracy
        return {"loss": np.array(errors), "accuracy": np.array(accuracy)}


def saveModel():

    def BinaryCrossEntropy(y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        return -np.mean(y_true * np.log(y_pred) + (1-y_true) * np.log(1-y_pred))

    def BinaryCrossEntropyPrime(y_true, y_pred):
        return (y_pred - y_true) / y_true.size

    # Activation functions

    def Relu(inputs):
        return (inputs > 0) * inputs

    def Relu_prime(inputs):
        return (inputs > 0) * 1

    def sigmoid(inputs):
        return 1 / (1 + np.exp(-inputs))

    def sigmoid_prime(inputs):
        return sigmoid(inputs) * (1-sigmoid(inputs))

    data = pd.read_csv("./dataset/heart.csv")

    # data = data.drop_duplicates()
    X = np.array(data.drop('output', axis=1))  # Features
    y = np.array(data['output'])  # Target

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    object = StandardScaler()
    X_train = object.fit_transform(X_train)
    X_test = object.transform(X_test)

    net = NeuralNetwork()
    net.add(FullyConnectedLayer(13, 13))
    net.add(ActivationLayer(Relu, Relu_prime))
    net.add(FullyConnectedLayer(13, 26))
    net.add(ActivationLayer(Relu, Relu_prime))
    net.add(FullyConnectedLayer(26, 1))
    net.add(ActivationLayer(sigmoid, sigmoid_prime))
    net.use(BinaryCrossEntropy, BinaryCrossEntropyPrime)

    net.fit(X_train, y_train, epochs=1000, learning_rate=0.002)

    filepath = "./models/static/my_neural_net.sav"
    dill.dump(net, open(filepath, 'wb'))


def loadModel():
    filepath = "./models/static/my_neural_net.sav"
    return dill.load(open(filepath, 'rb'))


# saveModel()
