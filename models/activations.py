import numpy as np


# Loss functions
def MeanSquaredError(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def MeanSquaredErrorPrime(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size


def BinaryCrossEntropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(y_true * np.log(y_pred) + (1-y_true) * np.log(1-y_pred))


def BinaryCrossEntropyPrime(y_true, y_pred):
    return (y_pred - y_true) / y_true.size


# Activation functions
def Relu(inputs):
    def f(x): return (x > 0) * x
    return f(inputs)


def Relu_prime(inputs):
    def f(x): return (x > 0) * 1
    return f(inputs)


def sigmoid(inputs):
    return 1 / (1 + np.exp(-inputs))


def sigmoid_prime(inputs):
    return sigmoid(inputs) * (1-sigmoid(inputs))
