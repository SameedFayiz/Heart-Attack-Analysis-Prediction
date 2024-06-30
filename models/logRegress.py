import math
import numpy as np


def sigmoid(z):
    return 1 / (1 + math.exp(-z))


def predict_proba(X, weights, bias):
    return [sigmoid(sum(w * x for w, x in zip(weights, row)) + bias) for row in X]


def predict(X, weights, bias):
    return [1 if p >= 0.5 else 0 for p in predict_proba(X, weights, bias)]


def gradient_descent(X, y, learning_rate, num_iterations):
    n_samples, n_features = len(X), len(X[0])
    weights = [0] * n_features
    bias = 0

    for _ in range(num_iterations):
        y_pred = predict_proba(X, weights, bias)
        dw = [0] * n_features
        db = 0

        for i in range(n_samples):
            error = y_pred[i] - y[i]
            for j in range(n_features):
                dw[j] += error * X[i][j]
            db += error

        weights = [w - learning_rate * dw[j] /
                   n_samples for j, w in enumerate(weights)]
        bias -= learning_rate * db / n_samples

    return weights, bias
