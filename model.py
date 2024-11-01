import numpy as np

def calc_cost(X, y, w, b, m):
    predictions = np.dot(X, w) + b
    cost = (1 / (2 * m)) * np.sum((y - predictions) ** 2)
    return cost

def predict(X, w, b):
    return np.dot(X, w) + b
