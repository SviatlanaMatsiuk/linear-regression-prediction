import numpy as np
from model import calc_cost, predict


def gradient_descent(X, y, w, b, learning_rate, num_iterations):
    m, n = X.shape
    cost_history = []

    for i in range(num_iterations):
        predictions = predict(X, w, b)

        # Compute gradients
        dw = -(1 / m) * np.dot(X.T, (y - predictions))
        db = -(1 / m) * np.sum(y - predictions)

        # Update weights and bias
        w -= learning_rate * dw
        b -= learning_rate * db

        # Calculate and store cost
        cost = calc_cost(X, y, w, b, m)
        cost_history.append(cost)

        # Optional: Print cost every 100 iterations
        if i % 100 == 0:
            print(f"Iteration {i}: Cost {cost}")

    return w, b, cost_history
