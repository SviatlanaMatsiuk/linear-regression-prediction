import matplotlib.pyplot as plt
import numpy as np


def plot_cost_history(cost_history, num_iterations):
    plt.figure(figsize=(10, 6))
    plt.plot(range(num_iterations), cost_history, color='blue')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Cost Function Convergence')
    plt.show()


def plot_predictions(X, y, w, b, X_mean, X_std, features):
    fig, ax = plt.subplots(1, len(features), figsize=(16, 4), sharey=True)
    for i in range(len(ax)):
        ax[i].scatter(X[:, i] * X_std[i] + X_mean[i], y, label='Target', color='blue')

        X_temp = np.zeros_like(X)
        X_temp[:] = X[:]

        predictions = np.dot(X_temp, w) + b
        ax[i].scatter(X[:, i] * X_std[i] + X_mean[i], predictions, label='Prediction', color='red')
        ax[i].set_xlabel(features[i])

    ax[0].set_ylabel("Price")
    ax[0].legend()
    fig.suptitle("Target versus Prediction using Z-score Normalized Model")
    plt.show()
