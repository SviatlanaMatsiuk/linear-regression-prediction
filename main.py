import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Function to calculate the cost
def calc_cost(X, y, w, b, m):
    predictions = w * X + b
    cost = (1 / (2 * m)) * np.sum((y - predictions) ** 2)
    return cost

def predict(X, w, b):
    return w * X + b

# Load data
data = pd.read_csv('apartment_prices.csv')

# Standardizing features
X = data['SquareMeters'].values
X_mean = np.mean(X)
X_std = np.std(X)
X = (X - X_mean) / X_std
y = data['Price'].values

w = 0
b = 0
learning_rate = 0.01
num_iterations = 1000
m = len(y)  # number of training examples

# List to store cost values
cost_history = []

# Gradient Descent
for i in range(num_iterations):
    predictions = predict(X, w, b)
    dw = -(1 / m) * np.sum(X * (y - predictions))
    db = -(1 / m) * np.sum(y - predictions)
    w -= learning_rate * dw
    b -= learning_rate * db

    # Calculate and store the cost
    cost = calc_cost(X, y, w, b, m)
    cost_history.append(cost)

    if i % 100 == 0:
        print(f"Iteration {i}: Cost {cost}")

# Display final parameters
print(f"Final parameters: w = {w}, b = {b}")

# Plotting the data and the regression line
plt.figure(figsize=(14, 5))

# Plot 1: Data and regression line
plt.subplot(1, 2, 1)
plt.scatter(X * X_std + X_mean, y)  # convert back to original scale for plotting
plt.plot(X * X_std + X_mean, predict(X, w, b), color='red')  # regression line
plt.xlabel('Square Meters')
plt.ylabel('Price')
plt.title('Apartment Prices vs. Square Meters')

# Plot 2: Convergence plot (cost vs iterations)
plt.subplot(1, 2, 2)
plt.plot(range(num_iterations), cost_history, color='blue')
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Cost Function Convergence')

plt.tight_layout()
plt.show()

# Prediction for new value
sqm = 95
sqm_standardized = (sqm - X_mean) / X_std
predicted_price = predict(sqm_standardized, w, b)
print(f"Predicted price for {sqm} square meters: {predicted_price}")
