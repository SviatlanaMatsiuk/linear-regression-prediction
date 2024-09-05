import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Function to calculate the cost
def calc_cost(X, y, w, b, m):
    predictions = np.dot(X, w) + b
    cost = (1 / (2 * m)) * np.sum((y - predictions) ** 2)
    return cost

def predict(X, w, b):
    return np.dot(X, w) + b

# Load data
data = pd.read_csv('apartment_prices.csv')

features = ['SquareMeters', 'NoOfFloors', 'NoOfRooms', 'DistanceFromCenter']
X = data[features].values
y = data['Price'].values

# Standardizing features
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X = (X - X_mean) / X_std

m, n = X.shape  # m = number of training examples, n = number of features

w = np.zeros(n)  # Initialize weights
b = 0
learning_rate = 0.01
num_iterations = 1000

# List to store cost values
cost_history = []

# Gradient Descent
for i in range(num_iterations):
    predictions = predict(X, w, b)
    dw = -(1 / m) * np.dot(X.T, (y - predictions))
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

plt.figure(figsize=(10, 6))
plt.plot(range(num_iterations), cost_history, color='blue')
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Cost Function Convergence')
plt.show()

# Prediction for new value
new_data = np.array([95, 2, 3, 5])  # Example new data
new_data_standardized = (new_data - X_mean) / X_std
predicted_price = predict(new_data_standardized, w, b)
print(f"Predicted price for {new_data}: {predicted_price}")