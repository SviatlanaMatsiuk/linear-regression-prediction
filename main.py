import numpy as np

from data_processing import load_and_preprocess_data
from gradient_descent import gradient_descent
from model import predict
from visualization import plot_cost_history, plot_predictions

# Load and preprocess data
features = ['SquareMeters', 'NoOfFloors', 'NoOfRooms', 'DistanceFromCenter']
X, y, X_mean, X_std = load_and_preprocess_data('apartment_prices.csv', features, 'Price')

# Initialize parameters
m, n = X.shape
w = np.zeros(n)
b = 0
learning_rate = 0.01
num_iterations = 1000

# Run gradient descent
w, b, cost_history = gradient_descent(X, y, w, b, learning_rate, num_iterations)

# Display final parameters
print(f"Final parameters: w = {w}, b = {b}")

# Plot cost history
plot_cost_history(cost_history, num_iterations)

# Predict for a new data point
new_data = [95, 2, 3, 5]
new_data_standardized = (new_data - X_mean) / X_std
predicted_price = predict(new_data_standardized, w, b)
print(f"Predicted price for {new_data}: {predicted_price}")

# Plot predictions vs targets
plot_predictions(X, y, w, b, X_mean, X_std, features)
