"""Logistic"""
import numpy as np
import matplotlib.pyplot as plt

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Cost function (Log Loss)
def compute_cost(X, y, theta):
    m = len(y)
    h = sigmoid(X @ theta)
    epsilon = 1e-5
    cost = -(1/m) * np.sum(y*np.log(h+epsilon) + (1-y)*np.log(1-h+epsilon))
    return cost

# Gradient descent
def gradient_descent(X, y, theta, alpha, num_iterations):
    m = len(y)
    cost_history = []

    for _ in range(num_iterations):
        h = sigmoid(X @ theta)
        gradient = (1/m) * (X.T @ (h - y))
        theta -= alpha * gradient
        cost_history.append(compute_cost(X, y, theta))

    return theta, cost_history

# Predict 0 or 1
def predict(X, theta):
    prob = sigmoid(X @ theta)
    return (prob >= 0.5).astype(int)

# ------------------ User Input ------------------
n_features = int(input("Enter number of features: "))
alpha = float(input("Enter learning rate (alpha): "))
num_iterations = int(input("Enter number of iterations: "))
m = int(input("Enter number of data points: "))

X_data = []
y_data = []

for i in range(m):
    x = []
    for j in range(n_features):
        a = float(input(f"Enter feature {j+1} value for data point {i+1}: "))
        x.append(a)
    b = int(input(f"Enter target value (0/1) for data point {i+1}: "))
    X_data.append(x)
    y_data.append(b)

# Convert to numpy arrays
X_data = np.array(X_data)
y_data = np.array(y_data)

# Add bias (column of ones)
X = np.c_[np.ones((m, 1)), X_data]
y = y_data.reshape(m, 1)

# Initialize theta
theta_init = np.zeros((n_features + 1, 1))

# Run gradient descent
theta, cost_history = gradient_descent(X, y, theta_init, alpha, num_iterations)

# ------------------ Output ------------------
print("\nFinal theta (parameters):\n", theta)
print("Final cost:", cost_history[-1])

# Predict for all training data
y_pred = predict(X, theta)

# Create a table of original data + predictions
import pandas as pd
data_table = pd.DataFrame(X_data, columns=[f"Feature_{i+1}" for i in range(n_features)])
data_table['Actual'] = y_data
data_table['Predicted'] = y_pred
print("\nData with predictions:\n")
print(data_table)

# Plot cost over iterations
plt.plot(range(len(cost_history)), cost_history, 'b-')
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost vs Iterations (Logistic Regression)")
plt.show()


"""Enter number of features: 2
Enter learning rate (alpha): .01
Enter number of iterations: 200
Enter number of data points: 20
Enter feature 1 value for data point 1: 2
Enter feature 2 value for data point 1: 3
Enter target value (0/1) for data point 1: 0
Enter feature 1 value for data point 2: 1
Enter feature 2 value for data point 2: 5
Enter target value (0/1) for data point 2: 0
Enter feature 1 value for data point 3: 2
Enter feature 2 value for data point 3: 6
Enter target value (0/1) for data point 3: 0
Enter feature 1 value for data point 4: 3
Enter feature 2 value for data point 4: 7
Enter target value (0/1) for data point 4: 0
Enter feature 1 value for data point 5: 4
Enter feature 2 value for data point 5: 6
Enter target value (0/1) for data point 5: 0
Enter feature 1 value for data point 6: 6
Enter feature 2 value for data point 6: 5
Enter target value (0/1) for data point 6: 1
Enter feature 1 value for data point 7: 7
Enter feature 2 value for data point 7: 6
Enter target value (0/1) for data point 7: 1
Enter feature 1 value for data point 8: 8
Enter feature 2 value for data point 8: 5
Enter target value (0/1) for data point 8: 1
Enter feature 1 value for data point 9: 6
Enter feature 2 value for data point 9: 7
Enter target value (0/1) for data point 9: 1
Enter feature 1 value for data point 10: 7
Enter feature 2 value for data point 10: 8
Enter target value (0/1) for data point 10: 1
Enter feature 1 value for data point 11: 8
Enter feature 2 value for data point 11: 7
Enter target value (0/1) for data point 11: 1
Enter feature 1 value for data point 12: 9
Enter feature 2 value for data point 12: 6
Enter target value (0/1) for data point 12: 1
Enter feature 1 value for data point 13: 10
Enter feature 2 value for data point 13: 7
Enter target value (0/1) for data point 13: 1
Enter feature 1 value for data point 14: 11
Enter feature 2 value for data point 14: 8
Enter target value (0/1) for data point 14: 1
Enter feature 1 value for data point 15: 12
Enter feature 2 value for data point 15: 6
Enter target value (0/1) for data point 15: 1
Enter feature 1 value for data point 16: 13
Enter feature 2 value for data point 16: 7
Enter target value (0/1) for data point 16: 1
Enter feature 1 value for data point 17: 14
Enter feature 2 value for data point 17: 8
Enter target value (0/1) for data point 17: 1
Enter feature 1 value for data point 18: 15
Enter feature 2 value for data point 18: 7
Enter target value (0/1) for data point 18: 1
Enter feature 1 value for data point 19: 16
Enter feature 2 value for data point 19: 9
Enter target value (0/1) for data point 19: 1
Enter feature 1 value for data point 20: 17
Enter feature 2 value for data point 20: 5
Enter target value (0/1) for data point 20: 1

Final theta (parameters):
 [[-0.09775302]
 [ 0.57754195]
 [-0.3416437 ]]
Final cost: 0.18228037929916635

Data with predictions:

    Feature_1  Feature_2  Actual  Predicted
0         2.0        3.0       0          1
1         1.0        5.0       0          0
2         2.0        6.0       0          0
3         3.0        7.0       0          0
4         4.0        6.0       0          1
5         6.0        5.0       1          1
6         7.0        6.0       1          1
7         8.0        5.0       1          1
8         6.0        7.0       1          1
9         7.0        8.0       1          1
10        8.0        7.0       1          1
11        9.0        6.0       1          1
12       10.0        7.0       1          1
13       11.0        8.0       1          1
14       12.0        6.0       1          1
15       13.0        7.0       1          1
16       14.0        8.0       1          1
17       15.0        7.0       1          1
18       16.0        9.0       1          1
19       17.0        5.0       1          1
"""