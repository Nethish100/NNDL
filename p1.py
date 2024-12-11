import numpy as np

# Input data
x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])

# Random initialization of weights and bias
#np.random.seed(42)  # For reproducibility, remove this line if you want completely random values each run
w1 = np.random.uniform(-1, 1)  # Random value between -1 and 1
w2 = np.random.uniform(-1, 1)
bias = np.random.uniform(-1, 1)
learning_rate = 0.01

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Training loop
for epoch in range(100000):  # Train for 100,000 epochs
    for i in range(4):  # Iterate over all training examples
        # Calculate the weighted sum and apply sigmoid activation
        weighted_sum = x[i][0] * w1 + x[i][1] * w2 + bias
        result = sigmoid(weighted_sum)

        # Update weights and bias
        w1 += learning_rate * (y[i] - result) * x[i][0]
        w2 += learning_rate * (y[i] - result) * x[i][1]
        bias += learning_rate * (y[i] - result)

# Testing loop
for i in range(4):
    weighted_sum = x[i][0] * w1 + x[i][1] * w2 + bias
    result = sigmoid(weighted_sum)
    rounded_number = round(result, 6)
    print(f"{rounded_number} Predicted: {1 if rounded_number > 0.5 else 0}")
