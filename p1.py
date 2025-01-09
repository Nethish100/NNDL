import numpy as np
x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0,1])

w1 = np.random.uniform(-1, 1)  # Random value between -1 and 1
w2 = np.random.uniform(-1, 1)
bias = np.random.uniform(-1, 1)
learning_rate = 0.01

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

for epoch in range(100000):  
    for i in range(4): 
        # Forward pass
        ans = x[i][0] * w1 + x[i][1] * w2 + bias
        result = sigmoid(ans)
        error = y[i] - result
        
        # Update weights and bias using the sigmoid derivative
        delta = error * sigmoid_derivative(result)
        w1 += learning_rate * delta * x[i][0]
        w2 += learning_rate * delta * x[i][1]
        bias += learning_rate * delta

# Testing the trained perceptron
print("Testing the AND gate after training:")
for i in range(4):
    ans = x[i][0] * w1 + x[i][1] * w2 + bias
    result = sigmoid(ans)
    rounded_number = round(result, 10)
    print(f"{rounded_number} Predicted : {1 if rounded_number>0.5 else 0}")
