import numpy as np

x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,0,0,1])

w1 = np.random.uniform(-1, 1)  # Random value between -1 and 1
w2 = np.random.uniform(-1, 1)
bias = np.random.uniform(-1, 1)
l=0.01

def sigmoid(x):
    return 1/(1 + np.exp(-x)) 
    
for j in range(100000):
    
    for i in range(4):
        ans = x[i][0]*w1 + x[i][1]*w2+bias
        result = sigmoid(ans)

        w1 = w1 + (l * (y[i] - result)*x[i][0])
        w2 = w2 + (l * (y[i] - result)* x[i][1])
        bias= bias + l*(y[i]-result)
    
for i in range(4):
    ans = x[i][0]*w1 + x[i][1]*w2 +bias
    result = sigmoid(ans)
    rounded_number = round(result, 6)


    print(f"{rounded_number} Predicted : {1 if rounded_number>0.5 else 0}")
