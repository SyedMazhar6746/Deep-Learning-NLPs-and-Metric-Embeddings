#!/usr/bin/python3

import numpy as np 

x =      np.array([1, 2, 3, 4,  5,  6,  7,  8,  9, 10], dtype=np.float32).reshape(1, -1)
y_true = np.array([3, 5, 7, 9, 11, 13, 15, 17, 19, 21], dtype=np.float32).reshape(1, -1)

x_test = np.array([20]).reshape(1, 1)

w = np.random.randn(1).reshape(1, 1)
b = np.random.randn(1).reshape(1, 1)


# model prediction
def forward(x, w, b):
    return np.dot(w, x) + b

# loss function
def loss_fn(y_pred, y_true):
    return ((y_pred - y_true)**2).mean()

# gradient
def gradient_w(y_pred, y_true, x):
    return (2*x*(y_pred-y_true)).mean()

def gradient_b(y_pred, y_true):
    return (2*(y_pred-y_true)).mean()

learning_rate = 0.007

for i in range(50):
    y_pred = forward(x, w, b)

    loss = loss_fn(y_pred, y_true)

    dw = gradient_w(y_pred, y_true, x)
    db = gradient_b(y_pred, y_true)

    w -= learning_rate*dw 
    b -= learning_rate*db

    if i%2 == 0:
        print(f'step : {i}, loss : {loss}')

y_result = forward(x_test, w, b)
print(f'y_result : {y_result}')