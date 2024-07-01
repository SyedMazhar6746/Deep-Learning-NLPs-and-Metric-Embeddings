#!/usr/bin/python3

import torch
import torch.nn as nn
import torch.optim as optim

# Data and parameters, parameter initialization
x = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=torch.float32).view(1, -1)
y_true = torch.tensor([3, 5, 7, 9, 11, 13, 15, 17, 19, 21], dtype=torch.float32).view(1, -1)
x_test = torch.tensor([20], dtype=torch.float32).view(1, 1)

w = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

# Optimization procedure: gradient descent
optimizer = optim.SGD([w, b], lr=0.007)

for i in range(1000):
    # Affine regression model
    y_pred = w * x + b

    diff = (y_true-y_pred)

    # quadratic loss
    loss = torch.mean(diff**2)

    # loss = torch.nn.functional.mse_loss(y_pred, y_true)  # Mean squared error loss

    optimizer.zero_grad()   # Zero gradients
    loss.backward()         # Backward pass
    optimizer.step()        # Optimization step

    if i % 100 == 0:
        print(f'step: {i}, loss: {loss.item()}')

y_result = w * x_test + b
print(f'y_result: {y_result.item()}')