#!/usr/bin/python3

#!/usr/bin/python3

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import data_1
import matplotlib.pyplot as plt
import numpy as np

class PTDeep(nn.Module):
    def __init__(self, config, activation_func=nn.ReLU()):
        super(PTDeep, self).__init__()
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        self.activations = []
        
        for i in range(len(config) - 1):
            weight = nn.Parameter(torch.randn(config[i], config[i + 1]))
            bias = nn.Parameter(torch.randn(1, config[i + 1]))
            self.weights.append(weight)
            self.biases.append(bias)
            if i < len(config) - 2 or True:
                self.activations.append(activation_func)

    def forward(self, X):
        for i in range(len(self.weights)):
            X = torch.matmul(X, self.weights[i]) + self.biases[i]
            if i < len(self.weights) - 1:
                X = self.activations[i](X)
        return X

    def count_params(self):
        total_params = 0
        for name, param in self.named_parameters():
            print(f'Parameter: {name}, Size: {param.size()}')
            total_params += param.numel()
        print(f'Total Parameters: {total_params}')


def softmax(logits):
    exp_logits = torch.exp(logits - torch.max(logits, dim=1, keepdim=True)[0])
    # exp_logits = torch.exp(logits)
    return exp_logits / torch.sum(exp_logits, dim=1, keepdim=True)

def cross_entropy_loss(y_pred, y_true, weights, lambda_reg):
    # print('sizes', y_pred[1, :], y_true[1, :])
    # cross_entropy = -torch.sum(y_true * torch.log(y_pred + 1e-10)) / y_true.size(0)
    cross_entropy = -torch.mean(torch.log(torch.sum(y_true * (y_pred + 1e-10), axis=1)))
    # cross_entropy = -torch.sum(y_true * torch.log(y_pred)) / y_true.size(0)
    # print('size of weights', weights)
    # regularization = (0.5 * lambda_reg * sum(torch.sum(param ** 2) for param in weights))
    regularization = (0.5 * lambda_reg * sum(torch.sum(param ** 2) for param in weights)) 
    # regularization /= sum(1 for _ in weights)  # Convert the generator to a list before calculating length
    
    # print('cross', cross_entropy)
    # print('regularization', regularization)
    return cross_entropy + regularization

# def cross_entropy_loss(y_pred, y_true, weights, lambda_reg=1e-3):
#     cross_entropy = nn.functional.cross_entropy(y_pred, y_true)
#     regularization = 0.5 * lambda_reg * sum(torch.sum(param ** 2) for param in weights)
#     print(f'combined loss {(cross_entropy + regularization).dtype}')
#     return cross_entropy + regularization

def train(model, X, Yoh_, param_niter, param_delta, lambda_reg):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=param_delta, weight_decay=lambda_reg)
    # optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=lambda_reg)
    for i in range(param_niter):

        optimizer.zero_grad()
        # logits = model(X)
        # loss = cross_entropy_loss(logits, torch.max(Yoh_, 1)[1], model.weights, lambda_reg)
        logits = model(X)
        # probs = softmax(logits)
        loss = criterion(logits, torch.max(Yoh_, 1)[1])
        # loss = cross_entropy_loss(probs, Yoh_, model.parameters(), lambda_reg)
        # gradients = torch.autograd.grad(loss, model.parameters(), create_graph=True)
        # gradients = [grad.clone() for grad in gradients]
        # for param, grad in zip(model.parameters(), gradients):
        #     param.data -= param_delta * grad

        loss.backward()
        optimizer.step()

        # if i % 1000 == 0:
        #     print(f'step: {i}, loss: {loss.item()}')
        if i % 400 == 0:
            print(f'step: {i}, loss: {loss.item()}')

def train_once(model, X, Yoh_, param_delta, lambda_reg):
    gamma = 0.95
    scheduler = ExponentialLR(optimizer, gamma=gamma)
    criterion = torch.nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=param_delta, weight_decay=lambda_reg)
    optimizer = optim.Adam(model.parameters(), lr=param_delta, weight_decay=lambda_reg)
    optimizer.zero_grad()
    logits = model(X)
    loss = criterion(logits, torch.max(Yoh_, 1)[1])

    # probs = softmax(logits)
    # loss = cross_entropy_loss(probs, Yoh_, model.parameters(), lambda_reg)
    loss.backward()
    optimizer.step()
    return loss

    # optimizer = optim.SGD(model.parameters(), lr=param_delta)
    # optimizer.zero_grad()
    # logits = model(X)
    # probs = softmax(logits)
    # loss = cross_entropy_loss(probs, Yoh_, model.parameters(), lambda_reg)
    # loss.backward()
    # optimizer.step()
    # return loss

def eval(model, X):
    with torch.no_grad():
        logits = model(X)
        probs = softmax(logits) # (60000, 10)
    _, predicted = torch.max(probs, 1)
    return predicted.cpu().numpy()

def eval_val(model, X):
    with torch.no_grad():
        logits = model(X)
        probs = softmax(logits) # (60000, 10)
    # _, predicted = torch.max(probs, 1, axis=1)
    return probs

def main(seed_number, distributions, classes, n_o_samples_per_dist, epochs, learning_rate, regularization_lambda, config):

    np.random.seed(seed_number)
    K, C, N = distributions, classes, n_o_samples_per_dist
    
    if K == 0:
        X, Y_ = data_1.sample_gauss_2d(C, N)
    else:
        X, Y_ = data_1.sample_gmm_2d(K, C, N)
    
    Y_oh = data_1.class_to_onehot(Y_.flatten())

    # Define the model configuration
    model = PTDeep(config)

    # Print model parameters
    model.count_params()

    # Convert data to torch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    Y_oh_tensor = torch.tensor(Y_oh, dtype=torch.long)  # Use torch.long for class labels

    # Learn the parameters
    train(model, X_tensor, Y_oh_tensor, epochs, learning_rate, regularization_lambda)

    # Get predictions on training data
    predicted = eval(model, X_tensor)

    # Calculate performance metrics
    accuracy, recall, precision = data_1.eval_perf_multi(predicted, Y_)
    AP = data_1.eval_AP(Y_[np.argsort(predicted.flatten())])

    print("Accuracy: {} Recall: {} Precision: {} AP: {}".format(accuracy, recall, precision, AP))

    # Graph the decision surface
    decfun = lambda X: eval(model, torch.tensor(X, dtype=torch.float32))
    bbox = (np.min(X, axis=0), np.max(X, axis=0))
    data_1.graph_surface(decfun, bbox, offset=0.5)

    # Graph the data points
    data_1.graph_data(X, Y_, predicted, special=[])

    # Show the results
    plt.show()

if __name__ == "__main__":

    """ Tuning parameters """ 
    
    seed_number = 100
    distributions = 6
    classes = 2
    n_o_samples_per_dist = 10
    epochs = 15000
    learning_rate = 0.05
    regularization_lambda = 1e-4
    config = [2, 10, 10, 2]  # [input_dim, hidden_layer_dim1, hidden_layer_dim2, output_dim]


    # former example of pt_logreg
    # seed_number = 100
    # distributions = 0
    # classes = 3
    # n_o_samples_per_dist = 100
    # epochs = int(3e4)
    # learning_rate = 0.1
    # regularization_lambda = 1e-4
    # # config = [2, 10, 10, 2]  # [input_dim, hidden_layer_dim1, hidden_layer_dim2, output_dim]
    # config = [2, 3]  # [input_dim, hidden_layer_dim1, hidden_layer_dim2, output_dim]
    


    main(seed_number, distributions, classes, n_o_samples_per_dist, epochs, 
        learning_rate, regularization_lambda, config)

    """ That's all """
