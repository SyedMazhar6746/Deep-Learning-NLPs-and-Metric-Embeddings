#!/usr/bin/python3

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import data_1
import matplotlib.pyplot as plt

class PTLogreg(nn.Module):
    def __init__(self, D, C, param_lambda=0.0):
        super(PTLogreg, self).__init__()
        self.W = nn.Parameter(torch.randn(D, C))
        self.b = nn.Parameter(torch.randn(1, C))
        self.param_lambda = param_lambda

    def forward(self, X):
        scores = torch.mm(X, self.W) + self.b
        return torch.softmax(scores, dim=1)

    def get_loss(self, X, Yoh_):
        scores = torch.mm(X, self.W) + self.b
        cross_entropy_loss = torch.mean(-torch.log(torch.softmax(scores, dim=1)) * Yoh_)
        regularization_loss = self.param_lambda * torch.sum(self.W ** 2)  # L2 regularization
        total_loss = cross_entropy_loss + regularization_loss
        return total_loss

def train(model, X, Yoh_, param_niter, param_delta):
    optimizer = optim.SGD([model.W, model.b], lr=param_delta)
    for i in range(param_niter):
        optimizer.zero_grad()
        loss = model.get_loss(X, Yoh_)
        loss.backward()
        optimizer.step()
        if i % 1000 == 0:
            print(f'step: {i}, loss: {loss.item()}')

def eval(model, X):
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32)
        scores = model(X_tensor)
    return scores.numpy()


def multinomreg_decfun(model):
  def classify(X):
    return np.argmax(eval(model, X), axis=1, keepdims=True)
  return classify

def main(seed_number, distributions, classes, n_o_samples_per_dist, iterations, learning_rate, regularization_lambda):

    # Initialize random number generator
    np.random.seed(seed_number)

    K, C, N = distributions, classes, n_o_samples_per_dist
    X, Y_ = data_1.sample_gauss_2d(C, N)
    
    # Convert Y_ to one-hot encoding
    Y_oh = data_1.class_to_onehot(Y_.flatten()) # (200,2)

    # Define the model
    ptlr = PTLogreg(X.shape[1], Y_oh.shape[1], param_lambda=regularization_lambda)

    # Learn the parameters (X and Y_oh have to be of type torch.Tensor)
    train(ptlr, torch.tensor(X, dtype=torch.float32), torch.tensor(Y_oh, dtype=torch.float32), iterations, learning_rate)

    # Get probabilities on training data
    probs = eval(ptlr, torch.tensor(X, dtype=torch.float32))

    probs = np.argmax(probs, axis=1).reshape(-1, 1)
    Y_ = np.argmax(Y_oh, axis=1).reshape(-1, 1)

    # print('dimensions after', X.shape, Y_.shape, probs.shape)

    accuracy, recall, precision = data_1.eval_perf_multi(probs, Y_)
    AP = data_1.eval_AP(Y_[np.argsort(probs.flatten())])

    print("Accuracy: {} Recall: {} Precision: {} AP: {}".format(accuracy, recall, precision, AP))


    # graph the decision surface
    decfun = multinomreg_decfun(ptlr)
    bbox=(np.min(X, axis=0), np.max(X, axis=0))
    data_1.graph_surface(decfun, bbox, offset=0.5)

    # # graph the data points
    data_1.graph_data(X, Y_, probs, special=[])

    # # # show the results
    plt.show()    

if __name__ == "__main__":

    """ Tuning parameters """ 
    
    seed_number = 100
    distributions = 6
    classes = 3
    n_o_samples_per_dist = 100
    iterations = 100000
    learning_rate = 0.03
    regularization_lambda = 1e-3
    # hidden_layer_dim = 5
    main(seed_number, distributions, classes, n_o_samples_per_dist, iterations, learning_rate, regularization_lambda)

    """ That's all """ 
