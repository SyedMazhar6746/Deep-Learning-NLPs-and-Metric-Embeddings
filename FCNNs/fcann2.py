#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt

import pdb
import IPython

import data_1

import numpy as np

def stable_softmax(scores):
    shifted_scores = scores - np.max(scores, axis=1, keepdims=True)
    exp_shifted_scores = np.exp(shifted_scores)
    return exp_shifted_scores / np.sum(exp_shifted_scores, axis=1, keepdims=True)


def fcann2_train(X, Y_, param_niter=1000, param_delta=0.01, param_lambda=1e-3, hidden_dim=5):
    N, D = X.shape
    C = Y_.shape[1]
    W1 = np.random.randn(D, hidden_dim)
    b1 = np.zeros((1, hidden_dim))
    W2 = np.random.randn(hidden_dim, C)
    b2 = np.zeros((1, C))
    
    for i in range(param_niter):
        # Forward pass
        hidden_layer = np.maximum(0, np.dot(X, W1) + b1)  # ReLU activation
        scores = np.dot(hidden_layer, W2) + b2

        # Softmax activation for output layer
        probs = stable_softmax(scores)  

        # Compute loss with L2 regularization
        loss = -np.mean(np.sum(np.log(probs) * Y_, axis=1)) + 0.5 * param_lambda * (np.sum(W1**2) + np.sum(W2**2))
        
        if i % 1000 == 0:
            print("iteration {}: loss {}".format(i, loss))
        
        # Backpropagation
        dL_ds = probs - Y_
        grad_W2 = np.dot(hidden_layer.T, dL_ds) / N + param_lambda * W2
        grad_b2 = np.sum(dL_ds, axis=0, keepdims=True) / N
        grad_hidden = np.dot(dL_ds, W2.T)
        grad_hidden[hidden_layer <= 0] = 0  # ReLU derivative
        grad_W1 = np.dot(X.T, grad_hidden) / N + param_lambda * W1
        grad_b1 = np.sum(grad_hidden, axis=0, keepdims=True) / N
        
        # Update weights and biases
        W1 -= param_delta * grad_W1
        b1 -= param_delta * grad_b1
        W2 -= param_delta * grad_W2
        b2 -= param_delta * grad_b2
    
    return W1, b1, W2, b2

def fcann2_classify(X, W1, b1, W2, b2):
    hidden_layer = np.maximum(0, np.dot(X, W1) + b1)
    scores = np.dot(hidden_layer, W2) + b2
    probs = stable_softmax(scores)
    return probs

def multinomreg_decfun(W1, b1, W2, b2):
  def classify(X):
    return np.argmax(fcann2_classify(X, W1, b1, W2, b2), axis=1, keepdims=True)
  return classify

if __name__ == "__main__":

    """ Tuning parameters """ 
    
    seed_number = 100
    distributions = 6
    classes = 2
    n_o_samples_per_dist = 10
    iterations = 100000
    learning_rate = 0.05
    regularization_lambda = 1e-3
    hidden_layer_dim = 5

    """ That's all """ 

    np.random.seed(seed_number)

    # get the training dataset
    K, C, N = distributions, classes, n_o_samples_per_dist
    X, Y_ = data_1.sample_gmm_2d(K, C, N)
    
    # Convert Y_ to one-hot encoding
    Y_oh = data_1.class_to_onehot(Y_.flatten())

    # train the model
    W1, b1, W2, b2 = fcann2_train(X, Y_oh, param_niter=iterations, param_delta=learning_rate, param_lambda=regularization_lambda, hidden_dim=hidden_layer_dim)

    # evaluate the model on the training dataset
    probs = fcann2_classify(X, W1, b1, W2, b2)
    Y = np.argmax(probs, axis=1).reshape(-1, 1)
    Y_ = np.argmax(Y_oh, axis=1).reshape(-1, 1)

    # report performance
    accuracy, recall, precision = data_1.eval_perf_multi(Y, Y_)
    AP = data_1.eval_AP(Y_[np.argsort(Y.flatten())])
    print("Accuracy: {} Recall: {} Precision: {} AP: {}".format(accuracy, recall, precision, AP))

    # graph the decision surface
    decfun = multinomreg_decfun(W1, b1, W2, b2)
    bbox = (np.min(X, axis=0), np.max(X, axis=0))
    data_1.graph_surface(decfun, bbox, offset=0.5)

    # graph the data points
    data_1.graph_data(X, Y_, Y, special=[])

    # show the results
    plt.show()
    