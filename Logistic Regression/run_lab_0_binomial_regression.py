#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt

import pdb
import IPython

from Module_1 import data

def sigmoid(scores):
    return 1 / (1 + np.exp(-scores))


def binlogreg_train(X, Y_, param_niter=1000, param_delta=0.01):
    N, D = X.shape
    w = np.random.randn(D).reshape(-1, 1)  ## (Dx1) # Initialize weights using normal distribution N(0,1)
    b = 0  # Initialize bias

    
    for i in range(param_niter):
        # Classification scores
        scores = np.dot(X, w) + b # ((NxD)x(Dx1)) 
        
        # A posteriori class probabilities
        probs = sigmoid(scores)  # (200, 1)  # y_prediction
        
        # Loss
        loss = -np.mean(Y_ * np.log(probs) + (1 - Y_) * np.log(1 - probs))
        # Trace 
        if i % 100 == 0:
            print("iteration {}: loss {}".format(i, loss))

        
        # Derivative of the loss function with respect to classification scores
        dL_dscores = probs - Y_  # (200, 1) 
        
        # Gradients with respect to parameters
        grad_w = (1 / N) * np.dot(X.T, dL_dscores)
        grad_b = (1 / N) * np.sum(dL_dscores)
        
        # Modifying the parameters
        w -= param_delta * grad_w # .reshape(-1, 1)
        b -= param_delta * grad_b

        probs = binlogreg_classify(X, w,b)
        Y = (probs >= 0.5).astype(int)

    return w, b

def binlogreg_classify(X, w, b):

    # Calculate classification scores
    scores = np.dot(X, w) + b
    
    # Calculate posterior probabilities using sigmoid function
    probs = sigmoid(scores)
    
    return probs


def binlogreg_decfun(w,b):
  def classify(X):
    return binlogreg_classify(X, w,b)
  return classify

if __name__=="__main__":
    np.random.seed(100)

    # get the training dataset
    X,Y_ = data.sample_gauss_2d(2, 100)

    # train the model
    w,b = binlogreg_train(X, Y_,  param_niter=1000, param_delta=0.01)

    # evaluate the model on the training dataset
    probs = binlogreg_classify(X, w,b)
    
    Y = (probs >= 0.5).astype(int)

    # report performance
    accuracy, recall, precision = data.eval_perf_binary(Y, Y_)
    
    AP = data.eval_AP(Y_[np.argsort(probs.flatten())])
    print (accuracy, recall, precision, AP)

    # graph the decision surface
    decfun = binlogreg_decfun(w,b)
    bbox=(np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(decfun, bbox, offset=0.5)

    # graph the data points
    data.graph_data(X, Y_, Y, special=[])

    # show the results
    plt.show()    
