#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt

import pdb
import IPython

from Module_1 import data

def softmax(scores):
    expscores = np.exp(scores)
    return expscores / (np.sum(expscores, axis=1, keepdims=True))


def logreg_train(X, Y_, param_niter=1000, param_delta=0.01):
    N, D = X.shape
    C = Y_.shape[1]
    W = np.random.randn(int(D), int(C))
    b = np.zeros((1, int(C)))
    
    for i in range(param_niter):
        scores = np.dot(X, W) + b

        probs = softmax(scores)  # NxC
        logprobs = np.log(probs)  # NxC

        loss = -np.mean(np.sum(logprobs * Y_, axis=1)) # scalar
        
        if i % 1000 == 0:
            print("iteration {}: loss {}".format(i, loss))
        
        dL_ds = probs - Y_ # NxC
        grad_W = np.dot(X.T, dL_ds) / N  # DxC
        grad_b = np.sum(dL_ds, axis=0, keepdims=True) / N  # 1xC
        
        W -= param_delta * grad_W
        b -= param_delta * grad_b
    
    return W, b

def logreg_classify(X, W, b):
    scores = np.dot(X, W) + b
    probs = softmax(scores)
    return probs

def class_to_onehot(Y):
  Yoh=np.zeros((len(Y),max(Y)+1))
  Yoh[range(len(Y)),Y] = 1
  return Yoh

def multinomreg_decfun(w,b):
  def classify(X):
    return np.argmax(logreg_classify(X, w,b), axis=1, keepdims=True)
  return classify

if __name__ == "__main__":
    np.random.seed(100)

    # get the training dataset
    X, Y_ = data.sample_gauss_2d(3, 100)  # Change the number of classes here

    # Convert Y_ to one-hot encoding
    Y_oh = class_to_onehot(Y_.flatten())

    # train the model
    W, b = logreg_train(X, Y_oh, param_niter=20000, param_delta=0.1)

    # evaluate the model on the training dataset
    probs = logreg_classify(X, W, b)

    # Extracting indices from true and predicted classes
    
    Y = np.argmax(probs, axis=1).reshape(-1, 1)
    Y_ = np.argmax(Y_oh, axis=1).reshape(-1, 1)

    # # report performance
    accuracy, recall, precision = data.eval_perf_multi(Y, Y_)

    AP = data.eval_AP(Y_[np.argsort(Y.flatten())])
    print("Accuracy: {} Recall: {} Precision: {} AP: {}".format(accuracy, recall, precision, AP))


    # graph the decision surface
    decfun = multinomreg_decfun(W,b)
    bbox=(np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(decfun, bbox, offset=0.5)

    # # graph the data points
    data.graph_data(X, Y_, Y, special=[])

    # show the results
    plt.show()    