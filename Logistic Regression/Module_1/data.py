#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import random

import pdb
import IPython

# Random2DGaussian: sample random 2D data by using Gaussian random distribution
#     '''
#         Arguments
#         minx, maxx, miny, maxy: min and max values of the data points

#         Return values
#         samples: sampled data points of size Nx2.
#     '''

class Random2DGaussian:
    def __init__(self, minx=0, maxx=10, miny=0, maxy=10):
        self.minx = minx
        self.maxx = maxx
        self.miny = miny
        self.maxy = maxy
        # np.random.seed(100)  # Set seed for reproducibility
        self.mean = np.array([np.random.uniform(minx, maxx), np.random.uniform(miny, maxy)])
        eigvalx = (np.random.random_sample() * (maxx - minx) / 5) ** 2
        eigvaly = (np.random.random_sample() * (maxy - miny) / 5) ** 2
        angle = np.random.uniform(0, 2 * np.pi)
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                   [np.sin(angle), np.cos(angle)]])
        self.cov_matrix = np.dot(np.dot(rotation_matrix.T, np.diag([eigvalx, eigvaly])), rotation_matrix)

    def get_sample(self, nsamples):

        samples = np.random.multivariate_normal(self.mean, self.cov_matrix, nsamples) # to get the sample
        return samples

def graph_surface(function, rect, offset=0.5, width=256, height=256):
  """Creates a surface plot (visualize with plt.show)

  Arguments:
    function: surface to be plotted
    rect:     function domain provided as:
              ([x_min,y_min], [x_max,y_max])
    offset:   the level plotted as a contour plot

  Returns:
    None
  """

  lsw = np.linspace(rect[0][1], rect[1][1], width) 
  lsh = np.linspace(rect[0][0], rect[1][0], height)
  xx0,xx1 = np.meshgrid(lsh, lsw)
  grid = np.stack((xx0.flatten(),xx1.flatten()), axis=1)

  #get the values and reshape them
  values=function(grid)
  values=values.reshape((width,height))
  
  # fix the range and offset
  delta = offset if offset else 0
  maxval=max(np.max(values)-delta, - (np.min(values)-delta))
  
  # draw the surface and the offset
  plt.pcolormesh(xx0, xx1, values, 
     vmin=delta-maxval, vmax=delta+maxval)
    
  if offset != None:
    plt.contour(xx0, xx1, values, colors='black', levels=[offset])


def graph_data(X,Y_, Y, special=[]):
  """Creates a scatter plot (visualize with plt.show)

  Arguments:
      X:       datapoints
      Y_:      groundtruth classification indices
      Y:       predicted class indices
      special: use this to emphasize some points

  Returns:
      None
  """
  Y_ = Y_.flatten()
  Y = Y.flatten()
  
  # colors of the datapoint markers
  palette=([0.5,0.5,0.5], [1,1,1], [0.2,0.2,0.2])
  colors = np.tile([0.0,0.0,0.0], (Y_.shape[0],1))
  for i in range(len(palette)):
    colors[Y_==i] = palette[i]

  # sizes of the datapoint markers
  sizes = np.repeat(20, len(Y_))
  sizes[special] = 40
  
  # draw the correctly classified datapoints
  good = (Y_==Y)
  plt.scatter(X[good,0],X[good,1], c=colors[good], 
              s=sizes[good], marker='o', edgecolors='black')

  # draw the incorrectly classified datapoints
  bad = (Y_!=Y)
  plt.scatter(X[bad,0],X[bad,1], c=colors[bad], 
              s=sizes[bad], marker='s', edgecolors='black')

def class_to_onehot(Y):
  Yoh=np.zeros((len(Y),max(Y)+1))
  Yoh[range(len(Y)),Y] = 1
  return Yoh


def eval_perf_binary(Y, Y_):
  
  tp = sum(np.logical_and(Y==Y_, Y_==True))
  fn = sum(np.logical_and(Y!=Y_, Y_==True))
  tn = sum(np.logical_and(Y==Y_, Y_==False))
  fp = sum(np.logical_and(Y!=Y_, Y_==False))

  recall = tp / (tp + fn)
  precision = tp / (tp + fp)
  accuracy = (tp + tn) / (tp+fn + tn+fp)

  return accuracy, recall, precision

""" 
Takes the indices of true and predicted classes. (two 1D arrays
Return metrics for each class
"""

def eval_perf_multi(Y, Y_):
    num_classes = np.max(Y_) + 1
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

    for true_class, predicted_class in zip(Y_.flatten(), Y.flatten()):
        confusion_matrix[true_class, predicted_class] += 1

    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    total_samples = len(Y_)

    for i in range(num_classes):
        true_positive = confusion_matrix[i, i]
        predicted_positive = np.sum(confusion_matrix[:, i])
        actual_positive = np.sum(confusion_matrix[i, :])

        precision[i] = true_positive / predicted_positive if predicted_positive != 0 else 0
        recall[i] = true_positive / actual_positive if actual_positive != 0 else 0

    accuracy = np.sum(np.diag(confusion_matrix)) / total_samples

    return accuracy, precision, recall


def eval_AP(ranked_labels):
  """Recovers AP from ranked labels"""
  
  n = len(ranked_labels)
  pos = sum(ranked_labels)
  pos1 = sum(ranked_labels)
  neg = n - pos
  
  tp = pos
  tn = 0
  fn = 0
  fp = neg
  
  sumprec=0
  #IPython.embed()
  for x in ranked_labels:
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)    

    if x:
      sumprec += precision
    #IPython.embed()

    tp -= x
    fn += x
    fp -= not x
    tn += not x
  return sumprec/pos1

    
def sample_gauss_2d(nclasses, nsamples): # (2, 200)

    Gs = []
    Ys=[]
    for i in range(nclasses):
        Gs.append(Random2DGaussian())
        Ys.append(i)

    X = np.vstack([G.get_sample(nsamples) for G in Gs])    
    Y_= np.hstack([[Y]*nsamples for Y in Ys])
    
    return X,Y_.reshape([-1, 1]) # x = (200, 2) = [[x1, y1], [x2, y2]]  , Y = (200, 1) 
    