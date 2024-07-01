#!/usr/bin/python3

import numpy as np
from sklearn import svm
import data_1
import matplotlib.pyplot as plt

class KSVMWrap:
    def __init__(self, X, Y_, param_svm_c=1, param_svm_gamma='auto'):
        self.svm_classifier = svm.SVC(C=param_svm_c, gamma=param_svm_gamma)
        self.svm_classifier.fit(X, Y_.flatten())

    def predict(self, X):
        return self.svm_classifier.predict(X)

    def get_scores(self, X):
        return self.svm_classifier.decision_function(X)

    @property
    def support(self):
        return self.svm_classifier.support_

def main():
    seed_number = 100
    distributions = 6
    classes = 2
    n_o_samples_per_dist = 100 # 100

    np.random.seed(seed_number)
    K, C, N = distributions, classes, n_o_samples_per_dist
    X, Y_ = data_1.sample_gmm_2d(K, C, N)
    Y_oh = data_1.class_to_onehot(Y_.flatten())

    # Train KSVMWrap model
    svm_model = KSVMWrap(X, Y_)

    predicted_svm = svm_model.predict(X)
    accuracy_svm, recall_svm, precision_svm = data_1.eval_perf_multi(predicted_svm, Y_)
    AP_svm = data_1.eval_AP(Y_[np.argsort(predicted_svm.flatten())])
    print("KSVMWrap Model - Accuracy: {}, Recall: {}, Precision: {}, AP: {}".format(accuracy_svm, recall_svm, precision_svm, AP_svm))

    # Graph the decision surface and classification results for RBF SVM
    decfun_svm = lambda X: svm_model.predict(X)
    bbox_svm = (np.min(X, axis=0), np.max(X, axis=0))
    data_1.graph_surface(decfun_svm, bbox_svm, offset=0.5)
    data_1.graph_data(X, Y_, predicted_svm, special=svm_model.support)

    # Show the results
    plt.show()

if __name__ == "__main__":
    main()
