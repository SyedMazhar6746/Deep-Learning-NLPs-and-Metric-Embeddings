#!/usr/bin/python3

import torch
import torch.nn as nn
import torch.optim as optim
import data_1
import matplotlib.pyplot as plt
import numpy as np

class BatchNormLayer(torch.nn.Module):
    def __init__(self, input_size, momentum=0.9, epsilon=1e-5):
        super(BatchNormLayer, self).__init__()
        self.input_size = input_size
        self.momentum = momentum
        self.epsilon = epsilon

        # Initialize parameters (Learnable parameters)
        self.gamma = torch.nn.Parameter(torch.ones(1, input_size))  # Scaling parameter
        self.beta = torch.nn.Parameter(torch.zeros(1, input_size))  # Shift parameter
        self.running_mean = torch.zeros(1, input_size)
        self.running_var = torch.ones(1, input_size)

        # During training
        self.batch_mean = None
        self.batch_var = None

    def forward(self, x, training):
        if training:
            # Calculate batch mean and variance
            self.batch_mean = torch.mean(x, dim=0, keepdim=True)
            self.batch_var = torch.var(x, dim=0, keepdim=True)

            # Update running mean and variance
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.batch_var
        else:
            # During inference, use running statistics
            self.batch_mean = self.running_mean
            self.batch_var = self.running_var

        # Normalize
        x_normalized = (x - self.batch_mean) / torch.sqrt(self.batch_var + self.epsilon)

        # Scale and shift
        out = self.gamma * x_normalized + self.beta
        return out

class PTDeepWithBatchNorm(nn.Module):
    def __init__(self, config, activation_func=nn.ReLU()):
        super(PTDeepWithBatchNorm, self).__init__()
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        self.activations = []
        self.batch_norm_layers = [BatchNormLayer(config[i + 1]) for i in range(len(config) - 2)]

        for i in range(len(config) - 1):
            weight = nn.Parameter(torch.randn(config[i], config[i + 1]))
            bias = nn.Parameter(torch.randn(1, config[i + 1]))
            self.weights.append(weight)
            self.biases.append(bias)
            if i < len(config) - 2 or True:
                self.activations.append(activation_func)

    def forward(self, X, training=False):
        for i in range(len(self.weights)):
            X = torch.matmul(X, self.weights[i]) + self.biases[i]

            # Apply batch normalization after affine transformation
            if i < len(self.weights) - 1:
                X = self.batch_norm_layers[i].forward(X, training=training)

            # Non-linearity activation
            if i < len(self.weights) - 1:
                X = self.activations[i](X)
        return X

    def count_params(self):
        total_params = 0
        for name, param in self.named_parameters():
            print(f'Parameter: {name}, Size: {param.size()}')
            total_params += param.numel()
        print(f'Total Parameters: {total_params}')

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

    def forward(self, X, training=True): # training variablr is not used anywhere but to follow the convention
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
    return exp_logits / torch.sum(exp_logits, dim=1, keepdim=True)

def cross_entropy_loss(y_pred, y_true, weights, lambda_reg):
    cross_entropy = -torch.mean(torch.log(torch.sum(y_true * (y_pred + 1e-10), axis=1)))
    regularization = (0.5 * lambda_reg * sum(torch.sum(param ** 2) for param in weights)) 
    return cross_entropy + regularization


def train(model, X, Yoh_, param_niter, param_delta, lambda_reg):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=param_delta, weight_decay=lambda_reg)

    for i in range(param_niter):
        optimizer.zero_grad()
        logits = model(X, training=True)
        loss = criterion(logits, torch.max(Yoh_, 1)[1])
        loss.backward()
        optimizer.step()
        if i % 400 == 0:
            print(f'step: {i}, loss: {loss.item()}')

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
    return probs

def main(batch_normalization, seed_number, distributions, classes, n_o_samples_per_dist, epochs, learning_rate, regularization_lambda, config):

    np.random.seed(seed_number)
    K, C, N = distributions, classes, n_o_samples_per_dist
    
    if K == 0:
        X, Y_ = data_1.sample_gauss_2d(C, N)
    else:
        X, Y_ = data_1.sample_gmm_2d(K, C, N)

    Y_oh = data_1.class_to_onehot(Y_.flatten())

    # Define the model configuration
    if batch_normalization:
        model = PTDeepWithBatchNorm(config)
    else:
        model = PTDeep(config)

    # Print model parameters
    # model.count_params()

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
    epochs = 5000 # 15000 best
    learning_rate = 1e-3 # 0.05
    regularization_lambda = 1e-4
    config = [2, 10, 10, 2]  # [input_dim, hidden_layer_dim1, hidden_layer_dim2, output_dim]
    batch_normalization = True

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
    


    main(batch_normalization, seed_number, distributions, classes, n_o_samples_per_dist, epochs, 
        learning_rate, regularization_lambda, config)

    """ That's all """
