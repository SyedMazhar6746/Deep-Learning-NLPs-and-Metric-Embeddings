#!/usr/bin/python3

import torch
import torchvision
import matplotlib.pyplot as plt 
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np

import pt_deep_2
import data_1

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

def data_processing(device_name='cpu'):

    dataset_root = '/home/syed_mazhar/c++_ws/src/aa_zagreb_repo/Deep_Learning/Lab_01/data_mnist/mnist'
    mnist_train = torchvision.datasets.MNIST(dataset_root, train=True, download=False)
    mnist_test = torchvision.datasets.MNIST(dataset_root, train=False, download=False)

    x_train, y_train = mnist_train.data, mnist_train.targets
    x_test, y_test = mnist_test.data, mnist_test.targets

    # Normalizing the data
    x_train, x_test = x_train.float().div_(255.0), x_test.float().div_(255.0)

    # Flatten the images
    x_train_flat = x_train.view(x_train.shape[0], -1)
    x_test_flat = x_test.view(x_test.shape[0], -1)

    # Convert data to torch tensors
    X_train_tensor = x_train_flat
    Y_train_oh_tensor = torch.eye(10)[y_train]  # Convert to one-hot encoding
    Y_train_oh_tensor = Y_train_oh_tensor.float()  # Convert to float tensor
    Y_test_oh = torch.eye(10)[y_test].float()


    # Randomly separate 1/5 of the training data into the validation set
    num_samples = len(X_train_tensor)
    indices = np.random.permutation(num_samples)
    split = int(0.8 * num_samples)  # 80% training, 20% validation

    train_indices, val_indices = indices[:split], indices[split:]
    X_train, X_val = X_train_tensor[train_indices], X_train_tensor[val_indices]
    Y_train_oh, Y_val_oh = Y_train_oh_tensor[train_indices], Y_train_oh_tensor[val_indices]

    # train_mean = X_train.mean()
    # X_train, X_val, x_test_flat = (x - train_mean for x in (X_train, X_val, x_test_flat ))
    if device_name=='cuda':
        X_train_tensor = X_train.clone().detach().to(torch.float32).to('cuda')
        Y_train_oh_tensor = Y_train_oh.clone().detach().to(torch.long).to('cuda')
        X_val_tensor = X_val.clone().detach().to(torch.float32).to('cuda')
        Y_val_oh_tensor = Y_val_oh.clone().detach().to(torch.long).to('cuda')
        x_test_tensor = x_test_flat.clone().detach().to(torch.float32).to('cuda')
        Y_test_oh_tensor = Y_test_oh.clone().detach().to(torch.float32).to('cuda')
        return X_train_tensor, Y_train_oh_tensor, X_val_tensor, Y_val_oh_tensor, x_test_tensor, Y_test_oh_tensor
    else:
        # Convert torch tensors to NumPy arrays
        X_train_np, Y_train_oh_np, X_val_np, Y_val_oh_np, x_test_np, Y_test_oh_np = (
            X_train.numpy(), Y_train_oh.numpy(), X_val.numpy(), Y_val_oh.numpy(),
            x_test_flat.numpy(), Y_test_oh.numpy()
        )
        return X_train_np, Y_train_oh_np, X_val_np, Y_val_oh_np, x_test_np, Y_test_oh_np


def evaluate_model(model, X_data, Y_data):
    predictions = pt_deep_2.eval(model, X_data)
    accuracy, precision, recall, confusion_matrix = data_1.eval_perf_multi_mnist(predictions, Y_data.argmax(dim=1))
    return accuracy, precision.mean(), recall.mean(), confusion_matrix

def print_metrics(accuracy, precision, recall, confusion_matrix, mode):
    print(f"{mode} Accuracy: {accuracy:.2f}")
    print(f"{mode} Precision: {precision:.2f}")
    print(f"{mode} Recall: {recall:.2f}")
    print(f"{mode} Confusion Matrix:\n", confusion_matrix)

def plot_weight_matrices(model, show):
    if show:
        ## seeing the images of the classses.
        # Access weight matrices for each digit
        weight_matrices = model.weights
        # print('weight dimensions', weight_matrices) # ParameterList(  (0): Parameter containing: [torch.float32 of size 784x10])

        # # # Access the weight tensor directly
        weight_matrix = weight_matrices[0].cpu()
        # Plot and comment on the weight matrix for each digit separately
        for digit in range(10):
            # Extract the weights for the current digit
            weights_for_digit = weight_matrix[:, digit].detach().numpy().reshape(28, 28)
            
            plt.imshow(weights_for_digit, cmap='gray')
            plt.title(f'Weight Matrix for Digit {digit}')
            plt.show()

def create_batch(X_train_tensor, Y_train_oh_tensor, indices, batch_idx):
    start_idx = batch_idx * batch_size
    end_idx = start_idx + batch_size

    batch_indices = indices[start_idx:end_idx]
    X_batch = X_train_tensor[batch_indices]
    Y_batch = Y_train_oh_tensor[batch_indices]
    return X_batch, Y_batch

def train_once(model, X, Yoh_, optimizer, criterion):
    
    # optimizer = optim.SGD(model.parameters(), lr=param_delta, weight_decay=lambda_reg)
    optimizer.zero_grad()
    logits = model(X)
    loss = criterion(logits, torch.max(Yoh_, 1)[1])
    # probs = softmax(logits)
    # loss = cross_entropy_loss(probs, Yoh_, model.parameters(), lambda_reg)
    loss.backward()
    optimizer.step()
    return loss

def train_linear_ksvm(kernel_type, X_train_np, Y_train_oh_np, X_val_np, Y_val_oh_np, x_test_np, Y_test_oh_np):
    # Initialize and train Linear SVM
    linear_svm = SVC(kernel=kernel_type, decision_function_shape='ovo')
    linear_svm.fit(X_train_np, np.argmax(Y_train_oh_np, axis=1))

    # Predictions on Validation Set
    val_predictions_linear_svm = linear_svm.predict(X_val_np)

    # Evaluate Linear SVM on Validation Set
    accuracy_linear_svm_val = accuracy_score(np.argmax(Y_val_oh_np, axis=1), val_predictions_linear_svm)
    precision_linear_svm_val = precision_score(np.argmax(Y_val_oh_np, axis=1), val_predictions_linear_svm, average='weighted')
    recall_linear_svm_val = recall_score(np.argmax(Y_val_oh_np, axis=1), val_predictions_linear_svm, average='weighted')
    confusion_matrix_linear_svm_val = confusion_matrix(np.argmax(Y_val_oh_np, axis=1), val_predictions_linear_svm)

    # Print metrics for Linear SVM on Validation Set
    print(f"{kernel_type} SVM Metrics on Validation Set:")
    print_metrics(accuracy_linear_svm_val, precision_linear_svm_val, recall_linear_svm_val, confusion_matrix_linear_svm_val, mode='Validation')

    # Predictions on Test Set
    test_predictions_linear_svm = linear_svm.predict(x_test_np)

    # Evaluate Linear SVM on Test Set
    accuracy_linear_svm_test = accuracy_score(np.argmax(Y_test_oh_np, axis=1), test_predictions_linear_svm)
    precision_linear_svm_test = precision_score(np.argmax(Y_test_oh_np, axis=1), test_predictions_linear_svm, average='weighted')
    recall_linear_svm_test = recall_score(np.argmax(Y_test_oh_np, axis=1), test_predictions_linear_svm, average='weighted')
    confusion_matrix_linear_svm_test = confusion_matrix(np.argmax(Y_test_oh_np, axis=1), test_predictions_linear_svm)

    # Print metrics for Linear SVM on Test Set
    print(f"\n{kernel_type} SVM Metrics on Test Set:")
    print_metrics(accuracy_linear_svm_test, precision_linear_svm_test, recall_linear_svm_test, confusion_matrix_linear_svm_test, mode='Test')


def train_deep_model(model, X_train_tensor, Y_train_oh_tensor, X_val_tensor, Y_val_oh_tensor, epochs, 
                    learning_rate, regularization_lambda, print_at_every_x_epochs, batch_size, gamma):
    
    num_samples = len(X_train_tensor)
    num_batches = num_samples // batch_size
    
    best_val_loss = float('inf')
    best_model = None

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=regularization_lambda)
    scheduler = ExponentialLR(optimizer, gamma=gamma, verbose=False)

    for epoch in range(epochs):
        indices = np.random.permutation(num_samples)
        print(indices)
        for batch_idx in range(num_batches):
            X_batch, Y_batch = create_batch(X_train_tensor, Y_train_oh_tensor, indices, batch_idx)

            # Training
            loss = train_once(model, X_batch, Y_batch, optimizer, criterion)
        # loss = pt_deep_2.train_once(model, X_train_tensor, Y_train_oh_tensor, learning_rate, regularization_lambda) # to train the whole model
        if epoch % print_at_every_x_epochs == 0:
            print(f'step: {epoch+1}/{epochs}, Train loss: {loss.item()}')
            
        # Validation after each epoch
        val_predictions = pt_deep_2.eval_val(model, X_val_tensor)
        val_loss = pt_deep_2.cross_entropy_loss(val_predictions, Y_val_oh_tensor, model.parameters(), regularization_lambda)
        scheduler.step()
        # print('model param', (param for param in model.parameters()))
        if epoch % print_at_every_x_epochs == 0:
            print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {val_loss:.4f}")

        # Check if this is the best model so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()
    return best_model


def main(model_name, epochs, learning_rate, regularization_lambda, config, print_at_every_x_epochs, batch_size, gamma):


    if model_name == 'deep':
        X_train_tensor, Y_train_oh_tensor, X_val_tensor, Y_val_oh_tensor, x_test_tensor, Y_test_oh_tensor = data_processing(device_name='cuda')
        model = pt_deep_2.PTDeep(config).cuda()
        
        # Learn the parameters
        # pt_deep.train(model, X_train_tensor, Y_train_oh_tensor, epochs, learning_rate, regularization_lambda)
        # validation
        # The process aims to prevent overfitting by choosing the model that generalizes well to unseen validation data.
        best_model = train_deep_model(model, X_train_tensor, Y_train_oh_tensor, X_val_tensor, Y_val_oh_tensor, epochs, 
                                    learning_rate, regularization_lambda, print_at_every_x_epochs, batch_size, gamma)

        # Load the best model weights
        if best_model:
            model.load_state_dict(best_model)
        
        # Evaluate on Train Set
        train_accuracy, train_precision, train_recall, train_confusion_matrix = evaluate_model(model, X_train_tensor, Y_train_oh_tensor)
        print_metrics(train_accuracy, train_precision, train_recall, train_confusion_matrix, mode='Train')

        # Evaluate on Test Set
        test_accuracy, test_precision, test_recall, test_confusion_matrix = evaluate_model(model, x_test_tensor, Y_test_oh_tensor)
        print_metrics(test_accuracy, test_precision, test_recall, test_confusion_matrix, mode='Test')

        # Plot the weight matrices
        plot_weight_matrices(model, show=False)

    else:
        X_train_np, Y_train_oh_np, X_val_np, Y_val_oh_np, x_test_np, Y_test_oh_np = data_processing()
        train_linear_ksvm(model_name, X_train_np, Y_train_oh_np, X_val_np, Y_val_oh_np, x_test_np, Y_test_oh_np)

if __name__ == "__main__":

    epochs = int(60+1) # 2e3+1
    print_at_every_x_epochs = int(10) # 100 
    learning_rate = 1e-1  #0.05
    regularization_lambda = 0.0005# 0.05 best  #1e-2
    config = [784, 10]  # [input_dim, hidden_layer_dim1, hidden_layer_dim2, output_dim]
    # config = [784, 100, 10]  # [input_dim, hidden_layer_dim1, hidden_layer_dim2, output_dim]
    batch_size = 64
    gamma = 0.9

    model_names = ['deep', 'linear', 'rbf']
    # model_name = model_names[0]
    # model_name = model_names[1]
    model_name = model_names[2]
    main(model_name, epochs, learning_rate, regularization_lambda, config, print_at_every_x_epochs, batch_size, gamma) 