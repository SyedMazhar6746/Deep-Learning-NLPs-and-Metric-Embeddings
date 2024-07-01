#!/usr/bin/python3

import torch
import torchvision
import matplotlib.pyplot as plt 
import numpy as np

import pt_deep_2
import data_1

def data_processing():

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

    return X_train, Y_train_oh, X_val, Y_val_oh, x_test_flat, Y_test_oh

def main(epochs, learning_rate, regularization_lambda, config, print_at_every_x_epochs):

    X_train_tensor, Y_train_oh_tensor, X_val_tensor, Y_val_oh_tensor, x_test_tensor, Y_test_oh_tensor = data_processing()

    model = pt_deep_2.PTDeep(config).cuda()
    
    X_train_tensor = X_train_tensor.clone().detach().to(torch.float32).to('cuda')
    Y_train_oh_tensor = Y_train_oh_tensor.clone().detach().to(torch.long).to('cuda')
    X_val_tensor = X_val_tensor.clone().detach().to(torch.float32).to('cuda')
    Y_val_oh_tensor = Y_val_oh_tensor.clone().detach().to(torch.long).to('cuda')
    x_test_tensor = x_test_tensor.clone().detach().to(torch.float32).to('cuda')
    Y_test_oh_tensor = Y_test_oh_tensor.clone().detach().to(torch.float32).to('cuda')


    # Learn the parameters
    # pt_deep.train(model, X_train_tensor, Y_train_oh_tensor, epochs, learning_rate, regularization_lambda)
    
    # validation
    # The process aims to prevent overfitting by choosing the model that generalizes 
    # well to unseen validation data.
    best_val_loss = float('inf')
    best_model = None

    for epoch in range(epochs):
        # Training
        loss = pt_deep_2.train_once(model, X_train_tensor, Y_train_oh_tensor, learning_rate, regularization_lambda)
        if epoch % print_at_every_x_epochs == 0:
            print(f'step: {epoch}, loss: {loss.item()}')
        # Validation
        val_predictions = pt_deep_2.eval_val(model, X_val_tensor)
        val_loss = pt_deep_2.cross_entropy_loss(val_predictions, Y_val_oh_tensor, model.parameters(), regularization_lambda)
        # print('model param', (param for param in model.parameters()))
        if epoch % print_at_every_x_epochs == 0:
            print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {val_loss:.4f}")

        # Check if this is the best model so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()

    # Load the best model weights
    if best_model:
        model.load_state_dict(best_model)
    
    # x-train evaluation
    train_predictions = pt_deep_2.eval(model, X_train_tensor)
    train_accuracy, train_precision, train_recall, confusion_matrix = data_1.eval_perf_multi_mnist(train_predictions, Y_train_oh_tensor.argmax(dim=1))

    print(f"Train Accuracy: {train_accuracy:.2f}")
    print(f"Train Precision: {train_precision.mean():.2f}")
    print(f"Train Recall: {train_recall.mean():.2f}")

    print("Train Confusion Matrix:\n", confusion_matrix)

    # Evaluate the model on the test set
    test_predictions = pt_deep_2.eval(model, x_test_tensor)
    test_accuracy, test_precision, test_recall, confusion_matrix = data_1.eval_perf_multi_mnist(test_predictions, Y_test_oh_tensor.argmax(dim=1))

    print(f"test Accuracy: {test_accuracy:.2f}")
    print(f"test Precision: {test_precision.mean():.2f}")
    print(f"test Recall: {test_recall.mean():.2f}")

    print("Test Confusion Matrix:\n", confusion_matrix)

    # # # ## seeing the images of the classses.
    # # # # Access weight matrices for each digit
    # # # weight_matrices = model.weights
    # # # # print('weight dimensions', weight_matrices) # ParameterList(  (0): Parameter containing: [torch.float32 of size 784x10])

    # # # # # # Access the weight tensor directly
    # # # weight_matrix = weight_matrices[0].cpu()
    # # # # Plot and comment on the weight matrix for each digit separately
    # # # for digit in range(10):
    # # #     # Extract the weights for the current digit
    # # #     weights_for_digit = weight_matrix[:, digit].detach().numpy().reshape(28, 28)
        
    # # #     plt.imshow(weights_for_digit, cmap='gray')
    # # #     plt.title(f'Weight Matrix for Digit {digit}')
    # # #     plt.show()

if __name__ == "__main__":

    epochs = int(2000) # 2e3+1
    print_at_every_x_epochs = int(100) # 100 
    learning_rate = 0.05
    regularization_lambda = 0.05#1e-2
    config = [784, 10]  # [input_dim, hidden_layer_dim1, hidden_layer_dim2, output_dim]

    main(epochs, learning_rate, regularization_lambda, config, print_at_every_x_epochs) 