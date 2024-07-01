#!/usr/bin/python3
import torch
from torch import nn
from pathlib import Path
import os
from torch.utils.tensorboard import SummaryWriter
import skimage as ski
import math
from collections import Counter

import torch
from torch import nn
from torch.utils.data import Subset, DataLoader, TensorDataset
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

import os
import pickle
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score

import pdb
import skimage as ski
import skimage.io



# Define the CNN architecture for CIFAR-10
class ConvolutionalModel(nn.Module):
    def __init__(self, in_channels, conv1_width, conv2_width, fc1_width, fc2_width, class_count):
        super(ConvolutionalModel, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, conv1_width, kernel_size=5, stride=1, padding=2, bias=True)
        self.conv2 = nn.Conv2d(conv1_width, conv2_width, kernel_size=5, stride=1, padding=2, bias=True)
        
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)

        # Fully connected layers
        self.flatten = nn.Flatten()
        self.fc1_input_dim = conv2_width * 7 * 7  # Assuming output from conv2 is 32*8x8 for CIFAR-10
        self.fc1 = nn.Linear(self.fc1_input_dim, fc1_width, bias=True)
        self.fc2 = nn.Linear(fc1_width, fc2_width, bias=True)
        self.fc_logits = nn.Linear(fc2_width, class_count, bias=True)

        # Weight initialization
        self.reset_parameters()

    def reset_parameters(self):
      for m in self.modules():
        if isinstance(m, nn.Conv2d):
          nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
          nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear) and m is not self.fc_logits:
          nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
          nn.init.constant_(m.bias, 0)
      self.fc_logits.reset_parameters()

    def forward(self, x):
        
        h = self.conv1(x)
        h = torch.relu(h)
        h = self.pool(h)

        h = self.conv2(h)
        h = torch.relu(h)
        h = self.pool(h)

        # h = h.view(h.size(0), -1)  # Flatten the output before FC layers
        h = self.flatten(h)

        h = self.fc1(h)

        h = torch.relu(h)

        h = self.fc2(h)
        h = torch.relu(h)

        logits = self.fc_logits(h)
        return logits

class Helper():
    def __init__(self):
        super(Helper, self).__init__()

    def shuffle_data(self, data_x, data_y):
      indices = np.arange(data_x.shape[0])
      np.random.shuffle(indices)
      shuffled_data_x = np.ascontiguousarray(data_x[indices])
      shuffled_data_y = np.ascontiguousarray(data_y[indices])
      return shuffled_data_x, shuffled_data_y

    def unpickle(self, file):
      fo = open(file, 'rb')
      dict = pickle.load(fo, encoding='latin1')
      fo.close()
      return dict

    # Function to evaluate model performance
    def evaluate(self, data_loader, model, device, criterion):
        model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, labels in data_loader:
                # labels = labels.long()
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())

            # Calculate overall accuracy
            accuracy = sum(np.array(all_preds) == np.array(all_targets)) / len(all_preds)

            # Calculate confusion matrix
            confusion_mat = confusion_matrix(all_targets, all_preds)

            # Calculate precision and recall for each class
            precision = precision_score(all_targets, all_preds, average=None)
            recall = recall_score(all_targets, all_preds, average=None)

            return accuracy, confusion_mat, precision, recall

class Save_data():
    def __init__(self):
        super(Save_data, self).__init__()
   
    # Define loss and optimizer
    def save_filter(self, epoch, conv1_weights, count, avg_loss):
        writer.add_scalar('Loss/train', avg_loss, global_step=count)

        # Normalize the weights between 0 and 1 for visualization
        conv1_weights_normalized = (conv1_weights - conv1_weights.min()) / (conv1_weights.max() - conv1_weights.min())

        # Prepare an empty image with dimensions suitable for visualizing the filters
        empty_image = torch.zeros(3, 11, 47)

        # Place the filter weights into the empty image grid
        for i in range(conv1_weights_normalized.shape[0]):
            row = (i // 8) 
            col = (i % 8) 

            for ch in range(3):
                empty_image[ch][row * 6: row * 6 + 5, col * 6: col * 6 + 5] = conv1_weights_normalized[i][ch]

            # empty_image[0][row:row + 5, col:col + 5] = conv1_weights_normalized[i]
        # pdb.set_trace()

        # Add the image to TensorBoard
        writer.add_image('epoch_' + str(epoch) + '_Conv1_filters', empty_image, global_step=count)
        count += 1
        return count

    def draw_conv_filters(self, epoch, step, weights, save_dir):
        w = weights.cpu().numpy()  # Move tensor from GPU to CPU and convert to NumPy array
        w -= w.min()  # Normalize the values to ensure they are within the valid range [0, 1]
        w /= w.max()
        w *= 255  # Scale the values to [0, 255] for image representation
        w = w.astype(np.uint8)  # Convert to unsigned integer data type for image representation

        num_filters = w.shape[0]
        num_channels = w.shape[1]
        k = w.shape[2]
        assert w.shape[3] == w.shape[2]
        w = w.transpose(2, 3, 1, 0)
        
        border = 1
        cols = 8
        rows = math.ceil(num_filters / cols)
        width = cols * k + (cols - 1) * border
        height = rows * k + (rows - 1) * border
        img = np.zeros([height, width, num_channels], dtype=np.uint8)  # Create an empty image array
        
        for i in range(num_filters):
            r = int(i / cols) * (k + border)
            c = int(i % cols) * (k + border)
            img[r:r + k, c:c + k, :] = w[:, :, :, i]  # Fill the image with filter weights
        
        filename = f'epoch_{epoch:02d}_step_{step:06d}.png'
        ski.io.imsave(os.path.join(save_dir, filename), img)

    def draw_image(self, img, mean, std):
      img = img.transpose(1, 2, 0)
      img *= std
      img += mean
      img = img.astype(np.uint8)
      ski.io.imshow(img)
      ski.io.show()

    def plot_training_progress(self, save_dir, data):
      fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16,8))

      linewidth = 2
      legend_size = 10
      train_color = 'm'
      val_color = 'c'

      num_points = len(data['train_loss'])
      x_data = np.linspace(1, num_points, num_points)
      ax1.set_title('Cross-entropy loss')
      ax1.plot(x_data, data['train_loss'], marker='o', color=train_color,
              linewidth=linewidth, linestyle='-', label='train')
      ax1.plot(x_data, data['valid_loss'], marker='o', color=val_color,
              linewidth=linewidth, linestyle='-', label='validation')
      ax1.legend(loc='upper right', fontsize=legend_size)
      ax2.set_title('Average class accuracy')
      ax2.plot(x_data, data['train_acc'], marker='o', color=train_color,
              linewidth=linewidth, linestyle='-', label='train')
      ax2.plot(x_data, data['valid_acc'], marker='o', color=val_color,
              linewidth=linewidth, linestyle='-', label='validation')
      ax2.legend(loc='upper left', fontsize=legend_size)
      ax3.set_title('Learning rate')
      ax3.plot(x_data, data['lr'], marker='o', color=train_color,
              linewidth=linewidth, linestyle='-', label='learning_rate')
      ax3.legend(loc='upper left', fontsize=legend_size)

      save_path = os.path.join(save_dir, 'training_plot.png')
      print('Plotting in: ', save_path)
      plt.savefig(save_path)

def process_data(batch_size, val_split_at, img_height, img_width, num_channels):
  
  train_x = np.ndarray((0, img_height * img_width * num_channels), dtype=np.float32) # empty array
  train_y = []
  for i in range(1, 6):
    subset = helper.unpickle(os.path.join(DATA_DIR, 'data_batch_%d' % i))
    train_x = np.vstack((train_x, subset['data']))
    train_y += subset['labels']
  train_x = train_x.reshape((-1, num_channels, img_height, img_width)).transpose(0, 2, 3, 1)
  train_y = np.array(train_y, dtype=np.int32)

  subset = helper.unpickle(os.path.join(DATA_DIR, 'test_batch'))
  test_x = subset['data'].reshape((-1, num_channels, img_height, img_width)).transpose(0, 2, 3, 1).astype(np.float32)
  test_y = np.array(subset['labels'], dtype=np.int32)

  valid_size = val_split_at
  train_x, train_y = helper.shuffle_data(train_x, train_y)
  valid_x = train_x[:valid_size, ...]
  valid_y = train_y[:valid_size, ...]
  train_x = train_x[valid_size:, ...]
  train_y = train_y[valid_size:, ...]
  data_mean = train_x.mean((0, 1, 2))
  data_std = train_x.std((0, 1, 2))

  train_x = (train_x - data_mean) / data_std
  valid_x = (valid_x - data_mean) / data_std
  test_x = (test_x - data_mean) / data_std

  train_x = train_x.transpose(0, 3, 1, 2)
  valid_x = valid_x.transpose(0, 3, 1, 2)
  test_x = test_x.transpose(0, 3, 1, 2)

  train_x = torch.from_numpy(train_x)
  valid_x = torch.from_numpy(valid_x)
  test_x = torch.from_numpy(test_x)
  train_y = (torch.from_numpy(train_y)).long()
  valid_y = (torch.from_numpy(valid_y)).long()
  test_y = (torch.from_numpy(test_y)).long()

  batch_size = batch_size
  train_data = TensorDataset(train_x, train_y)
  val_data = TensorDataset(valid_x, valid_y)
  test_data = TensorDataset(test_x, test_y)

  train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False) # each image has a shape torch.Size([3, 32, 32])
  val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
  test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

  return train_loader, val_loader, test_loader, data_mean, data_std



"""===================== Initialization ======================================="""

helper = Helper()
save_data = Save_data()

DATA_DIR = 'cifar-10-python/cifar-10-batches-py'

SAVE_DIR = Path(__file__).parent / 'pytorch_cifar_out'
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# Fixed
img_height = 32
img_width = 32
num_channels = 3
num_classes = 10

# changeable
batch_size = 50
val_split_at = 5000
epoch = 8
weight_decay = 1e-3

config = {}
config['max_epochs'] = epoch
config['batch_size'] = batch_size
config['save_dir'] = SAVE_DIR
config['weight_decay'] = weight_decay
config['lr_policy'] = {1:{'lr':1e-2}, 3:{'lr':1e-3}, 5:{'lr':1e-4}, 7:{'lr':1e-5}}

writer = SummaryWriter(comment="CIFAR_First_experiment")





"""===================== Training ======================================="""

# load the data
train_loader, val_loader, test_loader, data_mean, data_std = process_data(batch_size, val_split_at, img_height, img_width, num_channels)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'training on {device}')

# Instantiate the model and put on GPU
model = ConvolutionalModel(in_channels=3, conv1_width=16, conv2_width=32, fc1_width=256, fc2_width=128, class_count=10)
model.to(device)

# Training loop
num_epochs = config['max_epochs']
train_losses = []
test_losses = []

# Initialize variables to control printing
print_every = 100  # Print loss every 100 iterations
running_loss = 0.0  # Track loss for printing

criterion = nn.CrossEntropyLoss()
lr_policy = config['lr_policy']
solver_config = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=solver_config, weight_decay=config['weight_decay'])
count = 1

# plotting
plot_data = {}
plot_data['train_loss'] = []
plot_data['valid_loss'] = []
plot_data['train_acc'] = []
plot_data['valid_acc'] = []
plot_data['lr'] = []

# def multiclass_hinge_loss(logits: torch.Tensor, target: torch.Tensor, delta=1.0):

#     batch_size = logits.size(0) # 50

#     # Extract the scores for the correct classes
#     correct_scores = logits[torch.arange(batch_size), target].view(-1, 1) # [50, 1]

#     # Compute the margins (difference between correct scores and other scores)
#     margins = logits - correct_scores + delta # [50, 10]

#     # Set the correct class margins to zero
#     margins[torch.arange(batch_size), target] = 0
#     # pdb.set_trace()
    
#     # Calculate the hinge loss
#     loss = torch.sum(torch.max(margins, torch.zeros_like(margins))) / batch_size

#     return loss

def multiclass_hinge_loss(logits, target, n_classes, delta=1.2):
    # Convert target labels to one-hot encoding
    label_one_hot = torch.nn.functional.one_hot(target, num_classes=n_classes).float() # [50, 10]
    # Create a mask to exclude the correct class for each example
    mask = 1 - label_one_hot # [50, 10]

    # Compute the logits of the correct class for each example
    # The result is a diagonal matrix with correct class logits.
    correct_logits = torch.diag(torch.sum(label_one_hot * logits, dim=1)) # [50, 50] [batch_size, batch_size]
    # This line creates a matrix where each row contains the correct logits for the corresponding example.
    correct_logits_matrix = torch.mm(correct_logits, torch.ones_like(logits)) # [50, 10] = [50, 50] x [50, 10]  # [batch_size, num_classes] = [batch_size, batch_size] x [batch_size, num_classes]

    # Compute the hinge loss for each class
    errors = torch.relu(logits - correct_logits_matrix + delta) # [50, 10]
    
    # Compute the loss per sample and mask out losses for correct predictions
    sample_losses = torch.sum(errors * mask, dim=1) # [50]
    
    # Calculate the average loss across samples in the batch
    loss = torch.mean(sample_losses)  # scalar
    # pdb.set_trace()
    return loss # [50, 10]

for epoch in range(num_epochs):

    if epoch in lr_policy:
      solver_config = lr_policy[epoch]['lr']
      optimizer = torch.optim.SGD(model.parameters(), lr=solver_config, weight_decay=config['weight_decay'])

      print("======================================================================================")
      print(f'learning rate set to {solver_config}')
      print("======================================================================================")

    train_loss = 0.0
    
    # Training
    model.train()
    for i, (images, labels) in enumerate(train_loader, 1):

        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        # # pdb.set_trace()
        # loss = criterion(outputs, labels)
        loss = multiclass_hinge_loss(outputs, labels, 10)
        # pdb.set_trace()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)

        # Track running loss
        running_loss += loss.item()

        # Print loss every 'print_every' iterations
        if i % print_every == 0:
            conv1_weights = model.conv1.weight
            avg_loss = running_loss / print_every
            count = save_data.save_filter(epoch, conv1_weights, count, avg_loss)
            print(f'Epoch [{epoch + 1}/{num_epochs}], Iteration [{i*batch_size}], Average Loss: {avg_loss:.4f}')
            running_loss = 0.0  # Reset running loss
        


    """===================== Validation ======================================="""

    model.eval()
    val_loss = 0.0
    val_correct = 0
    with torch.no_grad():
        for images, labels in val_loader:
            # labels = labels.long()
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            # loss = criterion(outputs, labels)
            loss = multiclass_hinge_loss(outputs, labels, 10)

            val_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
    
    # Compute validation loss and accuracy
    val_loss /= len(val_loader.dataset)
    val_accuracy = val_correct / len(val_loader.dataset)

    # Calculate average losses
    train_loss = train_loss / len(train_loader.dataset)
    
    # Training accuracy and metrics
    train_accuracy, train_confusion, train_precision, train_recall = helper.evaluate(train_loader, model, device, criterion)
    val_accuracy, val_confusion, val_precision, val_recall = helper.evaluate(val_loader, model, device, criterion)

    plot_data['train_loss'] += [train_loss]
    plot_data['valid_loss'] += [val_loss]
    plot_data['train_acc'] += [train_accuracy]
    plot_data['valid_acc'] += [val_accuracy]
    plot_data['lr'] += [solver_config]


    """====================================== Logging info to tensorboard ================================================"""

    writer.add_scalar('Loss/validation', val_loss, global_step=epoch)
    
    # Log accuracy and metrics to TensorBoard
    writer.add_scalar('Accuracy/Training', train_accuracy, global_step=epoch)
    writer.add_scalar('Accuracy/Validation', val_accuracy, global_step=epoch)
    

    """====================================== Logging info to tensorboard ================================================"""

    
    print("======================================================================================")
    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, val Loss: {val_loss:.4f}')
    print("======================================================================================")

# save the data
save_data.plot_training_progress(SAVE_DIR, plot_data)




"""===================== Testing ======================================="""

model.eval()
test_loss = 0.0
test_correct = 0
all_preds_test = []
all_targets_test = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        # loss = criterion(outputs, labels)
        loss = multiclass_hinge_loss(outputs, labels, 10)

        test_loss += loss.item() * images.size(0)

        _, predicted = torch.max(outputs, 1)
        test_correct += (predicted == labels).sum().item()

        all_preds_test.extend(predicted.cpu().numpy())
        all_targets_test.extend(labels.cpu().numpy())
        confusion_mat_test = confusion_matrix(all_targets_test, all_preds_test)

    test_loss /= len(test_loader.dataset)
    test_accuracy = test_correct / len(test_loader.dataset)

    correct_per_class = confusion_mat_test.diagonal()

    # Create a dictionary to map class indices to correct predictions
    class_correct_dict = {class_index: correct_count for class_index, correct_count in enumerate(correct_per_class)}

    # Sort the classes by their correct predictions in descending order
    sorted_classes = sorted(class_correct_dict, key=class_correct_dict.get, reverse=True)

    # Display the top 3 classes with the most correct predictions 
    top_3_classes = sorted_classes[:3]
    print('test confusion matrix', confusion_mat_test)
    print("Top 3 correctly predicted classes:")
    for class_index in top_3_classes:
        print(f"Class {class_index}: Correct Predictions = {class_correct_dict[class_index]}")
        
    print("======================================================================================")
    print(f'Test Loss: {test_loss:.4f}, test Accuracy: {test_accuracy:.4f}')
    print("======================================================================================")



"""===================== Testing to estimate the wrong classes based on loss======================================="""


model.eval()
incorrect_images = {}  # Store information about incorrectly classified images
incorrect_images['images'] = []
incorrect_images['labels'] = []
incorrect_images['predicted'] = []
incorrect_images['losses'] = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        for i in range(images.size(0)): # for each image of one batch
          image = images[i].unsqueeze(0)
          label = labels[i].unsqueeze(0)
          output = model(image)
            #   loss_individual_item = criterion(output, label)
          loss_individual_item = multiclass_hinge_loss(outputs, labels, 10)
          
          loss_individual = loss_individual_item.item()
          loss += loss_individual

          _, predicted = torch.max(output, 1)
          if predicted != label:
                # Store information for incorrect predicted
                if len(incorrect_images['losses']) < 20:
                    incorrect_images['predicted'].append(predicted.item())
                    incorrect_images['images'].append(image.squeeze().cpu())
                    incorrect_images['labels'].append(label.item())
                    incorrect_images['losses'].append(loss)
                else:
                   # Replace the image with the least loss if the current loss is greater than the minimum loss in the list
                  if loss > min(incorrect_images['losses']):
                      min_loss_index = incorrect_images['losses'].index(min(incorrect_images['losses']))
                      incorrect_images['predicted'][min_loss_index] = predicted.item()
                      incorrect_images['images'][min_loss_index] = image.squeeze().cpu()
                      incorrect_images['labels'][min_loss_index] = label.item()
                      incorrect_images['losses'][min_loss_index] = loss
    
    
    # Undo normalization and visualize the image
    # for i in range(len(incorrect_images['losses'])):
    #   print("======================================================================================")
    #   print('incorrect class: ', incorrect_images['predicted'][i])
    #   print('Correct class: ', incorrect_images['labels'][i])

    #   img = incorrect_images['images'][i].cpu().numpy()
    #   save_data.draw_image(img, mean=data_mean, std=data_std)
    #   if i == 5:
    #      break

writer.close()
