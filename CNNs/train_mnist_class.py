#!/usr/bin/python3
import os
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

class CovolutionalModel(nn.Module):
    def __init__(self, in_channels, conv1_width, conv2_width, fc1_width, class_count):
        super(CovolutionalModel, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, conv1_width, kernel_size=5, stride=1, padding=2, bias=True)
        self.conv2 = nn.Conv2d(conv1_width, conv2_width, kernel_size=5, stride=1, padding=2, bias=True)
        
        self.fc1_input_dim = conv2_width * 7 * 7  # Output from conv2 is 32*7x7
        self.fc1 = nn.Linear(self.fc1_input_dim, fc1_width, bias=True) # (32*7x7, 512)
        self.fc_logits = nn.Linear(fc1_width, class_count, bias=True)  # (512, 10)      

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.relu = nn.ReLU() # self.relu(input)

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

    def forward(self, x): # x = (N, C, H, W) input image
        
        h = self.conv1(x)
        h = torch.relu(h)
        h = self.pool(h)

        h = self.conv2(h)
        h = torch.relu(h)
        h = self.pool(h)

        h = h.view(h.size(0), -1)  # Flatten the output before FC layers
        h = self.fc1(h)
        h = torch.relu(h)
        logits = self.fc_logits(h)
        return logits

def create_save_directory():
    SAVE_DIR = Path(__file__).parent / 'pytorch_mnist_out'
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    return SAVE_DIR

def prepare_config(SAVE_DIR):
    config = {}
    config['max_epochs'] = 8
    config['batch_size'] = 50
    config['save_dir'] = SAVE_DIR
    config['weight_decay'] = 1e-3
    config['lr_policy'] = {1:{'lr':1e-1}, 3:{'lr':1e-2}, 5:{'lr':1e-3}, 7:{'lr':1e-4}}

    return config

def prepare_data_loaders(config):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = MNIST(root='./data', train=False, download=True, transform=transform)

    train_dataset_size = len(train_dataset)
    val_split_index = 55000
    indices = list(range(train_dataset_size))
    train_indices, val_indices = indices[:val_split_index], indices[val_split_index:]

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(train_dataset, val_indices)

    batch_size = config['batch_size']
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def save_filter(writer, epoch, conv1_weights, count, avg_loss):
    writer.add_scalar('Loss/train', avg_loss, global_step=count)
    conv1_weights_normalized = (conv1_weights - conv1_weights.min()) / (conv1_weights.max() - conv1_weights.min())

    empty_image = torch.zeros(11, 47)
    for i in range(conv1_weights_normalized.shape[0]):
        row = i // 8
        col = i % 8
        empty_image[row * 6: row * 6 + 5, col * 6: col * 6 + 5] = conv1_weights_normalized[i][0]

    writer.add_image('epoch_' + str(epoch) + '_Conv1_filters', empty_image.unsqueeze(0), global_step=count)
    count += 1
    return count

def train_model(model, train_loader, val_loader, config, device, writer):
    
    print_every = 100  # Print loss every 100 iterations
    running_loss = 0.0  # Track loss for printing

    lr_policy = config['lr_policy']
    num_epochs = config['max_epochs']
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=config['weight_decay'])
    count = 1

    for epoch in range(num_epochs):
        if epoch in lr_policy:
            solver_config = lr_policy[epoch]['lr']
            optimizer = torch.optim.SGD(model.parameters(), lr=solver_config, weight_decay=config['weight_decay'])
            print("======================================================================================")
            print(f'learning rate set to {solver_config}')
            print("======================================================================================")

        train_loss = 0.0
        model.train()
        for i, (images, labels) in enumerate(train_loader, 1):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

            # Track running loss
            running_loss += loss.item()

            # Print loss every 'print_every' iterations
            if i % print_every == 0:
                conv1_weights = model.conv1.weight
                avg_loss = running_loss / print_every
                count = save_filter(writer, epoch, conv1_weights, count, avg_loss)
                print(f'Epoch [{epoch + 1}/{num_epochs}], Iteration [{i}/{len(train_loader)}], Average Loss: {avg_loss:.4f}')
                running_loss = 0.0  # Reset running loss
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)

                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
        
        # Compute validation loss and accuracy
        val_loss /= len(val_loader.dataset)
        val_accuracy = val_correct / len(val_loader.dataset)

        # Log validation loss and accuracy to TensorBoard
        writer.add_scalar('Loss/validation', val_loss, global_step=epoch)
        writer.add_scalar('Accuracy/validation', val_accuracy, global_step=epoch)

        # Calculate average losses
        train_loss = train_loss / len(train_loader.dataset)

        print("======================================================================================")
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, val Loss: {val_loss:.4f}')
        print("======================================================================================")
        
def test_model(model, test_loader, criterion, device):
    model.eval()
    test_loss, test_correct = 0.0, 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)
            test_correct += (predicted == labels).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = test_correct / len(test_loader.dataset)

    print("======================================================================================")
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
    print("======================================================================================")

def main():
    
    SAVE_DIR = create_save_directory()
    config = prepare_config(SAVE_DIR)
    train_loader, val_loader, test_loader = prepare_data_loaders(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Training on {device}')

    model = CovolutionalModel(in_channels=1, conv1_width=16, conv2_width=32, fc1_width=512, class_count=10)
    model.to(device)
    writer = SummaryWriter(comment="First_experiment")

    train_model(model, train_loader, val_loader, config, device, writer)
    test_model(model, test_loader, nn.CrossEntropyLoss(), device)
    
    writer.close()

if __name__ == "__main__":
    main()
