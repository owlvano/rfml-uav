import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, input_dim: int, n_classes: int):
        super(CNN, self).__init__()

        # Convolutional layers with Batch Normalization and ReLU activation
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=11)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=5)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(16)
        self.pool3 = nn.MaxPool1d(2)

        # Calculate the output size of the convolutional layers
        self._to_linear = self._get_conv_output(input_dim)

        # Fully connected layers
        self.fc1 = nn.Linear(self._to_linear, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.prelu1 = nn.PReLU(init=0.25)

        self.fc2 = nn.Linear(512, 256)
        self.bn5 = nn.BatchNorm1d(256)
        self.prelu2 = nn.PReLU(init=0.25)

        self.fc3 = nn.Linear(256, 128)
        self.bn6 = nn.BatchNorm1d(128)
        self.prelu3 = nn.PReLU(init=0.25)

        self.fc4 = nn.Linear(128, n_classes)
        self.bn7 = nn.BatchNorm1d(n_classes)
        self.prelu4 = nn.PReLU(init=0.25)

    def _get_conv_output(self, input_dim):
        # Create a dummy tensor to pass through the convolutional layers
        x = torch.zeros(1, 1, input_dim)  # (batch_size, num_channels, sequence_length)

        # Pass the dummy input through the conv layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)

        # Flatten the output to compute the number of features for the fully connected layer
        x = torch.flatten(x, 1)
        return x.size(1)  # Return the flattened size (number of features)

    def forward(self, x):
        # Apply convolutional layers with ReLU, BatchNorm, and MaxPool
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)

        # Flatten the output from Conv layers to feed into FC layers
        x = torch.flatten(x, 1)

        # Apply fully connected layers with PReLU and BatchNorm
        x = self.fc1(x)
        x = self.bn4(x)
        x = self.prelu1(x)

        x = self.fc2(x)
        x = self.bn5(x)
        x = self.prelu2(x)

        x = self.fc3(x)
        x = self.bn6(x)
        x = self.prelu3(x)

        x = self.fc4(x)
        x = self.bn7(x)
        x = self.prelu4(x)

        # Softmax activation
        x = F.softmax(x, dim=1)

        return x
