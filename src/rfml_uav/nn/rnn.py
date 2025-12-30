import torch.nn as nn
import torch.nn.functional as F

class RNN(nn.Module):
    def __init__(self, input_dim: int, n_classes: int):
        super(RNN, self).__init__()
             
        # LSTM layers
        self.lstm1 = nn.LSTM(input_dim, 64, batch_first=True)
        self.lstm2 = nn.LSTM(64, 64, batch_first=True)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64, 512)
        self.bn1 = nn.BatchNorm1d(512)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        
        self.fc4 = nn.Linear(128, n_classes)
        self.bn4 = nn.BatchNorm1d(n_classes)
        
        # Activation functions (PReLU)
        self.prelu = nn.PReLU(init=0.25)

    def forward(self, x):
        # LSTM layers
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        
        # Flatten the output of LSTM
        x = x[:, -1, :]  # Take the output of the last time step
        
        # Fully connected layers with Batch Normalization and PReLU activation
        x = self.prelu(self.bn1(self.fc1(x)))
        x = self.prelu(self.bn2(self.fc2(x)))
        x = self.prelu(self.bn3(self.fc3(x)))
        
        # Output layer
        x = self.bn4(self.fc4(x))
        
        # Softmax
        return F.softmax(x, dim=1)