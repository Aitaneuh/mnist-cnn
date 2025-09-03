import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # Convolutions
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)   # [1,28,28] -> [32,28,28]
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # [32,28,28] -> [64,28,28]
        self.pool = nn.MaxPool2d(2,2)  # réduit de moitié
        # Fully connected
        self.fc1 = nn.Linear(64*7*7, 128)  # après 2x pool -> [64,7,7]
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # [32,14,14]
        x = self.pool(F.relu(self.conv2(x)))  # [64,7,7]
        x = x.view(-1, 64*7*7)                # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
