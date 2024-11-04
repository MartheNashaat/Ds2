import torch
from torch import nn
from torch.nn import functional as F

class CustomCNN(nn.Module):
    def _init_(self):
        super()._init_()
        
        # First convolutional block
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Second convolutional block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Third convolutional block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Fully connected layer
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.bn4 = nn.BatchNorm1d(256)
        
        # Output layer
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.dropout(F.relu(self.bn1(self.conv1(x))), p=0.25, training=self.training))
        x = self.pool(F.dropout(F.relu(self.bn2(self.conv2(x))), p=0.25, training=self.training))
        x = self.pool(F.dropout(F.relu(self.bn3(self.conv3(x))), p=0.25, training=self.training))
        x = x.view(-1, 128 * 4 * 4)
        x = F.dropout(F.relu(self.bn4(self.fc1(x))), p=0.5, training=self.training)
        x = self.fc2(x)
        return x