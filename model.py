import torch
import torch.nn as nn
import torch.nn.functional as F

class BCICNN(nn.Module):
    def __init__(self, num_classes=4):
        super(BCICNN, self).__init__()
        
        # Convolutional layers as required: 4 layers with 8, 16, 32, 64 feature maps
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate the size of feature maps after convolutions and pooling
        # Input: 224x224 -> After 4 max pooling layers with stride 2: 14x14
        feature_size = 14 * 14 * 64
        
        # Dense layers as required: 3 layers with 64, 128, 256 neurons
        self.fc1 = nn.Linear(feature_size, 256)
        self.dropout1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.5)
        
        self.fc3 = nn.Linear(128, 64)
        self.dropout3 = nn.Dropout(0.5)
        
        # Output layer
        self.fc4 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # Convolutional layers with ReLU activation and max pooling
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        
        # Flatten the output for the dense layers
        x = x.view(x.size(0), -1)
        
        # Dense layers with ReLU activation and dropout
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)
        
        # Output layer (no activation as we'll use CrossEntropyLoss)
        x = self.fc4(x)
        
        return x