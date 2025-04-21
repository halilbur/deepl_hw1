import torch.nn as nn
import torch.nn.functional as F

class BCICNN(nn.Module):
    def __init__(self, num_classes=4):
        super(BCICNN, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
                
        feature_size = 32 * 32 * 64 # for input 256x256
        
        # Dense layers
        self.fc1 = nn.Linear(feature_size, 64)
        
        self.fc2 = nn.Linear(64, 128)
        
        self.fc3 = nn.Linear(128, 256)
        
        # Output layer
        self.fc4 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))
        
        # Flatten the output for the dense layers
        x = x.view(x.size(0), -1)
        
        # Dense layers with ReLU activation
        x = F.relu(self.fc1(x))
        
        x = F.relu(self.fc2(x))
        
        x = F.relu(self.fc3(x))
        
        x = self.fc4(x)
        
        return x