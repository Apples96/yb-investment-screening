import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(1, 32, 3, 1)  # 1 input channel, 32 output channels, 3x3 kernel
        # Second convolutional layer
        self.conv2 = nn.Conv2d(32, 64, 3, 1)  # 32 input channels, 64 output channels, 3x3 kernel
        # Dropout layer to prevent overfitting
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        # Fully connected layers
        self.fc1 = nn.Linear(1600, 128)  # 1600 = 64 * 5 * 5
        self.fc2 = nn.Linear(128, 10)     # 10 output classes (digits 0-9)

    def forward(self, x):
        # First conv layer + ReLU + max pooling
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        # Second conv layer + ReLU + max pooling
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        
        # Flatten the output for the fully connected layer
        x = torch.flatten(x, 1)
        
        # First fully connected layer
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        # Output layer
        x = self.fc2(x)
        
        # Return log softmax
        output = F.log_softmax(x, dim=1)
        return output