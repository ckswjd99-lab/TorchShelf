import torch
import torch.nn as nn

class ConvNet770(nn.Module):
    def __init__(self):
        super(ConvNet770, self).__init__()
        
        # Conv Block
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(4)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Average Pooling
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Fully Connected Layer
        self.fc = nn.Linear(4 * 4 * 4, 10)  # CIFAR-10 has 10 classes

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc(x)
        return x
    
class ConvNet3250(nn.Module):
    def __init__(self):
        super(ConvNet3250, self).__init__()
        
        # First Conv Block
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(6)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second Conv Block
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=8, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(8)
        
        # Third Conv Block
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=10, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(10)
        
        # Fourth Conv Block
        self.conv4 = nn.Conv2d(in_channels=10, out_channels=18, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(18)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(18, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc(x)
        return x
    
class ConvNet6694(nn.Module):
    def __init__(self):
        super(ConvNet6694, self).__init__()

        # First Conv Block
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second Conv Block
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=12, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(12)
        
        # Third Conv Block
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=16, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(16)
        
        # Fourth Conv Block
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=24, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(24)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(24, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc(x)
        return x
