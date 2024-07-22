import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet_710(nn.Module):
    def __init__(self):
        super(ConvNet_710, self).__init__()
        
        # First Conv Block
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second Conv Block
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(4)
        
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        self.fc = nn.Linear(16, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc(x)
        return x
    

class ConvNet_3250(nn.Module):
    def __init__(self):
        super(ConvNet_3250, self).__init__()
        
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


class ConvNet_6694(nn.Module):
    def __init__(self):
        super(ConvNet_6694, self).__init__()
        
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


class ConvNet_9640(nn.Module):
    def __init__(self):
        super(ConvNet_9640, self).__init__()
        
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
        
        # Fifth Conv Block
        self.conv5 = nn.Conv2d(in_channels=24, out_channels=32, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, 10)
    
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
        
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1) # flatten
        x = self.fc(x)
        
        return x


class ConvNet_12282(nn.Module):
    def __init__(self):
        super(ConvNet_12282, self).__init__()
        # convolutional layer
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding = 1)
        self.conv3 = nn.Conv2d(32, 16, 3, padding = 1)
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layer
        self.fc1 = nn.Linear(16 * 4 * 4, 10)
        
        # Dropout
        self.dropout = nn.Dropout(p=0.2)
        
        # Output layer
        self.out = nn.LogSoftmax(dim = 1)

    def flatten(self, x):
        return x.view(x.size()[0], -1)
    
    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.dropout(self.pool(F.relu(self.conv1(x))))
        x = self.dropout(self.pool(F.relu(self.conv2(x))))
        x = self.dropout(self.pool(F.relu(self.conv3(x))))
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.out(x)
        return x


if __name__ == '__main__':
    model = ConvNet_710()
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters in ConvNet_710: {num_params}")

    model = ConvNet_3250()
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters in ConvNet_3250: {num_params}")

    model = ConvNet_6694()
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters in ConvNet_6694: {num_params}")

    model = ConvNet_9640()
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters in ConvNet_9640: {num_params}")

    model = ConvNet_12282()
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters in ConvNet_12282: {num_params}")
