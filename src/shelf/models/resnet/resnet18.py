import torch
import torch.nn as nn
import torch.nn.functional as F

from .bottleneck import BasicBlock, Bottleneck

class ResNet18(nn.Module):
    def __init__(self, input_size, num_output, input_channel=3):
        super(ResNet18, self).__init__()
        self.input_size = input_size if isinstance(input_size, tuple) else (input_size, input_size)
        self.input_channel = input_channel
        self.num_output = num_output

        # input: Tensor[batch_size, input_channel, input_size[0], input_size[1]]
        self.conv1 = nn.Conv2d(self.input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = nn.Sequential(
            BasicBlock(64, 64),
            BasicBlock(64, 64)
        )
        self.layer2 = nn.Sequential(
            BasicBlock(64, 128, stride=2),
            BasicBlock(128, 128)
        )
        self.layer3 = nn.Sequential(
            BasicBlock(128, 256, stride=2),
            BasicBlock(256, 256)
        )
        self.layer4 = nn.Sequential(
            BasicBlock(256, 512, stride=2),
            BasicBlock(512, 512)
        )

        # input: Tensor[batch_size, 512, input_size[0] // 32, input_size[1] // 32]
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * self.input_size[0] // 32 * self.input_size[1] // 32, self.num_output)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        hidden = self.maxpool(x)

        hidden = self.layer1(hidden)
        hidden = self.layer2(hidden)
        hidden = self.layer3(hidden)
        hidden = self.layer4(hidden)

        hidden = self.avgpool(hidden)
        hidden = torch.flatten(hidden, 1)
        x = self.fc(hidden)

        return x
    
