import torch
import torch.nn as nn
import torch.nn.functional as F

from .bottleneck import BasicBlock, Bottleneck

class ResNet101(nn.Module):
    def __init__(self, input_size, num_output, input_channel=3):
        super(ResNet101, self).__init__()
        self.input_size = input_size if isinstance(input_size, tuple) else (input_size, input_size)
        self.input_channel = input_channel
        self.num_output = num_output

        # input: Tensor[batch_size, input_channel, input_size[0], input_size[1]]
        self.conv1 = nn.Conv2d(self.input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = nn.Sequential(
            Bottleneck(64, 64),
            Bottleneck(256, 64),
            Bottleneck(256, 64)
        )
        self.layer2 = nn.Sequential(
            Bottleneck(256, 128, stride=2),
            Bottleneck(512, 128),
            Bottleneck(512, 128),
            Bottleneck(512, 128)
        )
        self.layer3 = nn.Sequential(
            Bottleneck(512, 256, stride=2),
            Bottleneck(1024, 256),
            Bottleneck(1024, 256),
            Bottleneck(1024, 256),
            Bottleneck(1024, 256),
            Bottleneck(1024, 256),
            Bottleneck(1024, 256),
            Bottleneck(1024, 256),
            Bottleneck(1024, 256),
            Bottleneck(1024, 256),
            Bottleneck(1024, 256),
            Bottleneck(1024, 256),
            Bottleneck(1024, 256),
            Bottleneck(1024, 256),
            Bottleneck(1024, 256),
            Bottleneck(1024, 256),
            Bottleneck(1024, 256),
            Bottleneck(1024, 256),
            Bottleneck(1024, 256),
            Bottleneck(1024, 256),
            Bottleneck(1024, 256),
            Bottleneck(1024, 256),
            Bottleneck(1024, 256)
        )
        self.layer4 = nn.Sequential(
            Bottleneck(1024, 512, stride=2),
            Bottleneck(2048, 512),
            Bottleneck(2048, 512)
        )


        # input: Tensor[batch_size, 2048, input_size[0] // 32, input_size[1] // 32]
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048 * self.input_size[0] // 32 * self.input_size[1] // 32, self.num_output)

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
    
